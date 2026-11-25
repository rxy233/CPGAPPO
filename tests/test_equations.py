"""Deterministic fixtures for the equations and boundary conditions."""
import math
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.constant import para
from Environment.environment import Environment
from Environment import computation
from scheduler.task_scheduler import TaskScheduler
from scheduler.graph_scheduler import GraphScheduler
from Algorithms.RealGATPPO.cpgappo_core import extract_cpgappo_state, compute_cpgappo_slack_reward
from Algorithms.RealGATPPO.agent_cpgappo import GAT_PPO_Agent_CPGAPPO, compute_guide_scores, compute_guide_scores_cp
from Algorithms.Train.train_cpgappo_unified import compute_cp_factor, build_cpgappo_safe_mask, application_credit, terminal_credit
from experiments.exp_utils import calc_timeout_rate, compute_score, generate_arrival_plan


class EquationTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(para)
        para['user_num'] = 2
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.env = Environment(2, 20, 60, 0)
        self.env.generate_components(seed=1)
        graphs = [nx.DiGraph([(0, 1), (0, 2)]), nx.DiGraph([(0, 1), (1, 2)])]
        self.env.subgraph_list = [SimpleNamespace(nx_graph=g, enter_nodes=[0], exit_nodes=[n for n in g if not g.out_degree(n)]) for g in graphs]
        self.ts = TaskScheduler(2, 3, 60, self.env, tasksize=[102400], seed=1,
                                tight_deadline_config={'deadline_slot_per_user': [55, 33]})
        self.ts.reset()
        self.ts.new_arrival(2, 0, fixed_uids=[0, 1])
        self.gs = GraphScheduler(self.env.basegraph, self.env.subgraph_list, self.ts, device='cpu')

    def tearDown(self):
        para.clear();para.update(self.saved)

    def test_eq6_7_deadlines(self):
        self.assertEqual(self.ts.get_task_deadline_slot(1), 33)
        self.assertEqual(self.ts.get_app_deadline_slot(0), min(math.floor(.3*1.05*55*2), math.floor(.3*55*2)))
        self.assertEqual(self.ts.get_app_deadline_slot(1), min(math.floor(.3*1.05*55*3), math.floor(.3*33*3)))

    def test_eq8_14_transmission(self):
        size, distance, bw, power = 102400, 100, 20, 2.0
        e, delay = computation.upload_consumption([size, distance, bw], 0, 'e', local_trans=power, local_wait=123)
        rate = bw*1e6*math.log2(1+power*distance**(-para['theta'])/para['yita2'])
        self.assertAlmostEqual(delay, 8*size/rate)
        self.assertAlmostEqual(e, power*delay)
        wired_e, wired_t = computation.upload_consumption(size, 0, 'c')
        self.assertEqual(wired_e, 0)
        self.assertAlmostEqual(wired_t, 8*size/para['R_ec'])

    def test_local_execution_energy(self):
        size, f = 102400, 4e8
        e, t = computation.execute_consumption(size, f, 0, 'l', local_wait=999)
        self.assertAlmostEqual(t, size*320/f)
        self.assertAlmostEqual(e, para['kappa']*f**3*t)
        self.assertEqual(computation.execute_consumption(size, 7e9, 0, 'c')[0], 0)

    def test_eq1_4_dual_pipeline(self):
        self.ts.update((0,0), .1, .025, local=True, exec_duration=.025)
        mask=self.ts.get_action_mask(0, 100)
        self.assertFalse(mask[0]);self.assertTrue(mask[1])
        self.ts.check_timeouts(1)
        self.assertAlmostEqual(max(0,self.ts.devices_exe_useful[0]-.01), .015)
        self.ts.check_timeouts(3)
        self.assertTrue(self.ts.get_action_mask(0, 100)[0])

    def test_same_slot_flag_even_zero_duration(self):
        self.ts.update((0,0), 0, 0, local=True, exec_duration=0)
        self.assertFalse(self.ts.get_action_mask(0, 100)[0])
        self.ts.check_timeouts(1)
        self.assertTrue(self.ts.get_action_mask(0, 100)[0])

    def test_readiness_and_recorded_start(self):
        self.assertTrue(math.isinf(self.ts._calculate_est(0,1)))
        self.ts.finish_time[0][0]=.015
        self.ts.start_time[0][1]=.015
        self.ts.check_timeouts(3)
        self.assertAlmostEqual(self.ts.calculate_start_time(0,1,True), .015)
        self.assertNotIn((0,2), self.gs.get_tasks(1))

    def test_eq9_edge_queue_and_release(self):
        edge=0
        self.ts.edge_core_finish[edge]=[.1,.2,.3,.4]
        self.ts.edge_update((0,0),102400,(.02,.01),(0,.03),edge)
        self.assertAlmostEqual(self.ts.finish_time[0][0], .13)
        self.assertAlmostEqual(self.ts.devices_upload_useful[0], .01)
        self.assertEqual(self.env.edges[edge].used_storage,102400)
        self.ts.check_timeouts(14)
        self.assertEqual(self.env.edges[edge].used_storage,0)
        self.assertEqual(self.env.edges[edge].total_difficulty,0)

    def test_eq10_cloud_time_and_energy(self):
        d=self.env.device_list[0];size=102400
        distance=min(d.edge_distances)
        up_e,up_t=computation.upload_consumption([size,distance,d.local_bw],0,'e',local_trans=d.local_trans)
        _,wire_t=computation.upload_consumption(size,0,'c')
        _,exec_t=computation.execute_consumption(size,self.env.cloud.cloud_power,0,'c')
        energy,delay=self.ts.add_task(0,0,1)
        self.assertAlmostEqual(energy,up_e)
        self.assertAlmostEqual(delay,up_t+wire_t+exec_t)
        self.assertAlmostEqual(self.ts.finish_time[0][0],delay)

    def test_eq17_18_ready_cp_and_rank(self):
        self.gs.rank_u={0:{0:.1,1:999,2:999},1:{0:.2,1:.1,2:.01}}
        self.assertAlmostEqual(self.gs.get_slack_cp(0,0),.33-.1)
        self.assertEqual(self.gs.get_tasks(0,sort_tasks=True),[(1,0),(0,0)])
        self.gs.rank_u[0][0]=2;self.gs.rank_u[1][0]=1
        self.assertEqual(self.gs.get_tasks(0,sort_tasks=True),[(1,0)])

    def test_eq19_global_graph_and_target(self):
        data,g,mask=extract_cpgappo_state(self.ts,(0,0),0,0)
        self.assertEqual(data.x.shape,(6,27));self.assertEqual(g.shape,(10,))
        self.assertEqual(int(data.x[:,-1].sum()),1)
        self.assertEqual(data.edge_index.shape,(2,4))
        self.ts.enter_time[1]=float('inf')
        smaller,_,_=extract_cpgappo_state(self.ts,(0,0),0,0)
        self.assertEqual(smaller.num_nodes,3)
        self.assertTrue(torch.isfinite(data.x).all())

    def test_eq20_21_model_dimensions(self):
        data,g,_=extract_cpgappo_state(self.ts,(0,0),0)
        for backward in (True,False):
            model=GAT_PPO_Agent_CPGAPPO(use_backward=backward).policy
            logits,value=model(Batch.from_data_list([data,data]),torch.stack([g,g]))
            self.assertEqual(logits.shape,(2,10));self.assertEqual(value.shape,(2,1))
            self.assertTrue(torch.isfinite(logits).all())

    def test_eq22_empty_mask_deferred(self):
        self.ts.devices_exe_useful[0]=1;self.ts.devices_upload_useful[0]=1
        data,g,mask=extract_cpgappo_state(self.ts,(0,0),0)
        self.assertFalse(any(mask))
        with self.assertRaises(ValueError):
            GAT_PPO_Agent_CPGAPPO().take_action(data,g,mask)

    def test_eq22_unreachable_edge(self):
        self.env.device_list[0].edge_distances=[float('inf')]*8
        mask=self.ts.get_action_mask(0,102400)
        self.assertTrue(mask[0]);self.assertFalse(mask[1:].any())

    def test_eq23_26_cp_guide(self):
        self.assertEqual(compute_cp_factor(self.ts,0,0),.25)
        self.assertEqual(compute_cp_factor(self.ts,0,1),0)
        self.ts.check_timeouts(30)
        base,valid=compute_guide_scores(self.ts,0,(0,0),0,.3)
        guided,valid_cp=compute_guide_scores_cp(self.ts,0,(0,0),0,.3)
        remaining=.45*.25*self.ts.get_app_deadline_slot(0)*.01
        expected=base[valid]+1.25*np.maximum(0,remaining-(.03-base[valid]))
        np.testing.assert_allclose(guided[valid],expected,rtol=1e-6)
        self.assertEqual(np.argmin(base[valid]),np.argmin(guided[valid]))

    def test_eq28_29_shield_and_fallback(self):
        mask=np.array([True,True]+[False]*8)
        scores=np.array([.1,.5]+[100]*8)
        safe=build_cpgappo_safe_mask(self.ts,0,0,mask,scores,cp_slack=.3)
        self.assertEqual(safe.tolist(),[True]+[False]*9)
        fallback=build_cpgappo_safe_mask(self.ts,0,0,mask,scores+10,cp_slack=.3)
        np.testing.assert_array_equal(fallback,mask)
        scores[1]=.33*1.12
        normal=build_cpgappo_safe_mask(self.ts,0,0,mask,scores,cp_slack=.3)
        high=build_cpgappo_safe_mask(self.ts,0,0,mask,scores,cp_slack=0)
        self.assertTrue(normal[1]);self.assertFalse(high[1])

    def test_appendix_reserved_features(self):
        self.ts.route_stats['local']=2
        data,_,_=extract_cpgappo_state(self.ts,(0,0),0)
        self.assertEqual(float(data.x[:,16:26].abs().sum()),0)

    def test_eq5_15_power_and_sla_objective(self):
        from system_objective import instantaneous_power,jsop_cost
        self.ts.devices_exe_useful[0]=1
        self.ts.devices_upload_useful[0]=1
        d=self.env.device_list[0]
        self.assertAlmostEqual(instantaneous_power(self.ts,0),para['kappa']*d.local_power**3+d.local_trans)
        self.ts.application_finished={0}
        self.ts.application_timeout_finished={0,1}
        self.ts.exit_time=[.5,.29]
        self.ts.energy=[2,4]
        self.assertAlmostEqual(jsop_cost(self.ts,1,2,3),.33+.29+2*6+3*2)

    def test_eq30_33_credit_and_slack(self):
        self.assertEqual(application_credit(True,True),4)
        self.assertEqual(application_credit(False,True),-7)
        for finish,penalty in ((.1,0),(.3,-1),(.5,-5-2*.17)):
            self.ts.finish_time[0][0]=finish
            r,_=compute_cpgappo_slack_reward(self.ts,0,0,2,.4,0,33)
            self.assertAlmostEqual(r,-.05*2-.1*.4+penalty)
        self.assertEqual(terminal_credit(0),1)
        self.assertEqual(terminal_credit(.2),-1)

    def test_eq34_masked_ppo_update(self):
        data,g,_=extract_cpgappo_state(self.ts,(0,0),0)
        mask=np.array([False,True]+[False]*8)
        for guide in (0,.1):
            agent=GAT_PPO_Agent_CPGAPPO(K_epochs=1,lambda_guide=guide)
            for reward in (1,-1):
                a,lp=agent.take_action(data,g,mask)
                self.assertEqual(a,1)
                agent.put_data((data,g,a,reward,lp,False,mask,1))
            self.assertTrue(math.isfinite(agent.update()))
            self.assertTrue(all(torch.isfinite(p).all() for p in agent.policy.parameters()))

    def test_eq36_38_late_completion_and_unfinished(self):
        self.ts.application_finished={0}
        self.ts.application_timeout_finished={0,1}
        self.ts.exit_time=[.5,.29]
        self.ts.finish_time[0]={0:.5,1:.56,2:.6}
        metrics=calc_timeout_rate(self.ts)
        self.assertEqual(metrics['app_timeout_rate'],.5)
        self.assertAlmostEqual(metrics['task_timeout_rate'],5/6)
        self.ts.energy=[2,4]
        energy,delay=self.ts.get_avg_results(timeout_charge='deadline')
        self.assertEqual(energy,3)
        self.assertAlmostEqual(delay,(.33+.29)/2)
        self.assertEqual(metrics['unfinished_apps'],0)

    def test_selected_cost_raw_minimum_and_threshold(self):
        self.assertEqual(compute_score(.3,.7,.05,0),0)
        self.assertAlmostEqual(compute_score(.3,.7,.1,0),20)
        self.assertEqual(compute_score(.3,.7,0,.5),1)
        self.assertLess(compute_score(.3,.7,.05,0),compute_score(.3,.7,.1,0))

    def test_arrival_separation_and_reset(self):
        a=generate_arrival_plan(1,200,150,.3,.2,27,55)
        self.assertEqual(a,generate_arrival_plan(1,200,150,.3,.2,27,55))
        self.assertNotEqual(a,generate_arrival_plan(21001,200,150,.3,.2,27,55))
        self.ts.route_stats['edge']=99;self.ts.remain_times[0]=100
        self.ts.reset()
        self.assertEqual(self.ts.route_stats['edge'],0)
        self.assertEqual(self.ts.remain_times[0],0)


if __name__=='__main__':
    unittest.main()
