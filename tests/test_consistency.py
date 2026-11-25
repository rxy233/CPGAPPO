"""Small deterministic checks of dispatch, arrivals, and metric semantics."""
import copy
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Algorithms.Benchmark import Benchmark
from Algorithms.Train.train_dynamic_gat_dqn import get_stable_active_users
from Algorithms.Train.train_cpgappo_unified import application_credit, terminal_credit
from Environment.environment import Environment
from experiments.exp_utils import CONFIG, calc_timeout_rate, generate_arrival_plan
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from system_objective import jsop_cost
from utils.constant import para


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.config = patch.dict(CONFIG, TOPK_MIN=15, TOPK_MAX=30)
        self.config.start()
        self.addCleanup(self.config.stop)
        self.gs = GraphScheduler.__new__(GraphScheduler)
        self.gs.ts = SimpleNamespace(
            env=SimpleNamespace(task_complex_index=0),
            enter_time=[0.0] * 60,
            task_size=np.full((60, 2), 100000.0),
            start_time=[{0: 0.0, 1: 0.0} for _ in range(60)],
            get_app_deadline_slot=lambda uid: 100,
        )
        self.gs.rank_u = {uid: {0: 0.1, 1: 0.1} for uid in range(60)}

    def test_adaptive_count_boundaries_and_stable_ties(self):
        for count, expected in [(0, 0), (1, 1), (15, 15), (16, 15),
                                (30, 15), (31, 30), (50, 30), (51, 30), (60, 30)]:
            with self.subTest(count=count):
                tasks = [(u, 0) for u in reversed(range(count))]
                original = tasks.copy()
                selected = self.gs.pick_topk_tasks(0, tasks, k=1)
                self.assertEqual(selected, tasks[:expected])
                self.assertEqual(tasks, original)

    def test_adaptive_limits_and_explicit_fixed_count(self):
        tasks = [(u, 0) for u in range(60)]
        with patch.dict(CONFIG, TOPK_MIN=7, TOPK_MAX=40,
                        ADAPTIVE_TOPK=False, SMART_TOPK=False):
            self.assertEqual(len(self.gs.pick_topk_tasks(0, tasks[:30], k=1)), 7)
            self.assertEqual(len(self.gs.pick_topk_tasks(0, tasks[:31], k=1)), 30)
            self.assertEqual(len(self.gs.pick_topk_tasks(0, tasks, k=1)), 40)
            self.assertEqual(len(self.gs.pick_topk_tasks(0, tasks, k=6, adaptive=False)), 6)

    def test_adjusted_slack_then_rank_then_start_time(self):
        tasks = [(0, 0), (1, 0), (2, 0), (3, 0)]
        self.gs.ts.task_size[1, 0] = 400000
        self.gs.rank_u[2][0] = 0.3
        self.gs.ts.start_time[3][0] = -0.01
        self.assertEqual(self.gs.pick_topk_tasks(0, tasks),
                         [(2, 0), (1, 0), (3, 0), (0, 0)])
        # Equal per-application slack and task sizes leave rank as the next key.
        self.gs.rank_u[0] = {0: 0.1, 1: 0.2}
        self.assertEqual(self.gs.pick_topk_tasks(0, [(0, 0), (0, 1)]),
                         [(0, 1), (0, 0)])

    def test_benchmark_applies_topk_to_each_nonlearning_strategy(self):
        tasks = [(u, 0) for u in range(35)]
        self.gs.get_tasks = Mock(return_value=tasks)
        for method in (0, 2, 4, 6, 7):
            with self.subTest(method=method):
                benchmark = Benchmark(SimpleNamespace(), self.gs, self.gs.ts, 0, seed=1)
                with patch.object(self.gs, 'pick_topk_tasks', return_value=[]) as select:
                    actions, _ = benchmark.get_actions(method, 0, 0)
                self.assertEqual(actions, [])
                select.assert_called_once_with(0, tasks, k=30)

    def test_gatdqn_filters_infeasible_tasks_before_topk(self):
        tasks = [(u, 0) for u in range(31)]
        self.gs.get_tasks = Mock(return_value=tasks)
        self.gs.ts.get_task_size_bytes = lambda uid, sid: 100000
        self.gs.ts.get_action_mask = lambda uid, size, now: torch.tensor([uid != 0])
        users, selected = get_stable_active_users(self.gs.ts, self.gs, 0, k=1)
        self.assertEqual(selected, tasks[1:16])
        self.assertEqual(users, list(range(1, 16)))


class ArrivalTests(unittest.TestCase):
    SEEDS = (1, 3, 5, 7, 42, 21001, 21002, 21003)

    def setUp(self):
        self.settings = patch.dict(para, user_num=150)
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def test_total_identity_repeatability_and_distinct_seeds(self):
        schedules = []
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                plan = generate_arrival_plan(seed, 8000, 2000, .3, .2, 27, 55)
                self.assertEqual(plan, generate_arrival_plan(seed, 8000, 2000, .3, .2, 27, 55))
                self.assertEqual(plan['plan'], [len(slot) for slot in plan['schedule']])
                self.assertEqual(sum(plan['plan']), 150)
                self.assertEqual(sorted(u for slot in plan['schedule'] for u in slot), list(range(150)))
                self.assertTrue(all(not slot for slot in plan['schedule'][2000:]))
                schedules.append(tuple(tuple(slot) for slot in plan['schedule']))
        self.assertEqual(len(set(schedules)), len(self.SEEDS))

    def test_initialization_and_forward_relocation(self):
        for seed in self.SEEDS:
            uniform = generate_arrival_plan(seed, 8000, 2000, .3, 0, 27, 55)
            burst = generate_arrival_plan(seed, 8000, 2000, .3, .2, 27, 55)
            positions = {uid: t for t, users in enumerate(uniform['schedule']) for uid in users}
            self.assertEqual(sorted(positions.values()), [int(i * 2000 / 150) for i in range(150)])
            for slot, users in enumerate(burst['schedule']):
                self.assertTrue(all(slot <= positions[uid] for uid in users))
            self.assertLessEqual(max(burst['plan']), 56)  # Relocated users plus the existing arrival.

    def test_base_prob_is_compatibility_only_and_rng_is_local(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        first = generate_arrival_plan(42, 8000, 2000, 0, .2, 27, 55)
        self.assertEqual(first, generate_arrival_plan(42, 8000, 2000, 1, .2, 27, 55))
        self.assertEqual(python_state, random.getstate())
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.settings = patch.dict(para, user_num=4)
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.env = Environment(4, 20, 60, 0)
        self.env.generate_components(seed=1)
        self.env.subgraph_list = [SimpleNamespace(nx_graph=nx.empty_graph(2, create_using=nx.DiGraph),
                                                 enter_nodes=[0, 1], exit_nodes=[0, 1]) for _ in range(4)]
        self.ts = TaskScheduler(4, 2, 60, self.env, tasksize=[102400], seed=1,
                                tight_deadline_config={'deadline_slot_per_user': [55] * 4})
        self.ts.reset()
        self.ts.new_arrival(3, 0, fixed_uids=[0, 1, 2])
        self.gs = GraphScheduler(self.env.basegraph, self.env.subgraph_list, self.ts, device='cpu')
        self.bc = Benchmark(self.env, self.gs, self.ts, 0, seed=1)

    def state(self):
        names = ('energy', 'total_energy', 'finish_time', 'start_time', 'rest_tasks',
                 'exit_time', 'devices_exe_useful', 'devices_upload_useful',
                 'pipeline_dispatch_times', 'edge_jobs', 'edge_core_finish',
                 'core_remaining_work', 'application_finished', 'application_timeout_finished')
        result = {n: copy.deepcopy(getattr(self.ts, n)) for n in names}
        result['storage'] = [e.used_storage for e in self.env.edges]
        return result

    def test_rejected_actions_leave_execution_and_resources_untouched(self):
        self.ts.devices_exe_useful[0] = 1.0
        self.ts.devices_upload_useful[0] = 1.0
        for action in (-1, 0, 1, *range(2, para['edge_num'] + 2), para['edge_num'] + 2):
            with self.subTest(action=action):
                before = self.state()
                with patch.object(self.ts, 'add_task', wraps=self.ts.add_task) as add:
                    reward, info = self.bc.step([[(0, 0), action]])
                add.assert_not_called()
                self.assertEqual(self.state(), before)
                self.assertEqual((reward, info['step_energy'], info['step_delay']), (0, 0, 0))

    def test_storage_reachability_and_cloud_checks(self):
        self.env.device_list[0].edge_distances = [1.0] * para['edge_num']
        edge = self.env.edges[0]
        edge.used_storage = edge.max_storage
        before = self.state()
        self.bc.step([[(0, 0), 2]])
        self.assertEqual(before, self.state())
        self.env.device_list[0].edge_distances = [float('inf')] * para['edge_num']
        for action in (1, 2):
            before = self.state()
            self.bc.step([[(0, 0), action]])
            self.assertEqual(before, self.state())
        self.env.device_list[0].edge_distances = [1.0] * para['edge_num']
        with patch.object(self.env.cloud, 'is_available', return_value=False):
            before = self.state()
            self.bc.step([[(0, 0), 1]])
            self.assertEqual(before, self.state())

    def test_batch_rechecks_resources_after_each_action(self):
        with patch.object(self.ts, 'add_task', wraps=self.ts.add_task) as add:
            self.bc.step([[(0, 0), 0], [(0, 1), 0]])
        self.assertEqual(add.call_count, 1)
        self.assertTrue(np.isfinite(self.ts.finish_time[0][0]))
        self.assertEqual(self.ts.finish_time[0][1], float('inf'))

    def test_both_completed_states_timeout_and_unarrived(self):
        budget = self.ts.get_app_deadline_slot(0) * para['slot_interval']
        for sid in (0, 1):
            self.ts.update((0, sid), .5, budget / 4, local=True, exec_duration=0)
            self.ts.update((1, sid), 1.0, 1.0, local=True, exec_duration=0)
        self.ts.finalize_episode(199)
        self.assertEqual(self.ts.application_finished, {0, 1})
        self.assertEqual(self.ts.application_timeout_finished, {1, 2})
        self.ts.energy[3] = 1000  # Unarrived applications do not enter averages.
        metrics = calc_timeout_rate(self.ts)
        self.assertAlmostEqual(metrics['app_timeout_rate'], 1 / 3)
        self.assertAlmostEqual(metrics['task_timeout_rate'], 2 / 3)
        self.assertEqual(metrics['subtask_stats']['total'], 6)
        self.assertEqual(metrics['app_timeout_count'], 2)
        self.assertEqual(metrics['app_finished_count'], 2)
        energy, delay = self.ts.get_avg_results(timeout_charge='deadline')
        self.assertAlmostEqual(energy, 1.0)
        self.assertAlmostEqual(delay, (budget / 4 + 2 * budget) / 3)
        self.assertEqual(jsop_cost(self.ts, 0, 0, 1), 2)
        self.assertEqual(application_credit(True, True), 4)
        self.assertEqual(application_credit(False, True), -7)
        self.assertEqual(terminal_credit(0), 1)
        self.assertAlmostEqual(terminal_credit(metrics['app_timeout_rate']), -5 / 3)

    def test_no_arrived_applications(self):
        self.ts.reset()
        metrics = calc_timeout_rate(self.ts)
        self.assertEqual(metrics['app_timeout_rate'], 0)
        self.assertEqual(metrics['task_timeout_rate'], 0)
        self.assertEqual(metrics['subtask_stats']['total'], 0)


if __name__ == '__main__':
    unittest.main()
