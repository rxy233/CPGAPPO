import os
import sys
import numpy as np
import torch
from pathlib import Path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from experiments.exp_utils import CONFIG, init_worker, generate_arrival_plan, load_deadline_config, load_arrival_plan, compute_score, calc_timeout_rate, safe_rest_tasks_total, all_arrived_done, get_graph_cache, apply_arrival_plan, subtask_partition_stats
from Environment.environment import Environment
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from Algorithms.Benchmark import Benchmark as BenchmarkClass
from utils.constant import para
from Algorithms.StrongBaselines.transedge_style_agent import TransEdgeStylePPOAgent
from Algorithms.StrongBaselines.transedge_style_core import extract_transedge_state
print_lock = None

def compute_structured_step_reward(app_timeout, app_done, task_timeout, energy_norm, delay_norm, action_val, route_stats, edge_feasible=True, local_feasible=True):
    if app_timeout:
        return -3.0
    if app_done:
        return 2.0 - 0.5 * energy_norm - 0.5 * delay_norm
    reward = 0.0
    if task_timeout:
        reward -= 0.8
    else:
        reward += 0.15
    reward += 0.08 * (1.0 - energy_norm)
    reward += 0.08 * (1.0 - delay_norm)
    if route_stats is not None:
        total = max(1, route_stats.get('local', 0) + route_stats.get('edge', 0) + route_stats.get('cloud', 0))
        local_ratio = route_stats.get('local', 0) / total
        edge_ratio = route_stats.get('edge', 0) / total
        cloud_ratio = route_stats.get('cloud', 0) / total
        if action_val == 1 and cloud_ratio > 0.7 and edge_feasible:
            reward -= 0.2 * ((cloud_ratio - 0.7) / 0.3)
        if action_val >= 2 and edge_ratio < 0.25 and edge_feasible:
            reward += 0.1 * ((0.25 - edge_ratio) / 0.25)
        if action_val == 0 and local_ratio < 0.1 and local_feasible:
            reward += 0.04
    return reward

def set_print_lock(lock):
    global print_lock
    print_lock = lock

def eval_transedge_style_once(agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS, use_heuristic_sort=False):
    user_num = para['user_num']
    subgraph_num = 20
    basegraph_num = 60
    env = ts.env
    env_seed = CONFIG['SEED'] if 'SEED' in CONFIG else 0
    deadline_config = load_deadline_config(CONFIG.get('RUN_DIR', ''))
    ts_eval = TaskScheduler(user_num, subgraph_num, basegraph_num, env, tight_deadline_config=deadline_config, seed=env_seed)
    gs_eval = GraphScheduler(env.basegraph, env.subgraph_list, ts_eval)
    bc_eval = BenchmarkClass(env, gs_eval, ts_eval, task_complex_index, effective=True, seed=env_seed)
    ts_eval.env = env
    ts_eval.using_Algorithm = -1
    bc_eval.reset()
    ts_eval.reset()
    ts_eval.route_stats = {'local': 0, 'edge': 0, 'cloud': 0, 'recent': []}
    agent.policy.eval()
    eval_task2action = {}
    eval_decision_count = 0
    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts_eval, slot, arrival_plan)
        ts_eval.check_timeouts(slot)
        tasks = gs_eval.get_tasks(slot, sort_tasks=use_heuristic_sort)
        if not tasks:
            if slot >= CONFIG['STOP_ARRIVAL_STEP'] and safe_rest_tasks_total(ts_eval.rest_tasks) == 0 and all_arrived_done(ts_eval):
                break
            continue
        for task in tasks:
            s_data, mask = extract_transedge_state(ts_eval, task, slot=slot, task_complex_index=task_complex_index)
            if not np.any(mask):
                continue
            act, _ = agent.take_action(s_data, action_mask=mask, deterministic=True, route_stats=getattr(ts_eval, 'route_stats', None))
            bc_eval.step([[task, act]])
            eval_task2action[tuple(task)] = int(act)
            eval_decision_count += 1
            if act == 0:
                ts_eval.route_stats['local'] += 1
            elif act == 1:
                ts_eval.route_stats['cloud'] += 1
            else:
                ts_eval.route_stats['edge'] += 1
            ts_eval.route_stats['recent'].append(int(act))
            if len(ts_eval.route_stats['recent']) > 100:
                ts_eval.route_stats['recent'].pop(0)
    ts_eval.finalize_episode(MAX_STEPS - 1)
    e_succ, d_succ = ts_eval.get_avg_results(only_successful=True)
    to_info = calc_timeout_rate(ts_eval)
    app_to = float(to_info['app_timeout_rate'])
    task_to = float(to_info.get('task_timeout_rate', to_info['app_timeout_rate']))
    if app_to > 1.0:
        app_to /= 100.0
    if task_to > 1.0:
        task_to /= 100.0
    assert 0.0 <= app_to <= 1.0, f'app_to={app_to} 超出范围'
    assert 0.0 <= task_to <= 1.0, f'task_to={task_to} 超出范围'
    sla0 = CONFIG.get('SLA0', 0.95)
    kappa = CONFIG.get('KAPPA', 2.0)
    v_cap = 3.0
    score = compute_score(E_avg=e_succ, D_succ_avg=d_succ, rho_app=app_to, rho_task=task_to, sla0=sla0, kappa=kappa, v_cap=v_cap)
    eval_missed_tasks = 0
    eval_total_timeout_tasks = 0
    eval_timeout_scheduled = 0
    eval_timeout_unscheduled = 0
    for uid in range(ts_eval.user_num):
        if ts_eval.enter_time[uid] == float('inf'):
            continue
        for sid, ft in ts_eval.finish_time[uid].items():
            if ft == float('inf'):
                eval_total_timeout_tasks += 1
                if (uid, sid) in eval_task2action:
                    eval_timeout_scheduled += 1
                else:
                    eval_timeout_unscheduled += 1
                    eval_missed_tasks += 1
    if eval_missed_tasks > 0:
        pass
    partition = subtask_partition_stats(ts_eval, eval_task2action)
    to_info['action_stats'] = {'local': partition['local'], 'cloud': partition['cloud'], 'edge': partition['edge'], 'timeout': partition['timeout'], 'unknown': partition['unknown'], 'total': partition['total_subtasks'], 'total_actions': len(eval_task2action)}
    total_e = ts_eval.get_sum_energy()
    return (float(e_succ), float(d_succ), float(score), to_info, float(total_e))

def train_transedge_style_wrapper(gpu_id, seed_offset, use_heuristic_sort=False, episodes=50):
    init_worker(seed_offset, para, CONFIG)
    if 'RUN_DIR' in os.environ:
        CONFIG['RUN_DIR'] = os.environ['RUN_DIR']
    if 'EVAL_MODE' in os.environ:
        CONFIG['EVAL_MODE'] = os.environ['EVAL_MODE'] == 'True'
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint_dir = None
    if CONFIG.get('RUN_DIR'):
        checkpoint_dir = Path(CONFIG['RUN_DIR']) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        if checkpoint_dir:
            old_checkpoint = checkpoint_dir / 'TransEdgeStyle.pt'
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        user_num = para['user_num']
        subgraph_num = 20
        basegraph_num = 60
        task_complex = para['task_complex']
        MAX_STEPS = CONFIG['MAX_STEPS']
        if isinstance(task_complex, (list, tuple)):
            task_complex_index = (CONFIG['SEED'] + seed_offset) % len(task_complex)
        else:
            task_complex_index = int(task_complex) if isinstance(task_complex, int) else 0
        env = Environment(user_num, subgraph_num, basegraph_num, task_complex_index)
        run_dir = CONFIG.get('RUN_DIR', '')
        plan_data = load_arrival_plan(run_dir, seed_offset) if run_dir else None
        if plan_data is None:
            env_seed = CONFIG['SEED'] + seed_offset
            arrival_plan = generate_arrival_plan(env_seed, MAX_STEPS, CONFIG['STOP_ARRIVAL_STEP'], 0.3, CONFIG.get('BURST_PROB', 0.15), max(1, CONFIG.get('BURST_SIZE', 4) // 2), CONFIG.get('BURST_SIZE', 4))
        else:
            arrival_plan = plan_data.get('arrival_plan', [])
            env_seed = plan_data.get('env_seed', CONFIG['SEED'] + seed_offset)
            pass
        env.generate_components(seed=env_seed)
        G = get_graph_cache(user_num, subgraph_num, basegraph_num, project_root)
        if G is not None and env.basegraph:
            env.basegraph.nx_graph = G
        deadline_config = load_deadline_config(run_dir)
        if deadline_config is None:
            pass
        else:
            pass
        ts = TaskScheduler(user_num, subgraph_num, basegraph_num, env, tight_deadline_config=deadline_config, seed=env_seed)
        gs = GraphScheduler(env.basegraph, env.subgraph_list, ts)
        bc = BenchmarkClass(env, gs, ts, task_complex_index, effective=True, seed=env_seed)
        ts.env = env
        ts.using_Algorithm = -1
        bc.reset()
        ts.reset()
        try:
            dummy_task = (0, 0)
            dummy_data, _ = extract_transedge_state(ts, dummy_task, slot=0, task_complex_index=task_complex_index)
            state_dim = dummy_data.x.shape[1]
        except Exception as e:
            state_dim = 52
        action_dim = para['edge_num'] + 2
        agent = TransEdgeStylePPOAgent(node_dim=state_dim, action_dim=action_dim, device=device, lr=0.0003, gamma=0.99, lmbda=0.95, eps_clip=0.2, K_epochs=4, entropy_coef=0.01)
        best_score = float('inf')
        best_metrics = None
        best_eval_score = float('inf')
        best_eval_metrics = None
        best_eval_appTO = 1.0
        EVAL_EVERY = 5
        eval_curve_data = []
        eval_run_count = 0
        ee0, dd0, sc0, to0, te0 = eval_transedge_style_once(agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS, use_heuristic_sort)
        best_eval_score = sc0
        best_eval_appTO = to0['app_timeout_rate']
        best_eval_metrics = {'e': ee0, 'd': dd0, 'score': sc0, 'total_energy': te0, 'timeout_info': dict(to0)}
        if checkpoint_dir is not None:
            torch.save(agent.policy.state_dict(), checkpoint_dir / 'TransEdgeStyle.pt')
        for episode in range(episodes):
            bc.reset()
            ts.reset()
            ts.route_stats = {'local': 0, 'edge': 0, 'cloud': 0, 'recent': []}
            agent.set_episode(episode)
            episode_rewards = []
            decision_count = 0
            action_count = [0] * action_dim
            total_actions = 0
            task2action = {}
            prev_app_to = set(ts.application_timeout_finished)
            prev_app_fin = set(ts.application_finished)
            arrived_tasks_total = 0
            for slot in range(MAX_STEPS):
                apply_arrival_plan(ts, slot, arrival_plan)
                ts.check_timeouts(slot)
                tasks = gs.get_tasks(slot, sort_tasks=use_heuristic_sort)
                if not tasks:
                    if slot >= CONFIG['STOP_ARRIVAL_STEP'] and safe_rest_tasks_total(ts.rest_tasks) == 0 and all_arrived_done(ts):
                        break
                    continue
                for task in tasks:
                    arrived_tasks_total += 1
                    state_data, mask_bin = extract_transedge_state(ts, task, slot=slot, task_complex_index=task_complex_index)
                    action, log_prob = agent.take_action(state_data, action_mask=mask_bin, route_stats=getattr(ts, 'route_stats', None))
                    if action < len(action_count):
                        action_count[action] += 1
                    total_actions += 1
                    task2action[tuple(task)] = int(action)
                    if action == 0:
                        ts.route_stats['local'] += 1
                    elif action == 1:
                        ts.route_stats['cloud'] += 1
                    else:
                        ts.route_stats['edge'] += 1
                    ts.route_stats['recent'].append(int(action))
                    if len(ts.route_stats['recent']) > 100:
                        ts.route_stats['recent'].pop(0)
                    reward, info = bc.step([[task, action]])
                    curr_app_to = set(ts.application_timeout_finished)
                    curr_app_fin = set(ts.application_finished)
                    new_app_to = curr_app_to - prev_app_to
                    new_app_fin = curr_app_fin - prev_app_fin
                    prev_app_to, prev_app_fin = (curr_app_to, curr_app_fin)
                    current_energy = float(info.get('step_energy', 0.0))
                    e_norm = max(0.0, min(1.0, (current_energy - 0.3) / (3.0 - 0.3)))
                    app_id = task[0]
                    step_delay = float(info.get('step_delay', 0.0))
                    delay_norm = max(0.0, min(1.0, step_delay / 5.0))
                    task_timeout = float(reward) < -0.1
                    route_stats = getattr(ts, 'route_stats', None)
                    r_utility = compute_structured_step_reward(app_timeout=app_id in new_app_to, app_done=app_id in new_app_fin, task_timeout=task_timeout, energy_norm=e_norm, delay_norm=delay_norm, action_val=action, route_stats=route_stats, edge_feasible=True, local_feasible=True)
                    r_scaled = r_utility
                    done = False
                    state_cpu = state_data.clone().cpu()
                    route_stats_snapshot = None
                    if hasattr(ts, 'route_stats'):
                        route_stats_snapshot = {'local': ts.route_stats.get('local', 0), 'edge': ts.route_stats.get('edge', 0), 'cloud': ts.route_stats.get('cloud', 0), 'recent': list(ts.route_stats.get('recent', []))}
                    agent.put_data((state_cpu, action, r_scaled, log_prob, done, mask_bin, route_stats_snapshot))
                    episode_rewards.append(r_scaled)
                    decision_count += 1
            terminal_bonus_total = 0.0
            if len(agent.memory) > 0:
                temp_to_info = calc_timeout_rate(ts)
                temp_app_to = float(temp_to_info['app_timeout_rate'])
                if temp_app_to > 1.0:
                    temp_app_to /= 100.0
                if temp_app_to <= 1e-06:
                    terminal_bonus_total = 1.0
                else:
                    terminal_bonus_total = -5.0 * temp_app_to
            missed_tasks = 0
            total_timeout_tasks = 0
            timeout_scheduled = 0
            timeout_unscheduled = 0
            for uid in range(ts.user_num):
                if ts.enter_time[uid] == float('inf'):
                    continue
                for sid, ft in ts.finish_time[uid].items():
                    if ft == float('inf'):
                        total_timeout_tasks += 1
                        if (uid, sid) in task2action:
                            timeout_scheduled += 1
                        else:
                            timeout_unscheduled += 1
                            missed_tasks += 1
            if missed_tasks > 0:
                pass
            num_transitions = len(agent.memory)
            if num_transitions > 0:
                terminal_bonus_per_step = terminal_bonus_total / num_transitions if terminal_bonus_total != 0 else 0.0
                updated_memory = []
                for transition in agent.memory:
                    updated_reward = transition[2] + terminal_bonus_per_step
                    updated_transition = (transition[0], transition[1], updated_reward, transition[3], transition[4], transition[5], transition[6])
                    updated_memory.append(updated_transition)
                agent.memory = updated_memory
                if terminal_bonus_total != 0:
                    pass
                episode_rewards.append(terminal_bonus_total)
            if len(agent.memory) > 0:
                agent.update()
                agent.clear_memory()
            if missed_tasks == 0 and total_timeout_tasks > 0:
                pass
            _te_succ = ts.get_avg_results(only_successful=True)
            _te_all = ts.get_avg_results(only_successful=False, timeout_charge='deadline')
            _te_to = calc_timeout_rate(ts)
            curve_cost = compute_score(_te_all[0], _te_all[1], _te_to['app_timeout_rate'], _te_to.get('task_timeout_rate', 0))
            if (episode + 1) % 20 == 0 and total_actions > 0:
                if action_dim > 1:
                    pass
                if action_dim > 2:
                    edge_total = sum(action_count[2:])
            if total_actions > 0:
                local_pct = 100.0 * action_count[0] / total_actions
                cloud_pct = 100.0 * action_count[1] / total_actions if action_dim > 1 else 0.0
                edge_count = sum(action_count[2:]) if action_dim > 2 else 0
                edge_pct = 100.0 * edge_count / total_actions if action_dim > 2 else 0.0
            if (episode + 1) % EVAL_EVERY == 0:
                ee, dd, sc, to, te = eval_transedge_style_once(agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS, use_heuristic_sort)
                curr_appTO = to['app_timeout_rate']
                eval_run_count += 1
                eval_curve_data.append({'episode': episode + 1, 'eval_run': eval_run_count, 'score': float(sc), 'energy': float(ee), 'total_energy': float(te), 'delay': float(dd), 'app_timeout_rate': float(curr_appTO), 'task_timeout_rate': float(to.get('task_timeout_rate', curr_appTO))})
                if curr_appTO < best_eval_appTO - 1e-09 or (abs(curr_appTO - best_eval_appTO) < 1e-09 and sc < best_eval_score):
                    best_eval_score = sc
                    best_eval_appTO = curr_appTO
                    best_eval_metrics = {'e': float(ee), 'd': float(dd), 'score': float(sc), 'total_energy': float(te), 'timeout_info': dict(to)}
                    if checkpoint_dir:
                        torch.save(agent.policy.state_dict(), checkpoint_dir / 'TransEdgeStyle.pt')
                else:
                    pass
            if episode_rewards:
                episode_rewards_array = np.array(episode_rewards)
                total_reward = float(np.sum(episode_rewards))
            else:
                total_reward = 0.0
                pass
            train_to_info = calc_timeout_rate(ts)
            e_succ, d_succ = ts.get_avg_results(only_successful=True)
            from contextlib import nullcontext
            with print_lock if print_lock else nullcontext():
                pass
            if checkpoint_dir:
                try:
                    train_curve_file = Path(CONFIG['RUN_DIR']) / 'curves' / f'TransEdgeStyle_seed{seed_offset}_train.csv'
                    train_curve_file.parent.mkdir(parents=True, exist_ok=True)
                    from experiments.exp_utils import append_curve_row
                    append_curve_row(train_curve_file, {'episode': episode + 1, 'reward': total_reward, 'score': float(curve_cost), 'energy': float(e_succ), 'delay': float(d_succ), 'app_timeout_rate': float(train_to_info['app_timeout_rate']), 'task_timeout_rate': float(train_to_info.get('task_timeout_rate', 0)), 'best_eval_score': float(best_eval_score), 'curve_type': 'training'})
                except Exception as curve_err:
                    pass
        if checkpoint_dir and len(eval_curve_data) > 0:
            try:
                eval_curve_file = Path(CONFIG['RUN_DIR']) / 'curves' / f'TransEdgeStyle_seed{seed_offset}_eval.csv'
                eval_curve_file.parent.mkdir(parents=True, exist_ok=True)
                from experiments.exp_utils import append_curve_row
                for eval_data in eval_curve_data:
                    append_curve_row(eval_curve_file, {'episode': eval_data['episode'], 'eval_run': eval_data['eval_run'], 'score': eval_data['score'], 'energy': eval_data['energy'], 'delay': eval_data['delay'], 'app_timeout_rate': eval_data['app_timeout_rate'], 'task_timeout_rate': eval_data['task_timeout_rate'], 'best_score': float(best_eval_score), 'curve_type': 'evaluation'})
            except Exception as eval_curve_err:
                pass
        if best_eval_metrics is not None:
            try:
                if checkpoint_dir:
                    checkpoint_path = checkpoint_dir / 'TransEdgeStyle.pt'
                    if checkpoint_path.exists():
                        best_model_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
                        agent.policy.load_state_dict(best_model_state)
            except Exception as e:
                pass
            best_eval_metrics['timeout_info']['inference_time_ms'] = 0.0
            best_eval_metrics['timeout_info']['inference_stats'] = {'median': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0}
            _best_to = best_eval_metrics['timeout_info']
            _best_app_pct = float(_best_to.get('app_timeout_rate', 0)) * 100.0
            _best_task_pct = float(_best_to.get('task_timeout_rate', 0)) * 100.0
            if 'action_stats' in _best_to:
                stats = _best_to['action_stats']
            flattened_metrics = {'app_timeout_rate': _best_app_pct, 'task_timeout_rate': _best_task_pct, 'score': best_eval_metrics.get('score', 0), 'inference_time_ms': _best_to.get('inference_time_ms', 0.0), 'action_stats': _best_to.get('action_stats', {}), 'timeout_rate': _best_to, 'total_energy': best_eval_metrics.get('total_energy', best_eval_metrics['e'])}
            sys.stdout.flush()
            agent.print_inference_stats()
            return (best_eval_metrics['e'], best_eval_metrics['d'], flattened_metrics)
        elif best_metrics:
            _app_pct = float(to_info['app_timeout_rate']) * 100.0
            to_info_with_inference = dict(to_info)
            to_info_with_inference['inference_time_ms'] = 0.0
            to_info_with_inference['inference_stats'] = {'median': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0}
            best_metrics = {'e': float(e_succ), 'd': float(d_succ), 'score': float(score), 'timeout_info': to_info_with_inference, 'action_count': [0] * (para['edge_num'] + 2), 'total_actions': 0, 'task2action': {}}
            try:
                total_e = ts.get_sum_energy()
            except:
                total_e = float(e_succ)
            best_metrics['total_energy'] = float(total_e)
            try:
                partition = subtask_partition_stats(ts, {})
                best_metrics['timeout_info']['action_stats'] = {'local': partition.get('local', 0), 'cloud': partition.get('cloud', 0), 'edge': partition.get('edge', 0), 'timeout': partition.get('timeout', 0), 'unknown': partition.get('unknown', 0), 'total': partition.get('total_subtasks', 0), 'total_actions': 0}
            except:
                pass
        if 'timeout_info' in best_metrics:
            to_info = best_metrics['timeout_info']
            flattened_metrics = {'app_timeout_rate': float(to_info.get('app_timeout_rate', 0)) * 100.0, 'task_timeout_rate': float(to_info.get('task_timeout_rate', 0)) * 100.0, 'score': best_metrics.get('score', 0), 'inference_time_ms': to_info.get('inference_time_ms', 0.0), 'action_stats': to_info.get('action_stats', {}), 'timeout_rate': to_info, 'total_energy': best_metrics.get('total_energy', best_metrics['e'])}
        else:
            flattened_metrics = {'app_timeout_rate': 0.0, 'task_timeout_rate': 0.0, 'score': best_metrics.get('score', 0), 'inference_time_ms': 0.0, 'action_stats': {}, 'timeout_rate': {}, 'total_energy': best_metrics.get('total_energy', best_metrics['e'])}
        sys.stdout.flush()
        agent.print_inference_stats()
        return (best_metrics['e'], best_metrics['d'], flattened_metrics)
    except Exception as e:
        raise
        return (1000.0, 1000.0, {'app_timeout_rate': 100.0, 'task_timeout_rate': 100.0, 'score': float('inf')})
