import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from Environment.environment import Environment
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from Algorithms import Benchmark
from utils.constant import para
from Algorithms.PPO.ppo_gae import PPO_GAE
from experiments.exp_utils import CONFIG, init_worker, load_arrival_plan, generate_arrival_plan, apply_arrival_plan, load_deadline_config, get_graph_cache, get_feature_dim, graph_state_to_vector, get_task_size_bytes, calc_timeout_rate, to_scalar, save_model_bundle, safe_rest_tasks_total, append_curve_row, diagnose_timeout, compute_score, all_arrived_done, subtask_outcome_stats, subtask_partition_stats
from Algorithms.Train.common import decode_action, ensure_trace_dir
print_lock = None

def set_print_lock(lock):
    global print_lock
    print_lock = lock

def train_ppo_wrapper(gpu_id, seed_offset, use_heuristic_sort=False, episodes=50):
    init_worker(seed_offset, para, CONFIG)
    if 'RUN_DIR' in os.environ:
        CONFIG['RUN_DIR'] = os.environ['RUN_DIR']
    if 'EVAL_MODE' in os.environ:
        CONFIG['EVAL_MODE'] = os.environ['EVAL_MODE'] == 'True'
    if PPO_GAE is None:
        raise ImportError('PPO_GAE 未成功导入')
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    if CONFIG.get('RUN_DIR'):
        checkpoint_dir = Path(CONFIG['RUN_DIR']) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_dir = None
    try:
        user_num = para['user_num']
        subgraph_num = 20
        basegraph_num = 60
        task_complex = para['task_complex']
        MAX_STEPS = CONFIG['MAX_STEPS']
        if isinstance(task_complex, (list, tuple)):
            task_complex_index = (CONFIG['SEED'] + seed_offset) % len(task_complex)
        else:
            task_complex_index = int(task_complex) if isinstance(task_complex, int) else 0
        env = Environment(user_num=user_num, subgraph_num=subgraph_num, basegraph_num=basegraph_num, task_complex_index=task_complex_index)
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
        if deadline_config:
            pass
        else:
            pass
        ts = TaskScheduler(user_num, subgraph_num, basegraph_num, env, tight_deadline_config=deadline_config, seed=env_seed)
        gs = GraphScheduler(env.basegraph, env.subgraph_list, ts)
        bc = Benchmark.Benchmark(env, gs, ts, task_complex_index, effective=True, seed=env_seed)
        ts.env = env
        ts.using_Algorithm = -1
        bc.reset()
        ts.reset()
        slot_interval = para['slot_interval']
        if hasattr(ts, 'app_deadline_slots'):
            deadline_slots = [ts.get_app_deadline_slot(i) for i in range(min(10, user_num))]
            deadline_abs = []
            for i in range(min(10, user_num)):
                if ts.enter_time[i] != float('inf'):
                    dead_abs = float(ts.enter_time[i]) + int(ts.get_app_deadline_slot(i)) * slot_interval
                    deadline_abs.append(f'{dead_abs:.3f}s')
                else:
                    deadline_abs.append('N/A')
        else:
            pass
        if isinstance(arrival_plan, dict) and 'schedule' in arrival_plan:
            schedule = arrival_plan.get('schedule', [])
        n_states = get_feature_dim(env, gs, task_complex_index)
        n_actions = para['edge_num'] + 2
        sys.stdout.flush()
        agent = PPO_GAE(n_states=n_states, n_hiddens=min(n_states * 3, 768), n_actions=n_actions, actor_lr=0.0003, critic_lr=0.001, lmbda=0.95, epochs=5, eps=0.2, gamma=0.99, device=device, batch_size=512, ent_coef=0.05)
        EPISODES = episodes
        BATCH_STEPS = 500
        best_score = float('inf')
        best_metrics = None
        sys.stdout.flush()
        is_eval_mode = CONFIG.get('EVAL_MODE', False)
        checkpoint_path = None

        def get_param_norm(model):
            return sum((float(p.abs().sum().cpu().item()) for p in model.parameters()))
        before_norm = get_param_norm(agent.actor)
        sys.stdout.flush()
        timeout_info = {'app_timeout_rate': 1.0, 'task_timeout_rate': 1.0}
        e, d, current_score = (1000.0, 1000.0, 1000.0)
        for episode in range(EPISODES):
            ts.reset()
            bc.reset()
            transition_dicts = {}
            step_count = 0
            last_print_step = 0
            episode_rewards = []
            action_count = [0] * n_actions
            total_actions = 0
            mask_stats = {'all_valid': 0, 'only_local': 0, 'cloud_edge_banned': 0}
            task2action = {}
            for slot in range(MAX_STEPS):
                apply_arrival_plan(ts, slot, arrival_plan)
                ts.check_timeouts(slot)
                prev_app_to = set(getattr(ts, 'application_timeout_finished', set()))
                prev_app_fin = set(getattr(ts, 'application_finished', set()))
                tasks = gs.get_tasks(slot, sort_tasks=False)
                if not tasks:
                    if slot >= CONFIG['STOP_ARRIVAL_STEP']:
                        from experiments.exp_utils import safe_rest_tasks_total, all_arrived_done
                        if safe_rest_tasks_total(ts.rest_tasks) == 0 and all_arrived_done(ts):
                            break
                    continue
                for task in tasks:
                    task_id = f'{task[0]}_{task[1]}'
                    if task_id not in transition_dicts:
                        transition_dicts[task_id] = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': [], 'masks': []}
                    state_data = gs.get_graph_state_new(env, task, task_complex_index, slot=slot)
                    state = graph_state_to_vector(state_data, method='mean')
                    if hasattr(state, 'to'):
                        state = state.to(agent.device)
                    user_id, subtask_id = task
                    task_size_bytes = get_task_size_bytes(ts, user_id, subtask_id)
                    now_time = slot * para['slot_interval']
                    base_mask = ts.get_action_mask(user_id, task_size_bytes, now_time)
                    if not bool(base_mask.any()):
                        continue
                    mask_add = list(base_mask)
                    mask_bin = [float(bool(m)) for m in mask_add]
                    mask_bin = mask_bin[:n_actions] + [0.0] * max(0, n_actions - len(mask_bin))
                    for eid in range(para['edge_num']):
                        core_times = [max(0, t - now_time) for t in ts.edge_useful[eid]]
                        min_wait = min(core_times) if core_times else 0.0
                        avg_wait = sum(core_times) / len(core_times) if core_times else 0.0
                        if avg_wait > 1.5:
                            action_idx = 2 + eid
                            if action_idx < len(mask_bin):
                                mask_bin[action_idx] = 0.0
                    if sum(mask_bin) < 0.5:
                        mask_bin = [float(bool(m)) for m in base_mask]
                        if step_count < 10:
                            pass
                    valid_actions = int(sum(mask_bin))
                    if valid_actions == n_actions:
                        mask_stats['all_valid'] += 1
                    elif valid_actions == 1 and mask_bin[0] == 1.0:
                        mask_stats['only_local'] += 1
                    else:
                        mask_stats['cloud_edge_banned'] += 1
                    if valid_actions == 1:
                        only_cloud = mask_bin[1] > 0.5
                        only_local = mask_bin[0] > 0.5
                        only_edge = any((mask_bin[i] > 0.5 for i in range(2, len(mask_bin))))
                        if only_cloud and slot % 1000 < 5:
                            pass
                        if only_local and slot % 1000 < 5:
                            pass
                        if only_edge and slot % 1000 < 5:
                            pass
                    action, *rest = agent.take_action(state, task_id, action_mask=mask_bin, deterministic=False)
                    task2action[tuple(task)] = int(action)
                    if action < len(action_count):
                        action_count[action] += 1
                    total_actions += 1
                    step_result = bc.step([[task, action]])
                    reward = step_result[0]
                    info = step_result[1] if len(step_result) > 1 else {}
                    app_id = user_id
                    curr_to = set(ts.application_timeout_finished)
                    curr_fin = set(ts.application_finished)
                    new_to = curr_to - prev_app_to
                    new_fin = curr_fin - prev_app_fin
                    prev_app_to, prev_app_fin = (curr_to, curr_fin)
                    r_val = to_scalar(reward)

                    def compute_utility_step_reward(app_timeout, app_done, energy_norm, action_val, r_subtask, finish_ratio=0.0, local_wait_time=0.0, episode_num=0):
                        if app_timeout:
                            return -10.0 - 2.0 * (1.0 - finish_ratio)
                        if app_done:
                            return 10.0 + 1.0 * (1.0 - energy_norm)
                        step_reward = 0.0
                        if action_val >= 2:
                            edge_id = action_val - 2
                            core_times = [max(0, t - now_time) for t in ts.edge_useful[edge_id]]
                            min_wait = min(core_times) if core_times else 0.0
                            step_reward -= min_wait * 2.0
                        elif action_val == 0:
                            wait = max(0, ts.devices_exe_useful[user_id] - now_time)
                            step_reward -= wait * 2.0
                        if action_val == 1:
                            step_reward += 1.0
                        if r_subtask < -2.0:
                            step_reward -= 0.5
                        return step_reward
                    current_energy = float(info.get('step_energy', 0.0))
                    e_norm = max(0.0, min(1.0, (current_energy - 0.3) / (3.0 - 0.3)))
                    local_wait_time = 0.0
                    if action == 0:
                        now = slot * para['slot_interval']
                        free_time = float(ts.devices_exe_useful[user_id])
                        local_wait_time = max(0.0, free_time - now)
                    if hasattr(ts, 'finish_time') and app_id in ts.finish_time:
                        finished_count = len([t for t in ts.finish_time[app_id].values() if t != float('inf')])
                        total_count = max(1, len(ts.finish_time[app_id]))
                        finish_ratio = finished_count / total_count
                    else:
                        finish_ratio = 0.0
                    train_reward = compute_utility_step_reward(app_timeout=app_id in new_to, app_done=app_id in new_fin, energy_norm=e_norm, action_val=action, r_subtask=r_val, finish_ratio=finish_ratio, local_wait_time=local_wait_time, episode_num=episode)
                    train_reward = max(-12.0, min(11.0, train_reward))
                    done = False
                    from experiments.exp_utils import all_arrived_done, safe_rest_tasks_total
                    if all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
                        done = True
                    if hasattr(state, 'to'):
                        state_cpu = state.to('cpu')
                    else:
                        state_cpu = state
                    transition_dicts[task_id]['masks'].append(list(mask_bin))
                    transition_dicts[task_id]['states'].append(state_cpu)
                    transition_dicts[task_id]['actions'].append(action)
                    transition_dicts[task_id]['rewards'].append(train_reward)
                    episode_rewards.append(train_reward)
                    transition_dicts[task_id]['dones'].append(done)
                    next_state_data = gs.get_graph_state_new(env, task, task_complex_index, slot=slot)
                    next_state = graph_state_to_vector(next_state_data, method='mean')
                    if hasattr(next_state, 'to'):
                        next_state = next_state.to('cpu')
                    transition_dicts[task_id]['next_states'].append(next_state)
                    step_count += 1
                    if step_count % BATCH_STEPS == 0:
                        pass
                    if step_count - last_print_step >= 1000:
                        sys.stdout.flush()
                        last_print_step = step_count
                if step_count > MAX_STEPS:
                    break
            ts.finalize_episode(slot)
            timeout_info = calc_timeout_rate(ts)
            app_to = float(timeout_info['app_timeout_rate'])
            task_to = float(timeout_info['task_timeout_rate'])
            e_succ, d_succ = ts.get_avg_results(only_successful=True)
            current_score = compute_score(E_avg=e_succ, D_succ_avg=d_succ, rho_app=app_to, rho_task=task_to, sla0=CONFIG.get('SLA0', 0.95), kappa=CONFIG.get('KAPPA', 2.0), v_cap=3.0)
            terminal_bonus_total = -20.0 * app_to
            if app_to < 0.05:
                terminal_bonus_total += 5.0
            if transition_dicts:
                total_episode_steps = sum((len(d['rewards']) for d in transition_dicts.values()))
                if total_episode_steps > 0:
                    bonus_per_step = terminal_bonus_total / total_episode_steps
                    for t_id in transition_dicts:
                        for i in range(len(transition_dicts[t_id]['rewards'])):
                            transition_dicts[t_id]['rewards'][i] += bonus_per_step
                merged_transition = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': [], 'masks': []}
                for task_id, transition_dict in transition_dicts.items():
                    if transition_dict['states']:
                        merged_transition['masks'].extend(transition_dict['masks'])
                        merged_transition['states'].extend(transition_dict['states'])
                        merged_transition['actions'].extend(transition_dict['actions'])
                        merged_transition['rewards'].extend(transition_dict['rewards'])
                        merged_transition['next_states'].extend(transition_dict['next_states'])
                        merged_transition['dones'].extend(transition_dict['dones'])
                if merged_transition['states']:
                    actor_loss, critic_loss = agent.learn(merged_transition)
            agent.clear_task_data()
            e, d = ts.get_avg_results()
            total_subtasks, finished_subtasks, unfinished_subtasks, timeout_subtasks = subtask_outcome_stats(ts)
            if current_score < best_score:
                best_score = current_score
                flag = '新最佳'
                best_metrics = {'e': float(e), 'd': float(d_succ), 'score': float(current_score), 'timeout_info': dict(timeout_info), 'action_count': list(action_count), 'total_actions': int(total_actions), 'mask_stats': dict(mask_stats), 'task2action': dict(task2action)}
                partition = subtask_partition_stats(ts, task2action)
                best_metrics['timeout_info']['action_stats'] = {'local': partition['local'], 'cloud': partition['cloud'], 'edge': partition['edge'], 'timeout': partition['timeout'], 'unknown': partition['unknown'], 'total': partition['total_subtasks'], 'total_actions': int(total_actions)}
                if checkpoint_dir and (not is_eval_mode):
                    try:
                        meta = {'seed': int(CONFIG['SEED']), 'seed_offset': int(seed_offset), 'state_dim': int(n_states), 'action_dim': int(n_actions), 'edge_num': int(para['edge_num']), 'deadline_slot': int(para['deadline_slot']), 'uplink_range': list(para['uplink_range']), 'task_complex_index': int(task_complex_index), 'score': float(current_score), 'energy': float(e), 'delay': float(d_succ), 'app_timeout_rate': float(timeout_info['app_timeout_rate']), 'use_gnn': False}
                        save_model_bundle('PPO', agent, checkpoint_dir, **meta)
                    except Exception as save_err:
                        pass
            else:
                flag = ''
            if episode_rewards:
                episode_rewards_array = np.array(episode_rewards)
                total_reward = float(np.sum(episode_rewards))
            else:
                total_reward = 0.0
                pass
            if checkpoint_dir:
                try:
                    curve_file = Path(CONFIG['RUN_DIR']) / 'curves' / f'PPO_seed{seed_offset}.csv'
                    curve_file.parent.mkdir(parents=True, exist_ok=True)
                    append_curve_row(curve_file, {'episode': episode + 1, 'reward': total_reward, 'score': float(current_score), 'energy': float(e), 'delay': float(d_succ), 'app_timeout_rate': float(timeout_info['app_timeout_rate']), 'best_score': float(best_score)})
                except Exception as curve_err:
                    pass
            if (episode + 1) % 20 == 0:
                diagnose_timeout(ts, 'PPO')
                if total_actions > 0:
                    if n_actions > 1:
                        pass
                    if n_actions > 2:
                        edge_total = sum(action_count[2:])
                    if action_count[0] > 0.95 * total_actions:
                        if mask_stats['only_local'] > 0.5 * total_actions:
                            pass
                        else:
                            pass
            sys.stdout.flush()
        if best_metrics is not None:
            best_e = best_metrics['e']
            best_d = best_metrics['d']
            best_timeout_info = best_metrics['timeout_info']
            sys.stdout.flush()
            trace_dir = ensure_trace_dir(CONFIG['RUN_DIR']) if CONFIG.get('RUN_DIR') else None
            if trace_dir and 'trace_decisions' in locals() and trace_decisions:
                pd.DataFrame(trace_decisions).to_csv(trace_dir / 'PPO_decisions.csv', index=False)
            return (best_e, best_d, best_timeout_info)
        else:
            pass
            sys.stdout.flush()
            return (1000.0, 1000.0, {'timeout_rate': 1.0, 'app_timeout_rate': 1.0, 'task_timeout_rate': 1.0})
        for slot in range(MAX_STEPS):
            total_eval_slots += 1
            apply_arrival_plan(ts, slot, arrival_plan)
            ts.check_timeouts(slot)
            if slot >= CONFIG['STOP_ARRIVAL_STEP']:
                if all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
                    break
            tasks = gs.get_tasks(slot, sort_tasks=False)
            if not tasks:
                empty_slots += 1
                continue
            for task in tasks:
                task_id = f'{task[0]}_{task[1]}'
                if task_id not in transition_dicts:
                    transition_dicts[task_id] = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': [], 'masks': []}
                state_data = gs.get_graph_state_new(env, task, task_complex_index, slot=slot)
                state = graph_state_to_vector(state_data, method='mean')
                if hasattr(state, 'to'):
                    state = state.to(agent.device)
                user_id, subtask_id = task
                task_size_bytes = get_task_size_bytes(ts, user_id, subtask_id)
                now_time = slot * para['slot_interval']
                base_mask = ts.get_action_mask(user_id, task_size_bytes, now_time)
                if not bool(base_mask.any()):
                    continue
                mask_add = list(base_mask)
                mask_bin = [float(bool(m)) for m in mask_add]
                inference_start = time.time()
                action, *rest = agent.take_action(state, task_id, action_mask=mask_bin, deterministic=True)
                inference_time_ms = (time.time() - inference_start) * 1000.0
                inference_times.append(inference_time_ms)
                task2action[tuple(task)] = int(action)
                target, edge_id = decode_action(int(action))
                trace_decisions.append({'decision_idx': decision_idx, 'slot': slot, 'uid': int(user_id), 'sid': int(subtask_id), 'action': int(action), 'target': target, 'edge_id': int(edge_id)})
                decision_idx += 1
                if 0 <= action < len(action_counts):
                    action_counts[action] += 1
                total_eval_actions += 1
                step_result = bc.step([[task, action]])
                reward = step_result[0]
                step_count += 1
            if step_count > MAX_STEPS:
                break
        ts.finalize_episode(slot)
        eval_e, eval_d = ts.get_avg_results()
        eval_timeout_info = calc_timeout_rate(ts)
        eval_e_succ, eval_d_succ = ts.get_avg_results(only_successful=True)
        app_to = eval_timeout_info['app_timeout_rate']
        task_to = eval_timeout_info['task_timeout_rate']
        sla0 = CONFIG.get('SLA0', 0.95)
        kappa = CONFIG.get('KAPPA', 2.0)
        v_cap = 3.0
        eval_score = compute_score(E_avg=eval_e_succ, D_succ_avg=eval_d_succ, rho_app=app_to, rho_task=task_to, sla0=sla0, kappa=kappa, v_cap=v_cap)
        sys.stdout.flush()
        eval_timeout_info['score'] = eval_score
        total_subtasks, finished_subtasks, unfinished_subtasks, timeout_subtasks = subtask_outcome_stats(ts)
        eval_timeout_info['subtask_stats'] = {'total': int(total_subtasks), 'finished': int(finished_subtasks), 'unfinished': int(unfinished_subtasks), 'timeout': int(timeout_subtasks)}
        partition = subtask_partition_stats(ts, task2action)
        if total_eval_actions > 0:
            eval_timeout_info['action_stats'] = {'local': partition['local'], 'cloud': partition['cloud'], 'edge': partition['edge'], 'timeout': partition['timeout'], 'unknown': partition['unknown'], 'total': partition['total_subtasks'], 'total_slots': total_eval_slots, 'empty_slots': empty_slots, 'total_actions': total_eval_actions}
        if inference_times:
            eval_timeout_info['inference_time_ms'] = float(np.mean(inference_times))
            eval_timeout_info['inference_stats'] = {'median': float(np.median(inference_times)), 'min': float(np.min(inference_times)), 'max': float(np.max(inference_times)), 'std': float(np.std(inference_times)), 'count': len(inference_times)}
        else:
            eval_timeout_info['inference_time_ms'] = 0.0
            eval_timeout_info['inference_stats'] = {'median': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0}
        sys.stdout.flush()
        if trace_dir and trace_decisions:
            pd.DataFrame(trace_decisions).to_csv(trace_dir / 'PPO_decisions.csv', index=False)
        return (best_e, best_d, best_timeout_info)
    except Exception as e:
        raise
        return (1000.0, 1000.0, {'timeout_rate': 1.0, 'app_timeout_rate': 1.0, 'task_timeout_rate': 1.0, 'error': str(e)})
