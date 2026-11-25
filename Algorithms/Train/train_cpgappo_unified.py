import os, sys
import numpy as np
import torch
from pathlib import Path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from experiments.exp_utils import CONFIG, init_worker, generate_arrival_plan, load_deadline_config, compute_score, calc_timeout_rate, safe_rest_tasks_total, all_arrived_done, get_graph_cache, apply_arrival_plan, subtask_partition_stats, append_curve_row
from Environment.environment import Environment
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from Algorithms.Benchmark import Benchmark as BenchmarkClass
from utils.constant import para
from Algorithms.RealGATPPO.agent_cpgappo import GAT_PPO_Agent_CPGAPPO, compute_guide_scores_cp
from Algorithms.RealGATPPO.cpgappo_core import extract_cpgappo_state, compute_cpgappo_slack_reward
BONUS_FIN = 4.0
BONUS_TO = -7.0
TERMINAL_COEF = 5.0
SHIELD_THR_HIGH = 1.1
SHIELD_THR_NORM = 1.15

def application_credit(completed, timed_out):
    """Equation (30): recorded completion takes precedence in the same update.

    completed denotes membership in the newly recorded completion set,
    including late completion; timed_out denotes the timeout-finished set."""
    return BONUS_FIN if completed else BONUS_TO if timed_out else 0.0

def terminal_credit(app_timeout):
    """Equation (33), using recorded application noncompletion in [0, 1].

    The positive branch rewards complete recorded coverage, including
    late completions that entered the completed-application set."""
    return 1.0 if app_timeout <= 1e-06 else -TERMINAL_COEF * app_timeout

def compute_cp_factor(ts, uid, sid):
    try:
        g = ts.subgraph_list[uid].nx_graph
        succs = list(g.successors(sid))
    except Exception:
        return 0.0
    if not succs:
        return 0.0

    def _cp_depth_from(node, memo):
        if node in memo:
            return memo[node]
        try:
            node_succs = list(g.successors(node))
        except Exception:
            memo[node] = 0
            return 0
        if not node_succs:
            memo[node] = 0
            return 0
        depth = 1 + max((_cp_depth_from(s, memo) for s in node_succs))
        memo[node] = depth
        return depth
    cp_depth = _cp_depth_from(sid, {})
    return min(1.0, cp_depth / 4.0)

def build_cpgappo_safe_mask(ts, uid, now, mask_bin, guide_scores, cp_slack):
    app_deadline_slot = ts.get_app_deadline_slot(uid) if hasattr(ts, 'get_app_deadline_slot') else para['deadline_slot']
    if uid < len(ts.enter_time) and ts.enter_time[uid] != float('inf'):
        enter_time = float(ts.enter_time[uid])
    else:
        enter_time = float(now)
    time_left = max(0.0, enter_time + app_deadline_slot * para['slot_interval'] - now)
    budget = max(1e-06, app_deadline_slot * para['slot_interval'])
    is_high_risk = time_left < 0.5 * budget or cp_slack < 0.5 * time_left
    threshold = SHIELD_THR_HIGH if is_high_risk else SHIELD_THR_NORM
    safe_mask = np.array(mask_bin, dtype=bool)
    for a in range(len(safe_mask)):
        if safe_mask[a] and guide_scores[a] > time_left * threshold:
            safe_mask[a] = False
    if not safe_mask.any():
        safe_mask = np.array(mask_bin, dtype=bool)
    return safe_mask

def eval_cpgappo_once(agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS, use_shield=True, use_cp=True):
    ts.reset()
    eval_task2action = {}
    action_count = [0] * (para['edge_num'] + 2)
    total_actions = 0
    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts, slot, arrival_plan)
        ts.check_timeouts(slot)
        tasks = gs.get_tasks(slot, sort_tasks=use_cp)
        if not tasks:
            if slot >= CONFIG['STOP_ARRIVAL_STEP'] and safe_rest_tasks_total(ts.rest_tasks) == 0 and all_arrived_done(ts):
                break
            continue
        for task in tasks:
            uid = task[0]
            now = slot * para['slot_interval']
            s_data, g_feat, mask_bin = extract_cpgappo_state(ts, task, slot, task_complex_index)
            if not np.any(mask_bin):
                continue
            masked_scores = None
            if use_shield:
                guide_scores, guide_valid = compute_guide_scores_cp(ts, uid, task, task_complex_index, now)
                masked_scores = np.where(np.asarray(guide_valid) & np.asarray(mask_bin, dtype=bool), guide_scores, np.inf)
                action_mask = build_cpgappo_safe_mask(ts, uid, now, mask_bin, masked_scores, gs.get_slack_cp(uid, slot, now=now))
            else:
                action_mask = mask_bin
            action, _ = agent.take_action(s_data, g_feat, action_mask=action_mask, deterministic=True)
            bc.step([[task, action]])
            eval_task2action[tuple(task)] = int(action)
            if action < len(action_count):
                action_count[action] += 1
            total_actions += 1
    ts.finalize_episode(MAX_STEPS - 1)
    try:
        e, d = ts.get_avg_results(only_successful=True)
    except Exception:
        e, d = (0.0, 0.0)
    to_info = calc_timeout_rate(ts)
    rho_app = float(to_info['app_timeout_rate'])
    rho_task = float(to_info.get('task_timeout_rate', 0))
    if rho_app > 1.0:
        rho_app /= 100.0
    if rho_task > 1.0:
        rho_task /= 100.0
    score = compute_score(e, d, rho_app, rho_task)
    total_energy = ts.total_energy if hasattr(ts, 'total_energy') else 0.0
    partition = subtask_partition_stats(ts, eval_task2action)
    to_info['action_stats'] = {'local': partition['local'], 'cloud': partition['cloud'], 'edge': partition['edge'], 'timeout': partition['timeout'], 'unknown': partition['unknown'], 'total': partition['total_subtasks'], 'total_actions': len(eval_task2action)}
    total_sub = partition['total_subtasks']
    finished_sub = partition['local'] + partition['cloud'] + partition['edge']
    to_info['subtask_stats'] = {'total': total_sub, 'finished': finished_sub, 'unfinished': total_sub - finished_sub}
    return (e, d, score, to_info, total_energy)

def train_cpgappo(gpu_id, seed_offset, episodes=20, lr=0.0003, entropy_coef=0.02, lambda_guide=0.1, use_backward=True, use_shield=True, use_app_credit=True, use_cp=True, use_full_credit=False, algo_tag='CPGAPPO_unified'):
    init_worker(seed_offset, para, CONFIG)
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint_dir = None
    curves_dir = None
    if CONFIG.get('RUN_DIR'):
        checkpoint_dir = Path(CONFIG['RUN_DIR']) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        curves_dir = Path(CONFIG['RUN_DIR']) / 'curves'
        curves_dir.mkdir(parents=True, exist_ok=True)
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
    env_seed = CONFIG['SEED'] + seed_offset
    arrival_plan = generate_arrival_plan(env_seed, MAX_STEPS, CONFIG['STOP_ARRIVAL_STEP'], 0.3, CONFIG.get('BURST_PROB', 0.15), max(1, CONFIG.get('BURST_SIZE', 4) // 2), CONFIG.get('BURST_SIZE', 4))
    env.generate_components(seed=env_seed)
    G = get_graph_cache(user_num, subgraph_num, basegraph_num, project_root)
    if G is not None and env.basegraph:
        env.basegraph.nx_graph = G
    deadline_config = load_deadline_config('')
    ts = TaskScheduler(user_num, subgraph_num, basegraph_num, env, tight_deadline_config=deadline_config, seed=env_seed)
    gs = GraphScheduler(env.basegraph, env.subgraph_list, ts)
    bc = BenchmarkClass(env, gs, ts, task_complex_index, effective=True, seed=env_seed)
    ts.env = env
    ts.using_Algorithm = -1
    bc.reset()
    ts.reset()
    try:
        dummy_task = (0, 0)
        dummy_data, dummy_gfeat, _ = extract_cpgappo_state(ts, dummy_task, slot=0, task_complex_index=task_complex_index)
        state_dim = dummy_data.x.shape[1]
        global_dim = dummy_gfeat.shape[0]
    except Exception:
        state_dim, global_dim = (27, 10)
    action_dim = para['edge_num'] + 2
    if CONFIG.get('OPTIMIZATION_SEED') is not None:
        import random
        optimization_seed = int(CONFIG['OPTIMIZATION_SEED'])
        random.seed(optimization_seed)
        np.random.seed(optimization_seed)
        torch.manual_seed(optimization_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(optimization_seed)
    agent = GAT_PPO_Agent_CPGAPPO(node_dim=state_dim, global_dim=global_dim, action_dim=action_dim, device=device, lr=lr, entropy_coef=entropy_coef, lambda_guide=lambda_guide, use_backward=use_backward)
    best_eval_score = float('inf')
    best_eval_appTO = 1.0
    best_eval_metrics = None
    ee0, dd0, sc0, to0, te0 = eval_cpgappo_once(agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS, use_shield=use_shield, use_cp=use_cp)
    best_eval_score = sc0
    best_eval_metrics = dict(e=ee0, d=dd0, score=sc0, total_energy=te0, timeout_info=dict(to0))
    if checkpoint_dir is not None:
        torch.save(agent.policy.state_dict(), checkpoint_dir / f'{algo_tag}.pt')
    best_eval_appTO = to0['app_timeout_rate']
    for episode in range(episodes):
        bc.reset()
        ts.reset()
        action_count = [0] * action_dim
        total_actions = 0
        episode_rewards = []
        shield_total, shield_blocked = (0, 0)
        app_fin_count, app_to_count = (0, 0)
        cp_credit_total = 0.0
        cp_credit_touched = 0
        prev_app_to = set(ts.application_timeout_finished)
        prev_app_fin = set(ts.application_finished)
        app_buffer = {}
        for slot in range(MAX_STEPS):
            apply_arrival_plan(ts, slot, arrival_plan)
            ts.check_timeouts(slot)
            tasks = gs.get_tasks(slot, sort_tasks=use_cp)
            if not tasks:
                if slot >= CONFIG['STOP_ARRIVAL_STEP'] and safe_rest_tasks_total(ts.rest_tasks) == 0 and all_arrived_done(ts):
                    break
                continue
            now = slot * para['slot_interval']
            for task in tasks:
                uid = task[0]
                sid = task[1]
                s_data, g_feat, mask_bin = extract_cpgappo_state(ts, task, slot, task_complex_index)
                if not np.any(mask_bin):
                    continue
                guide_best_action = 0
                action_mask = mask_bin
                buffer_mask = mask_bin
                need_guide = use_shield or (lambda_guide > 0.0 and (not use_shield))
                if need_guide:
                    guide_scores, guide_valid = compute_guide_scores_cp(ts, uid, task, task_complex_index, now)
                    masked_scores = np.where(np.asarray(guide_valid) & np.asarray(mask_bin, dtype=bool), guide_scores, np.inf)
                    guide_best_action = int(np.argmin(masked_scores))
                    safe_mask = build_cpgappo_safe_mask(ts, uid, now, mask_bin, masked_scores, gs.get_slack_cp(uid, slot, now=now))
                    if use_shield:
                        action_mask = safe_mask
                        buffer_mask = safe_mask
                        raw_mask_count = np.sum(mask_bin)
                        safe_mask_count = np.sum(action_mask)
                        shield_total += 1
                        if safe_mask_count < raw_mask_count:
                            shield_blocked += 1
                    else:
                        buffer_mask = action_mask
                action, log_prob = agent.take_action(s_data, g_feat, action_mask=action_mask)
                if action < len(action_count):
                    action_count[action] += 1
                total_actions += 1
                reward, info = bc.step([[task, action]])
                current_energy = float(info.get('step_energy', 0.0))
                step_delay = float(info.get('step_delay', 0.0))
                r_scaled, slack = compute_cpgappo_slack_reward(ts, uid, sid, current_energy, step_delay, ts.enter_time[uid], ts.get_app_deadline_slot(uid))
                cp_factor = compute_cp_factor(ts, uid, sid)
                if use_full_credit:
                    cp_factor = 1.0
                state_cpu = s_data.clone().cpu()
                app_buffer.setdefault(uid, []).append([state_cpu, g_feat.clone().cpu(), action, r_scaled, log_prob, False, buffer_mask if isinstance(buffer_mask, np.ndarray) else np.array(buffer_mask, dtype=bool), guide_best_action, cp_factor, total_actions])
                episode_rewards.append(r_scaled)
                curr_app_to = set(ts.application_timeout_finished)
                curr_app_fin = set(ts.application_finished)
                new_app_to = curr_app_to - prev_app_to
                new_app_fin = curr_app_fin - prev_app_fin
                prev_app_to, prev_app_fin = (curr_app_to, curr_app_fin)
                app_fin_count += len(new_app_fin)
                app_to_count += len(new_app_to)
                for f_uid in new_app_fin | new_app_to:
                    if f_uid in app_buffer:
                        if use_app_credit:
                            bonus = application_credit(f_uid in new_app_fin, f_uid in new_app_to)
                            for t in app_buffer[f_uid]:
                                cpf = float(t[8]) if len(t) > 8 else 0.0
                                weighted_bonus = bonus * cpf
                                t[3] += weighted_bonus
                                cp_credit_total += abs(weighted_bonus)
                                if cpf > 1e-06:
                                    cp_credit_touched += 1
                                agent.put_data(tuple(t))
                        else:
                            for t in app_buffer[f_uid]:
                                agent.put_data(tuple(t))
                        del app_buffer[f_uid]
        for t_list in app_buffer.values():
            for t in t_list:
                agent.put_data(tuple(t))
        app_buffer.clear()
        ts.finalize_episode(MAX_STEPS - 1)
        temp_to = calc_timeout_rate(ts)
        temp_app_to = float(temp_to['app_timeout_rate'])
        if temp_app_to > 1.0:
            temp_app_to /= 100.0
        terminal_bonus = terminal_credit(temp_app_to)
        agent.memory.sort(key=lambda transition: transition[9])
        n_t = len(agent.memory)
        if n_t > 0:
            bonus_per = terminal_bonus / n_t
            updated = []
            for t in agent.memory:
                updated.append((t[0], t[1], t[2], t[3] + bonus_per, t[4], t[5], t[6], t[7]))
            agent.memory = updated
            agent.update()
            agent.clear_memory()
        ep_reward_sum = sum(episode_rewards)
        shield_rate = shield_blocked / max(shield_total, 1)
        sys.stdout.flush()
        if curves_dir is not None:
            try:
                train_curve_file = curves_dir / f'{algo_tag}_seed{seed_offset}_train.csv'
                append_curve_row(train_curve_file, {'episode': episode + 1, 'reward': float(ep_reward_sum), 'score': float(-temp_app_to), 'energy': float(getattr(ts, 'total_energy', 0.0) / max(1, user_num)), 'delay': 0.0, 'app_timeout_rate': float(temp_app_to), 'task_timeout_rate': float(temp_to.get('task_timeout_rate', 0.0)), 'curve_type': 'training'})
            except Exception:
                pass
        if (episode + 1) % 5 == 0 or episode + 1 == episodes:
            ee, dd, sc, to_info, te = eval_cpgappo_once(agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS, use_shield=use_shield, use_cp=use_cp)
            eval_appTO = to_info['app_timeout_rate']
            if sc < best_eval_score:
                best_eval_score = sc
                best_eval_appTO = eval_appTO
                best_eval_metrics = {'e': ee, 'd': dd, 'score': sc, 'total_energy': te, 'timeout_info': dict(to_info)}
                if checkpoint_dir is not None:
                    try:
                        torch.save(agent.policy.state_dict(), checkpoint_dir / f'{algo_tag}.pt')
                    except Exception:
                        raise
            else:
                pass
            sys.stdout.flush()
            if curves_dir is not None:
                try:
                    eval_curve_file = curves_dir / f'{algo_tag}_seed{seed_offset}_eval.csv'
                    append_curve_row(eval_curve_file, {'episode': episode + 1, 'score': float(sc), 'energy': float(ee), 'delay': float(dd), 'app_timeout_rate': float(eval_appTO), 'task_timeout_rate': float(to_info.get('task_timeout_rate', 0.0)), 'curve_type': 'evaluation'})
                except Exception:
                    pass
    if best_eval_metrics is None:
        ee, dd, sc, to_info, te = eval_cpgappo_once(agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS, use_shield=use_shield, use_cp=use_cp)
        best_eval_metrics = {'e': ee, 'd': dd, 'score': sc, 'total_energy': te, 'timeout_info': dict(to_info)}
    if checkpoint_dir is not None:
        try:
            torch.save(agent.policy.state_dict(), checkpoint_dir / f'{algo_tag}_last.pt')
        except Exception as ckpt_err:
            raise
    sys.stdout.flush()
    ti = best_eval_metrics['timeout_info']
    return (best_eval_metrics['e'], best_eval_metrics['d'], {'app_timeout_rate': float(ti.get('app_timeout_rate', 1.0)), 'task_timeout_rate': float(ti.get('task_timeout_rate', 1.0)), 'score': float(best_eval_metrics['score']), 'total_energy': float(best_eval_metrics['total_energy']), 'inference_time_ms': float(np.mean(agent.inference_times)) if len(agent.inference_times) > 0 else 0.0, 'action_stats': ti.get('action_stats', {}), 'timeout_rate': {'app_timeout_rate': float(ti.get('app_timeout_rate', 1.0)), 'task_timeout_rate': float(ti.get('task_timeout_rate', 1.0)), 'action_stats': ti.get('action_stats', {}), 'subtask_stats': ti.get('subtask_stats', {})}, 'subtask_stats': ti.get('subtask_stats', {})})

def train_cpgappo_dual_cpgappo(gpu_id, seed_offset, use_heuristic_sort=True, episodes=20, lr=0.0003, entropy_coef=0.02, lambda_guide=0.1):
    e, d, metrics = train_cpgappo(gpu_id=gpu_id, seed_offset=seed_offset, episodes=episodes, lr=lr, entropy_coef=entropy_coef, lambda_guide=lambda_guide, use_backward=True, use_shield=True, use_app_credit=True, use_cp=True, algo_tag='CPGAPPO')
    return (float(e), float(d), metrics)

def train_cpgappo_dual_wo_guidece(gpu_id, seed_offset, use_heuristic_sort=True, episodes=20, lr=0.0003, entropy_coef=0.02, lambda_guide=0.1):
    e, d, metrics = train_cpgappo(gpu_id=gpu_id, seed_offset=seed_offset, episodes=episodes, lr=lr, entropy_coef=entropy_coef, lambda_guide=0.0, use_backward=True, use_shield=True, use_app_credit=True, use_cp=True, algo_tag='CPGAPPO_noguidece')
    return (float(e), float(d), metrics)

def train_cpgappo_dual_wo_shield(gpu_id, seed_offset, use_heuristic_sort=True, episodes=20, lr=0.0003, entropy_coef=0.02, lambda_guide=0.1):
    e, d, metrics = train_cpgappo(gpu_id=gpu_id, seed_offset=seed_offset, episodes=episodes, lr=lr, entropy_coef=entropy_coef, lambda_guide=lambda_guide, use_backward=True, use_shield=False, use_app_credit=True, use_cp=True, algo_tag='CPGAPPO_noshield')
    return (float(e), float(d), metrics)

def train_cpgappo_dual_wo_appcredit(gpu_id, seed_offset, use_heuristic_sort=True, episodes=20, lr=0.0003, entropy_coef=0.02, lambda_guide=0.1):
    e, d, metrics = train_cpgappo(gpu_id=gpu_id, seed_offset=seed_offset, episodes=episodes, lr=lr, entropy_coef=entropy_coef, lambda_guide=lambda_guide, use_backward=True, use_shield=True, use_app_credit=False, use_cp=True, algo_tag='CPGAPPO_noappcredit')
    return (float(e), float(d), metrics)

def train_cpgappo_dual_wo_cpseq(gpu_id, seed_offset, use_heuristic_sort=True, episodes=20, lr=0.0003, entropy_coef=0.02, lambda_guide=0.1):
    e, d, metrics = train_cpgappo(gpu_id=gpu_id, seed_offset=seed_offset, episodes=episodes, lr=lr, entropy_coef=entropy_coef, lambda_guide=lambda_guide, use_backward=True, use_shield=True, use_app_credit=True, use_cp=False, algo_tag='CPGAPPO_nocp')
    return (float(e), float(d), metrics)

def train_cpgappo_dual_forward_only(gpu_id, seed_offset, use_heuristic_sort=True, episodes=20, lr=0.0003, entropy_coef=0.02, lambda_guide=0.1):
    e, d, metrics = train_cpgappo(gpu_id=gpu_id, seed_offset=seed_offset, episodes=episodes, lr=lr, entropy_coef=entropy_coef, lambda_guide=lambda_guide, use_backward=False, use_shield=True, use_app_credit=True, use_cp=True, algo_tag='CPGAPPO_fwdonly')
    return (float(e), float(d), metrics)

def train_cpgappo_dual_all_off(gpu_id, seed_offset, use_heuristic_sort=True, episodes=20, lr=0.0003, entropy_coef=0.02, lambda_guide=0.1):
    e, d, metrics = train_cpgappo(gpu_id=gpu_id, seed_offset=seed_offset, episodes=episodes, lr=lr, entropy_coef=entropy_coef, lambda_guide=0.0, use_backward=True, use_shield=False, use_app_credit=False, use_cp=False, use_full_credit=True, algo_tag='CPGAPPO_alloff')
    return (float(e), float(d), metrics)
