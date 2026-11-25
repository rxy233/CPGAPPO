import os
import sys
import json
import time
import traceback
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import torch
from pathlib import Path
from experiments.exp_utils import CONFIG, init_worker, generate_arrival_plan, load_deadline_config, calc_timeout_rate, safe_rest_tasks_total, all_arrived_done, get_graph_cache, apply_arrival_plan, subtask_partition_stats, load_model_bundle, get_feature_dim, graph_state_to_vector, get_task_size_bytes
from Environment.environment import Environment
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from Algorithms.Benchmark import Benchmark as BenchmarkClass
from utils.constant import para
EVALUATION_ARRIVAL_SEED = None
DEADLINE_SLOT = 55
BURST_PROB = 0.2
BURST_SIZE = 55
MAX_STEPS = 8000
STOP_ARRIVAL_STEP = 2000
DAG_MATRIX = {'chain': os.path.normpath(os.path.join(_PROJECT_ROOT, 'matrix', 'matrix_60_chain.txt')), 'default': os.path.normpath(os.path.join(_PROJECT_ROOT, 'matrix', 'matrix_60.txt')), 'wide': os.path.normpath(os.path.join(_PROJECT_ROOT, 'matrix', 'matrix_60_wide.txt'))}
_OUT_ROOT_DEFAULT = os.path.join(_PROJECT_ROOT, 'results')
OUT_ROOT = os.environ.get('CPGAPPO_DALL_OUT_ROOT', '')
if not OUT_ROOT:
    import glob as _glob
    _cands = sorted(_glob.glob(os.path.join(_OUT_ROOT_DEFAULT, 'main_comparison_*')))
    OUT_ROOT = _cands[-1] if _cands else _OUT_ROOT_DEFAULT

def _build_env(matrix_path, run_dir, seed):
    """Build a seeded environment and the requested arrival trace.

    Environment construction uses seed; EVALUATION_ARRIVAL_SEED overrides
    only the arrival plan for held-out evaluation. Nonlearning policies
    call Benchmark.get_actions; GATDQN uses get_stable_active_users.
    DGMA-adapt, TransEdgeStyle, and PPO iterate GraphScheduler.get_tasks
    directly. Every placement is committed through Benchmark.step."""
    os.environ['MATRIX_OVERRIDE_PATH'] = matrix_path
    para['deadline_slot'] = DEADLINE_SLOT
    CONFIG['RUN_DIR'] = run_dir
    CONFIG['BURST_PROB'] = BURST_PROB
    CONFIG['BURST_SIZE'] = BURST_SIZE
    CONFIG['MAX_STEPS'] = MAX_STEPS
    CONFIG['STOP_ARRIVAL_STEP'] = STOP_ARRIVAL_STEP
    CONFIG['SEED'] = 0
    init_worker(seed, para, CONFIG)
    device = torch.device('cuda:0' if torch.cuda.is_available() else torch.device('cpu'))
    user_num = para['user_num']
    subgraph_num = 20
    basegraph_num = 60
    task_complex = para['task_complex']
    _max_steps = CONFIG['MAX_STEPS']
    if isinstance(task_complex, (list, tuple)):
        task_complex_index = (CONFIG['SEED'] + seed) % len(task_complex)
    else:
        task_complex_index = int(task_complex) if isinstance(task_complex, int) else 0
    env = Environment(user_num, subgraph_num, basegraph_num, task_complex_index)
    env.decision_timing = True
    env_seed = CONFIG['SEED'] + seed
    arrival_plan = generate_arrival_plan(env_seed if EVALUATION_ARRIVAL_SEED is None else EVALUATION_ARRIVAL_SEED, _max_steps, CONFIG['STOP_ARRIVAL_STEP'], 0.3, CONFIG.get('BURST_PROB', 0.15), max(1, CONFIG.get('BURST_SIZE', 4) // 2), CONFIG.get('BURST_SIZE', 4))
    env.generate_components(seed=env_seed)
    G = get_graph_cache(user_num, subgraph_num, basegraph_num, _PROJECT_ROOT)
    if G is not None and env.basegraph:
        env.basegraph.nx_graph = G
    deadline_config = load_deadline_config(run_dir)
    ts = TaskScheduler(user_num, subgraph_num, basegraph_num, env, tight_deadline_config=deadline_config, seed=env_seed)
    gs = GraphScheduler(env.basegraph, env.subgraph_list, ts)
    bc = BenchmarkClass(env, gs, ts, task_complex_index, effective=True, seed=env_seed)
    ts.env = env
    ts.using_Algorithm = -1
    bc.reset()
    ts.reset()
    return {'env': env, 'ts': ts, 'gs': gs, 'bc': bc, 'arrival_plan': arrival_plan, 'task_complex_index': task_complex_index, 'max_steps': _max_steps, 'device': device, 'user_num': user_num}

def write_timing(ts, run_dir, checkpoint='selected'):
    import csv
    if ts.decision_timer is None or not ts.decision_timer.samples:
        return
    target = Path(run_dir) / 'timing'
    target.mkdir(parents=True, exist_ok=True)
    path = target / f'{checkpoint}_arrival{EVALUATION_ARRIVAL_SEED}.csv'
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=['slot', 'decisions', 'decision_ms'])
        writer.writeheader()
        writer.writerows(ts.decision_timer.samples)

def _assemble_result(algo, dag, seed, e_succ, d_succ, e_all, d_all, to_info, total_energy, ckpt_used, ckpt_path, elapsed):
    rho_app_pct = float(to_info['app_timeout_rate']) * 100.0
    rho_task_pct = float(to_info.get('task_timeout_rate', 0)) * 100.0
    return {'status': 'ok', 'algorithm': algo, 'dag': dag, 'seed': seed, 'D_succ': float(d_succ), 'D_all': float(d_all), 'E_succ': float(e_succ), 'E_all': float(e_all), 'AppTO': float(rho_app_pct), 'TaskTO': float(rho_task_pct), 'energy': float(e_succ), 'total_energy': float(total_energy), 'ckpt_used': ckpt_used, 'ckpt_path': ckpt_path, 'config': {'timeout_charge_mode': 'deadline', 'deadline_slot': DEADLINE_SLOT, 'matrix_path': DAG_MATRIX[dag], 'burst_prob': BURST_PROB, 'burst_size': BURST_SIZE, 'max_steps': MAX_STEPS, 'stop_arrival_step': STOP_ARRIVAL_STEP}, 'action_stats': to_info.get('action_stats', {}), 'subtask_stats': to_info.get('subtask_stats', {}), 'elapsed_sec': round(elapsed, 1)}

def reeval_dgma_adapt(dag, seed, use_heuristic_sort=True):
    from Algorithms.StrongBaselines.dgma_adapt_agent import DGMAAdaptAgent
    from Algorithms.StrongBaselines.dgma_adapt_core import extract_dgma_adapt_state
    t_start = time.time()
    run_dir = os.path.join(OUT_ROOT, f'{dag}__DGMA_adapt__seed{seed}')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, 'DGMA_adapt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'DGMA_adapt ckpt not found: {ckpt_path}')
    ctx = _build_env(DAG_MATRIX[dag], run_dir, seed)
    env = ctx['env']
    device = ctx['device']
    try:
        s_data, _ = extract_dgma_adapt_state(ctx['ts'], (0, 0), slot=0, task_complex_index=ctx['task_complex_index'])
        node_dim = s_data.x.shape[1]
    except Exception as e:
        node_dim = 27
    action_dim = para['edge_num'] + 2
    hidden_dim = 64
    batch_size = 64
    buffer_capacity = 20000
    agent = DGMAAdaptAgent(node_dim=node_dim, action_dim=action_dim, device=device, hidden_dim=hidden_dim, num_layers=2, lr_actor=0.0003, lr_critic=0.0003, gamma=0.99, tau=0.005, gumbel_tau=1.0, batch_size=batch_size, buffer_capacity=buffer_capacity, per_alpha=0.6, per_beta0=0.4, per_beta_anneal_steps=10000, updates_per_step=1, min_updates_after=max(256, batch_size * 4), grad_clip=1.0)
    best_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    agent.actor.load_state_dict(best_state)
    agent.actor.eval()
    user_num = para['user_num']
    env_seed = CONFIG['SEED'] + seed
    deadline_config = load_deadline_config(run_dir)
    ts_eval = TaskScheduler(user_num, 20, 60, env, tight_deadline_config=deadline_config, seed=env_seed)
    gs_eval = GraphScheduler(env.basegraph, env.subgraph_list, ts_eval)
    bc_eval = BenchmarkClass(env, gs_eval, ts_eval, ctx['task_complex_index'], effective=True, seed=env_seed)
    ts_eval.env = env
    ts_eval.using_Algorithm = -1
    bc_eval.reset()
    ts_eval.reset()
    ts_eval.route_stats = {'local': 0, 'edge': 0, 'cloud': 0, 'recent': []}
    eval_task2action = {}
    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts_eval, slot, ctx['arrival_plan'])
        ts_eval.check_timeouts(slot)
        tasks = gs_eval.get_tasks(slot, sort_tasks=use_heuristic_sort)
        if not tasks:
            if slot >= STOP_ARRIVAL_STEP and safe_rest_tasks_total(ts_eval.rest_tasks) == 0 and all_arrived_done(ts_eval):
                break
            continue
        for task in tasks:
            s_data, mask = extract_dgma_adapt_state(ts_eval, task, slot=slot, task_complex_index=ctx['task_complex_index'])
            if not np.any(mask):
                continue
            act, _ = agent.take_action(s_data, action_mask=mask, deterministic=True)
            bc_eval.step([[task, act]])
            eval_task2action[tuple(task)] = int(act)
    ts_eval.finalize_episode(MAX_STEPS - 1)
    e_succ, d_succ = ts_eval.get_avg_results(only_successful=True)
    e_all, d_all = ts_eval.get_avg_results(only_successful=False, timeout_charge='deadline')
    to_info = calc_timeout_rate(ts_eval)
    app_to = float(to_info['app_timeout_rate'])
    task_to = float(to_info.get('task_timeout_rate', app_to))
    if app_to > 1.0:
        app_to /= 100.0
    if task_to > 1.0:
        task_to /= 100.0
    to_info['app_timeout_rate'] = app_to
    to_info['task_timeout_rate'] = task_to
    partition = subtask_partition_stats(ts_eval, eval_task2action)
    to_info['action_stats'] = {'local': partition['local'], 'cloud': partition['cloud'], 'edge': partition['edge'], 'timeout': partition['timeout'], 'unknown': partition['unknown'], 'total': partition['total_subtasks'], 'total_actions': len(eval_task2action)}
    write_timing(ts_eval, run_dir)
    total_e = ts_eval.get_sum_energy()
    elapsed = time.time() - t_start
    result = _assemble_result('DGMA_adapt', dag, seed, e_succ, d_succ, e_all, d_all, to_info, total_e, 'best', ckpt_path, elapsed)
    return result

def reeval_transedge(dag, seed, use_heuristic_sort=False):
    from Algorithms.StrongBaselines.transedge_style_agent import TransEdgeStylePPOAgent
    from Algorithms.StrongBaselines.transedge_style_core import extract_transedge_state
    t_start = time.time()
    run_dir = os.path.join(OUT_ROOT, f'{dag}__TransEdgeStyle__seed{seed}')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, 'TransEdgeStyle.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'TransEdgeStyle ckpt not found: {ckpt_path}')
    ctx = _build_env(DAG_MATRIX[dag], run_dir, seed)
    ts = ctx['ts']
    env = ctx['env']
    device = ctx['device']
    try:
        s_data, _ = extract_transedge_state(ts, (0, 0), slot=0, task_complex_index=ctx['task_complex_index'])
        state_dim = s_data.x.shape[1]
    except Exception as e:
        state_dim = 52
    action_dim = para['edge_num'] + 2
    agent = TransEdgeStylePPOAgent(node_dim=state_dim, action_dim=action_dim, device=device, lr=0.0003, gamma=0.99, lmbda=0.95, eps_clip=0.2, K_epochs=4, entropy_coef=0.01)
    best_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    agent.policy.load_state_dict(best_state)
    agent.policy.eval()
    user_num = para['user_num']
    env_seed = CONFIG['SEED'] + seed
    deadline_config = load_deadline_config(run_dir)
    ts_eval = TaskScheduler(user_num, 20, 60, env, tight_deadline_config=deadline_config, seed=env_seed)
    gs_eval = GraphScheduler(env.basegraph, env.subgraph_list, ts_eval)
    bc_eval = BenchmarkClass(env, gs_eval, ts_eval, ctx['task_complex_index'], effective=True, seed=env_seed)
    ts_eval.env = env
    ts_eval.using_Algorithm = -1
    bc_eval.reset()
    ts_eval.reset()
    ts_eval.route_stats = {'local': 0, 'edge': 0, 'cloud': 0, 'recent': []}
    eval_task2action = {}
    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts_eval, slot, ctx['arrival_plan'])
        ts_eval.check_timeouts(slot)
        tasks = gs_eval.get_tasks(slot, sort_tasks=use_heuristic_sort)
        if not tasks:
            if slot >= STOP_ARRIVAL_STEP and safe_rest_tasks_total(ts_eval.rest_tasks) == 0 and all_arrived_done(ts_eval):
                break
            continue
        for task in tasks:
            s_data, mask = extract_transedge_state(ts_eval, task, slot=slot, task_complex_index=ctx['task_complex_index'])
            if not np.any(mask):
                continue
            act, _ = agent.take_action(s_data, action_mask=mask, deterministic=True, route_stats=getattr(ts_eval, 'route_stats', None))
            bc_eval.step([[task, act]])
            eval_task2action[tuple(task)] = int(act)
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
    e_all, d_all = ts_eval.get_avg_results(only_successful=False, timeout_charge='deadline')
    to_info = calc_timeout_rate(ts_eval)
    app_to = float(to_info['app_timeout_rate'])
    task_to = float(to_info.get('task_timeout_rate', app_to))
    if app_to > 1.0:
        app_to /= 100.0
    if task_to > 1.0:
        task_to /= 100.0
    to_info['app_timeout_rate'] = app_to
    to_info['task_timeout_rate'] = task_to
    partition = subtask_partition_stats(ts_eval, eval_task2action)
    to_info['action_stats'] = {'local': partition['local'], 'cloud': partition['cloud'], 'edge': partition['edge'], 'timeout': partition['timeout'], 'unknown': partition['unknown'], 'total': partition['total_subtasks'], 'total_actions': len(eval_task2action)}
    write_timing(ts_eval, run_dir)
    total_e = ts_eval.get_sum_energy()
    elapsed = time.time() - t_start
    result = _assemble_result('TransEdgeStyle', dag, seed, e_succ, d_succ, e_all, d_all, to_info, total_e, 'best', ckpt_path, elapsed)
    return result

def reeval_gatdqn(dag, seed):
    from Algorithms.GNNRL.dynamic_gat_dqn import DynamicGAT_DQN_Agent as DynamicGATDQN
    from Algorithms.Train.train_dynamic_gat_dqn import get_stable_active_users, get_global_graph_state_dag, construct_action_mask, GRAPH_CACHE
    t_start = time.time()
    run_dir = os.path.join(OUT_ROOT, f'{dag}__GATDQN__seed{seed}')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, 'dynamic_gat_dqn_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'GATDQN ckpt not found: {ckpt_path}')
    ctx = _build_env(DAG_MATRIX[dag], run_dir, seed)
    env = ctx['env']
    ts = ctx['ts']
    gs = ctx['gs']
    bc = ctx['bc']
    device = ctx['device']
    base_node_dim = None
    for uid in range(min(10, para['user_num'])):
        try:
            sg_data, _ = gs.get_app_dag_data(env, uid, 0, ctx['task_complex_index'])
            if sg_data is not None and hasattr(sg_data, 'x') and (sg_data.x is not None):
                base_node_dim = int(sg_data.x.shape[1])
                break
        except Exception:
            pass
    if base_node_dim is None:
        base_node_dim = 10
    node_dim = base_node_dim + 2 + para['edge_num']
    action_dim = para['edge_num'] + 2
    agent = DynamicGATDQN(node_dim, action_dim, device, use_expert_data=False, expert_ratio=0.0)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    agent.policy_gpu.load_state_dict(checkpoint['policy_gpu'])
    agent.policy_cpu.load_state_dict(checkpoint['policy_gpu'])
    ts.reset()
    bc.reset()
    GRAPH_CACHE.reset()
    agent.policy_cpu.load_state_dict(agent.policy_gpu.state_dict())
    agent.policy_cpu.eval()
    eval_action_count = [0] * action_dim
    eval_total_actions = 0
    topk = 30
    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts, slot, ctx['arrival_plan'])
        ts.check_timeouts(slot)
        inner_steps = 0
        while True:
            active_users, ready_tasks = get_stable_active_users(ts, gs, slot, k=topk)
            if not active_users:
                break
            (state_x, state_edge_index, state_batch), task_to_global_idx, global_idx_to_task = get_global_graph_state_dag(env, gs, active_users, ctx['task_complex_index'], slot, base_node_dim)
            if state_x is None:
                break
            ready_mask = torch.zeros(state_x.shape[0], dtype=torch.bool)
            for t in ready_tasks:
                if t in task_to_global_idx:
                    ready_mask[task_to_global_idx[t]] = True
            if not ready_mask.any():
                break
            full_action_mask = construct_action_mask(ts, ready_tasks, task_to_global_idx, action_dim, state_x.shape[0], slot)
            node_idx, offload_action = agent.select_action((state_x, state_edge_index, state_batch), ready_mask, full_action_mask, training=False, custom_eps=0.0)
            if node_idx is None:
                break
            selected_task = global_idx_to_task[int(node_idx)]
            if selected_task is None:
                break
            bc.step([[selected_task, int(offload_action)]])
            eval_action_count[int(offload_action)] += 1
            eval_total_actions += 1
            inner_steps += 1
            if inner_steps >= 100:
                break
        if slot >= STOP_ARRIVAL_STEP and all_arrived_done(ts) and (safe_rest_tasks_total(ts.rest_tasks) == 0):
            break
    ts.finalize_episode(slot)
    e_succ, d_succ = ts.get_avg_results(only_successful=True)
    e_all, d_all = ts.get_avg_results(only_successful=False, timeout_charge='deadline')
    to_info = calc_timeout_rate(ts)
    app_to = float(to_info['app_timeout_rate'])
    task_to = float(to_info.get('task_timeout_rate', app_to))
    if app_to > 1.0:
        app_to /= 100.0
    if task_to > 1.0:
        task_to /= 100.0
    to_info['app_timeout_rate'] = app_to
    to_info['task_timeout_rate'] = task_to
    to_info['action_stats'] = {'local': eval_action_count[0], 'cloud': eval_action_count[1], 'edge': sum(eval_action_count[2:]), 'total_actions': eval_total_actions}
    write_timing(ts, run_dir)
    total_e = ts.get_sum_energy()
    elapsed = time.time() - t_start
    result = _assemble_result('GATDQN', dag, seed, e_succ, d_succ, e_all, d_all, to_info, total_e, 'best', ckpt_path, elapsed)
    return result

def reeval_ppo(dag, seed):
    from Algorithms.PPO.ppo_gae import PPO_GAE
    t_start = time.time()
    run_dir = os.path.join(OUT_ROOT, f'{dag}__PPO__seed{seed}')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, 'PPO.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'PPO ckpt not found: {ckpt_path}')
    ctx = _build_env(DAG_MATRIX[dag], run_dir, seed)
    env = ctx['env']
    ts = ctx['ts']
    gs = ctx['gs']
    bc = ctx['bc']
    device = ctx['device']
    n_states = get_feature_dim(env, gs, ctx['task_complex_index'])
    n_actions = para['edge_num'] + 2
    agent = PPO_GAE(n_states=n_states, n_hiddens=min(n_states * 3, 768), n_actions=n_actions, actor_lr=0.0003, critic_lr=0.001, lmbda=0.95, epochs=5, eps=0.2, gamma=0.99, device=device, batch_size=512, ent_coef=0.05)
    bundle = load_model_bundle(ckpt_path, device)
    if bundle is None or 'model_state' not in bundle:
        pth_path = os.path.join(ckpt_dir, 'ppo_model.pth')
        if os.path.exists(pth_path):
            actor_state = torch.load(pth_path, map_location=device, weights_only=True)
            agent.actor.load_state_dict(actor_state)
        else:
            raise RuntimeError(f'Cannot load PPO ckpt from {ckpt_dir}')
    else:
        ms = bundle['model_state']
        if 'actor' in ms:
            agent.actor.load_state_dict(ms['actor'])
        if 'critic' in ms:
            agent.critic.load_state_dict(ms['critic'])
        pass
    agent.actor.eval()
    agent.critic.eval()
    ts.reset()
    bc.reset()
    eval_action_count = [0] * n_actions
    eval_total_actions = 0
    task2action = {}
    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts, slot, ctx['arrival_plan'])
        ts.check_timeouts(slot)
        if slot >= STOP_ARRIVAL_STEP:
            if all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
                break
        tasks = gs.get_tasks(slot, sort_tasks=False)
        if not tasks:
            continue
        for task in tasks:
            state_data = gs.get_graph_state_new(env, task, ctx['task_complex_index'], slot=slot)
            state = graph_state_to_vector(state_data, method='mean')
            if hasattr(state, 'to'):
                state = state.to(agent.device)
            user_id, subtask_id = task
            task_size_bytes = get_task_size_bytes(ts, user_id, subtask_id)
            now_time = slot * para['slot_interval']
            base_mask = ts.get_action_mask(user_id, task_size_bytes, now_time)
            if not bool(base_mask.any()):
                continue
            mask_bin = [float(bool(m)) for m in base_mask]
            action, *rest = agent.take_action(state, f'{user_id}_{subtask_id}', action_mask=mask_bin, deterministic=True)
            bc.step([[task, int(action)]])
            task2action[tuple(task)] = int(action)
            if 0 <= action < n_actions:
                eval_action_count[action] += 1
            eval_total_actions += 1
    ts.finalize_episode(slot)
    e_succ, d_succ = ts.get_avg_results(only_successful=True)
    e_all, d_all = ts.get_avg_results(only_successful=False, timeout_charge='deadline')
    to_info = calc_timeout_rate(ts)
    app_to = float(to_info['app_timeout_rate'])
    task_to = float(to_info.get('task_timeout_rate', app_to))
    if app_to > 1.0:
        app_to /= 100.0
    if task_to > 1.0:
        task_to /= 100.0
    to_info['app_timeout_rate'] = app_to
    to_info['task_timeout_rate'] = task_to
    partition = subtask_partition_stats(ts, task2action)
    to_info['action_stats'] = {'local': partition['local'], 'cloud': partition['cloud'], 'edge': partition['edge'], 'timeout': partition['timeout'], 'unknown': partition['unknown'], 'total': partition['total_subtasks'], 'total_actions': len(task2action)}
    write_timing(ts, run_dir)
    total_e = ts.get_sum_energy()
    elapsed = time.time() - t_start
    result = _assemble_result('PPO', dag, seed, e_succ, d_succ, e_all, d_all, to_info, total_e, 'best', ckpt_path, elapsed)
    return result
BENCH_ALGO_MAP = {'HybridPSOGA': 8, 'Genetic': 6, 'Greedy': 4, 'Edge': 2, 'Local': 0}
BENCH_LABEL_TO_MODE = {'HybridPSOGA': 'HybridPSOGA', 'Genetic': 'Genetic', 'Greedy': 'Greedy', 'Edge-only': 'Edge', 'Local-only': 'Local'}

def reeval_bench(algo_label, dag, seed):
    mode = BENCH_LABEL_TO_MODE.get(algo_label)
    if mode is None:
        raise ValueError(f"Unknown bench algo '{algo_label}', expected one of {list(BENCH_LABEL_TO_MODE.keys())}")
    using_alg = BENCH_ALGO_MAP[mode]
    t_start = time.time()
    run_dir = os.path.join(OUT_ROOT, f'{dag}__{algo_label}__seed{seed}')
    ctx = _build_env(DAG_MATRIX[dag], run_dir, seed)
    env = ctx['env']
    ts = ctx['ts']
    gs = ctx['gs']
    bc = ctx['bc']
    bc = BenchmarkClass(env, gs, ts, ctx['task_complex_index'], effective=True, seed=CONFIG['SEED'] + seed, task_order_mode='random')
    ts.env = env
    ts.using_Algorithm = using_alg
    bc.reset()
    ts.reset()
    task2action = {}
    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts, slot, ctx['arrival_plan'])
        ts.check_timeouts(slot)
        if slot >= STOP_ARRIVAL_STEP:
            if all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
                break
        actions, _ = bc.get_actions(ts.using_Algorithm, slot, 0)
        if actions:
            bc.step(actions)
        for entry in actions:
            if len(entry) >= 2:
                task, action_raw = (entry[0], entry[1])
                try:
                    action = int(action_raw)
                except Exception:
                    action = 0
                task2action[tuple(task)] = action
    ts.finalize_episode(slot)
    e_succ, d_succ = ts.get_avg_results(only_successful=True)
    e_all, d_all = ts.get_avg_results(only_successful=False, timeout_charge='deadline')
    to_info = calc_timeout_rate(ts)
    app_to = float(to_info['app_timeout_rate'])
    task_to = float(to_info.get('task_timeout_rate', app_to))
    if app_to > 1.0:
        app_to /= 100.0
    if task_to > 1.0:
        task_to /= 100.0
    to_info['app_timeout_rate'] = app_to
    to_info['task_timeout_rate'] = task_to
    partition = subtask_partition_stats(ts, task2action)
    to_info['action_stats'] = {'local': partition['local'], 'cloud': partition['cloud'], 'edge': partition['edge'], 'timeout': partition['timeout'], 'unknown': partition['unknown'], 'total': partition['total_subtasks'], 'total_actions': len(task2action)}
    write_timing(ts, run_dir)
    total_e = ts.get_sum_energy()
    elapsed = time.time() - t_start
    result = _assemble_result(algo_label, dag, seed, e_succ, d_succ, e_all, d_all, to_info, total_e, 'n/a (heuristic)', 'n/a', elapsed)
    return result
REEVAL_FNS = {'DGMA_adapt': reeval_dgma_adapt, 'TransEdgeStyle': reeval_transedge, 'GATDQN': reeval_gatdqn, 'PPO': reeval_ppo, 'HybridPSOGA': reeval_bench, 'Genetic': reeval_bench, 'Greedy': reeval_bench, 'Edge-only': reeval_bench, 'Local-only': reeval_bench}
BENCH_ALGOS = {'HybridPSOGA', 'Genetic', 'Greedy', 'Edge-only', 'Local-only'}

def reeval_one(algo, dag, seed):
    fn = REEVAL_FNS.get(algo)
    if fn is None:
        raise ValueError(f"Unknown algo '{algo}', expected one of {list(REEVAL_FNS.keys())}")
    try:
        if algo in BENCH_ALGOS:
            return fn(algo, dag, seed)
        else:
            return fn(dag, seed)
    except Exception as ex:
        err_msg = f'{type(ex).__name__}: {ex}'
        traceback.print_exc()
        err_result = {'status': 'error', 'algorithm': algo, 'dag': dag, 'seed': seed, 'error': err_msg, 'traceback': traceback.format_exc()}
        run_dir = os.path.join(OUT_ROOT, f'{dag}__{algo}__seed{seed}')
        try:
            with open(os.path.join(run_dir, 'result_dall.json'), 'w', encoding='utf-8') as f:
                json.dump(err_result, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return err_result
