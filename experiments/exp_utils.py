import os
import sys
import json
import time
import random
import csv
import copy
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from utils.constant import para
CONFIG = {'NUM_PROCESSES': 4, 'MAX_RL_CONCURRENT': 1, 'MAX_STEPS': 8000, 'STOP_ARRIVAL_STEP': 2000, 'ARRIVAL_INTERVAL': 1, 'BURST_MODE': True, 'BURST_PROB': 0.2, 'BURST_SIZE': 55, 'SEED': 0, 'RESULTS_BASE_DIR': os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')), 'GPU_IDS': [0, 0, 0], 'SCORE_WINDOW': 20, 'USE_EXPERT_DATA': True, 'EXPERT_RATIO': 0.3, 'TOPK_TASKS': 30, 'SMART_TOPK': True, 'ADAPTIVE_TOPK': True, 'TOPK_MIN': 15, 'TOPK_MAX': 30, 'AUTO_EVAL_AFTER_TRAIN': False, 'EVAL_EVERY': 10, 'EVAL_EPISODES': 1, 'EVAL_USE_BEST': True, 'CURVE_RETRY': 10, 'SLA0': 0.99, 'KAPPA': 3.0, 'V_CAP_OVER': 2.0}

def clamp01(x):
    return float(max(0.0, min(1.0, x)))
NEG_INF = -1000000000.0

def mask_has_any_valid(mask):
    import numpy as np
    m = np.array(mask, dtype=np.float32)
    return bool((m > NEG_INF / 2).any())

def mask_allows(mask, action: int):
    import numpy as np
    m = np.array(mask, dtype=np.float32)
    if action < 0 or action >= len(m):
        return False
    return bool(m[action] > NEG_INF / 2)

def normalize_mask(mask, action_dim, force_fallback=None):
    import numpy as np
    m = np.array(mask, dtype=np.float32)
    if len(m) != action_dim:
        if len(m) > action_dim:
            m = m[:action_dim]
        else:
            pad = np.zeros(action_dim - len(m))
            m = np.concatenate([m, pad])
    m_min, m_max = (float(m.min()), float(m.max()))
    if m_min < -1000000.0:
        pass
    elif m_max > 0.5:
        m = np.where(m > 0.5, 0.0, NEG_INF).astype(np.float32)
    else:
        pass
    has_valid = (m > NEG_INF / 2).any()
    if not has_valid:
        fallback_action = force_fallback if force_fallback is not None else 1
        fallback_action = max(0, min(fallback_action, action_dim - 1))
        if not hasattr(normalize_mask, '_warned'):
            normalize_mask._warned = True
        m[:] = NEG_INF
        m[fallback_action] = 0.0
    return m

def compute_score(E_avg, D_succ_avg, rho_app, rho_task, E_min=0.3, E_max=3.0, D_min=0.7, D_max=5.0, phi1=0.5, phi2=0.5, kappa=2.0, eta=20.0, lam=2.0, sla0=0.95, v_cap=3.0, v_cap_over=3.0):
    """Return the checkpoint-selection cost; smaller values are preferred.

    rho_app is the recorded application noncompletion fraction and
    rho_task is TaskTimeout. The sla0 compatibility name specifies the
    tolerated noncompletion threshold through 1 - sla0. It does not
    convert rho_app into a strict application deadline-violation rate."""
    eps = 1e-12
    e_hat = np.clip((E_avg - E_min) / max(E_max - E_min, eps), 0.0, 1.0)
    d_hat = np.clip((D_succ_avg - D_min) / max(D_max - D_min, eps), 0.0, 1.0)
    base_cost = phi1 * e_hat + phi2 * d_hat
    delta = 1.0 - sla0
    v = rho_app / max(delta, 1e-06)
    if v <= 1.0:
        score = base_cost
    else:
        violation = v - 1.0
        penalty = eta * violation ** 2
        score = base_cost * (1.0 + violation) + penalty
    score += lam * rho_task
    return float(score)
import numpy as np

def to_scalar(val, default=0.0):
    if isinstance(val, torch.Tensor):
        val = val.item()
    if isinstance(val, (np.ndarray, list, tuple)):
        if len(val) == 0:
            return default
        return to_scalar(val[0], default)
    if val is None or pd.isna(val):
        return default
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return default
    return val

def save_model_bundle(algo_name, agent, checkpoint_dir, **meta_kwargs):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir = checkpoint_dir.parent
    arrival_data = None
    deadline_data = None
    try:
        arr_path = run_dir / 'arrivals' / 'arrival_seed0.json'
        if arr_path.exists():
            arrival_data = json.loads(arr_path.read_text(encoding='utf-8'))
    except Exception:
        arrival_data = None
    try:
        dl_path = run_dir / 'configs' / 'deadlines.json'
        if dl_path.exists():
            deadline_data = json.loads(dl_path.read_text(encoding='utf-8'))
    except Exception:
        deadline_data = None
    model_state = {}
    optim_state = {}
    if hasattr(agent, 'q_net'):
        model_state['q_net'] = agent.q_net.state_dict()
        if hasattr(agent, 'target_net'):
            model_state['target_net'] = agent.target_net.state_dict()
        if hasattr(agent, 'optimizer'):
            optim_state['optimizer'] = agent.optimizer.state_dict()
    if hasattr(agent, 'actor'):
        model_state['actor'] = agent.actor.state_dict()
        if hasattr(agent, 'critic'):
            model_state['critic'] = agent.critic.state_dict()
        if hasattr(agent, 'actor_optimizer'):
            optim_state['actor_optimizer'] = agent.actor_optimizer.state_dict()
        if hasattr(agent, 'critic_optimizer'):
            optim_state['critic_optimizer'] = agent.critic_optimizer.state_dict()
    if hasattr(agent, 'policy'):
        model_state['policy'] = agent.policy.state_dict()
        model_key = 'policy'
    elif hasattr(agent, 'q'):
        model_state['q'] = agent.q.state_dict()
        model_key = 'q'
    train_state = {}
    if hasattr(agent, 'step_count'):
        train_state['step_count'] = agent.step_count
    if hasattr(agent, 'epsilon'):
        train_state['epsilon'] = agent._get_epsilon() if hasattr(agent, '_get_epsilon') else getattr(agent, 'epsilon', 0.0)
    bundle = {'bundle_version': 2, 'algo': algo_name, 'created_at': time.time(), 'meta': dict(meta_kwargs), 'CONFIG_snapshot': copy.deepcopy(CONFIG), 'para_snapshot': copy.deepcopy(para), 'arrival_data': arrival_data, 'deadline_data': deadline_data, 'rng_state': _get_rng_state(), 'model_state': model_state, 'optim_state': optim_state, 'train_state': train_state}
    bundle_path = checkpoint_dir / f'{algo_name}.pt'
    torch.save(bundle, bundle_path)
    try:
        if 'q_net' in model_state:
            torch.save(model_state['q_net'], checkpoint_dir / f'{algo_name.lower()}_model.pth')
        elif 'actor' in model_state:
            torch.save(model_state['actor'], checkpoint_dir / f'{algo_name.lower()}_model.pth')
        elif 'policy' in model_state:
            torch.save(model_state['policy'], checkpoint_dir / f'{algo_name.lower()}_model.pth')
    except Exception as e:
        pass
    return bundle_path

def _get_rng_state():
    s = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state().cpu()}
    if torch.cuda.is_available():
        s['torch_cuda_all'] = [x.cpu() for x in torch.cuda.get_rng_state_all()]
    else:
        s['torch_cuda_all'] = None
    return s

def get_task_size_bytes(ts, user_id, subtask_id):
    s = getattr(ts, 'task_size', None)
    if s is None:
        return 0.0
    try:
        if isinstance(s, dict):
            v = s.get(subtask_id, list(s.values())[0])
        elif isinstance(s, (list, tuple, np.ndarray)):
            if len(s) == 0:
                return 0.0
            if isinstance(s[0], (list, tuple, np.ndarray)):
                v = s[user_id][subtask_id]
            else:
                v = s[subtask_id]
        else:
            v = s
        return float(v)
    except Exception:
        return 0.0

def subtask_outcome_stats(ts):
    total = finished = unfinished = timeout = 0
    eps = getattr(ts, 'TIMEOUT_EPSILON', 1e-06)
    for uid in range(ts.user_num):
        if ts.enter_time[uid] == float('inf'):
            continue
        try:
            task_deadline_abs = ts.enter_time[uid] + ts.get_task_deadline_slot(uid) * para['slot_interval']
        except:
            continue
        for sid, ft in ts.finish_time[uid].items():
            total += 1
            if ft != float('inf'):
                finished += 1
                if ft > task_deadline_abs + eps:
                    timeout += 1
            else:
                unfinished += 1
                timeout += 1
    return (total, finished, unfinished, timeout)

def subtask_partition_stats(ts, task2action):
    local = cloud = edge = timeout = timeout_scheduled = timeout_unscheduled = unknown = 0
    total = 0
    for uid in range(ts.user_num):
        if ts.enter_time[uid] == float('inf'):
            continue
        for sid, ft in ts.finish_time[uid].items():
            total += 1
            if ft == float('inf'):
                timeout += 1
                if (uid, sid) in task2action:
                    timeout_scheduled += 1
                else:
                    timeout_unscheduled += 1
                continue
            a = task2action.get((uid, sid), None)
            if a is None:
                unknown += 1
            elif a == 0:
                local += 1
            elif a == 1:
                cloud += 1
            else:
                edge += 1
    return {'total_subtasks': total, 'local': local, 'cloud': cloud, 'edge': edge, 'timeout': timeout, 'timeout_scheduled': timeout_scheduled, 'timeout_unscheduled': timeout_unscheduled, 'unknown': unknown}

def calc_timeout_rate(ts):
    """Return Eq. (36) application noncompletion and Eq. (37) task timeout.

    The denominator includes only arrived applications. Membership in
    application_finished counts as completion even after the deadline.
    TaskTimeout includes unfinished and late-completed arrived tasks.

    app_timeout_rate and combined are compatible names for the same
    noncompletion fraction. app_timeout_count counts membership in the
    timeout-finished set, which can overlap the completed set; it is not
    generally the numerator of app_timeout_rate. unfinished_apps counts
    arrivals in neither terminal set. Rates are fractions, before export
    to percentages by run.py and the evaluation entry points."""
    arrived = {uid for uid in range(ts.user_num) if np.isfinite(ts.enter_time[uid])}
    completed = arrived & set(ts.application_finished)
    timed_out = arrived & set(ts.application_timeout_finished)
    pending = arrived - completed - timed_out
    app_rate = len(arrived - completed) / len(arrived) if arrived else 0.0
    total, finished, unfinished, timeout = subtask_outcome_stats(ts)
    task_rate = timeout / total if total else 0.0
    return {'combined': app_rate, 'app_timeout_rate': app_rate, 'app_finished_count': len(completed), 'app_timeout_count': len(timed_out), 'unfinished_apps': len(pending), 'task_timeout_rate': task_rate, 'subtask_stats': {'total': total, 'finished': finished, 'unfinished': unfinished, 'timeout': timeout}}

def get_simple_timeout_rate(ts):
    result = calc_timeout_rate(ts)
    return result['combined'] if isinstance(result, dict) else result

def calc_application_timeout_rate(ts):
    return get_simple_timeout_rate(ts)

def _mask_to_valid(mask, action_dim=None):
    m = mask
    if hasattr(m, 'detach'):
        m = m.detach().cpu().numpy()
    m = np.asarray(m, dtype=np.float32).reshape(-1)
    if action_dim is not None and m.shape[0] != action_dim:
        return np.ones((action_dim,), dtype=bool)
    m_min, m_max = (float(m.min()), float(m.max()))
    if m_min < -1000000.0:
        return m > NEG_INF
    elif m_max > 0.5:
        return m > 0.5
    else:
        return np.ones_like(m, dtype=bool)

def get_expert_action(env, ts, user_id, subtask_id, task_complex, slot):
    task_size_bytes = get_task_size_bytes(ts, user_id, subtask_id)
    task_size_kb = task_size_bytes / 1024.0
    slot_interval = None
    if hasattr(env, 'env_para') and isinstance(env.env_para, dict) and ('slot_interval' in env.env_para):
        slot_interval = float(env.env_para['slot_interval'])
    else:
        slot_interval = 0.01
    now_time = slot * slot_interval
    base_mask = ts.get_action_mask(user_id, task_size_bytes, now_time)
    action_dim = len(base_mask)
    valid = _mask_to_valid(base_mask, action_dim=action_dim)
    if not valid.any():
        return 1
    local_ok = bool(valid[0])
    cloud_ok = bool(valid[1]) if action_dim > 1 else False
    edge_actions = [i for i in range(2, action_dim) if valid[i]]

    def pick_best_edge(edge_list):
        if not edge_list:
            return None
        try:
            return min(edge_list, key=lambda a: sum((1 for x in ts.edge_useful[a - 2] if x != 0)))
        except Exception:
            return edge_list[0]
    if task_size_kb < 400:
        if local_ok:
            return 0
        best_edge = pick_best_edge(edge_actions)
        if best_edge is not None:
            return int(best_edge)
        return 1 if cloud_ok else 0
    elif task_size_kb < 650:
        best_edge = pick_best_edge(edge_actions)
        if best_edge is not None:
            return int(best_edge)
        if local_ok:
            return 0
        return 1 if cloud_ok else 0
    else:
        if cloud_ok:
            return 1
        best_edge = pick_best_edge(edge_actions)
        if best_edge is not None:
            return int(best_edge)
        return 0

def graph_state_to_vector(state_data, method='mean'):
    if state_data is None:
        return torch.zeros(32, dtype=torch.float32)
    if hasattr(state_data, 'x') and state_data.x is not None:
        x = state_data.x
        if x is None or x.numel() == 0:
            return torch.zeros(32, dtype=torch.float32)
        if x.dim() == 2:
            if method == 'max':
                return x.max(dim=0)[0].cpu()
            else:
                return x.mean(dim=0).cpu()
        elif x.dim() == 1:
            return x.view(-1).cpu()
        else:
            return x.view(-1).cpu()
    elif isinstance(state_data, torch.Tensor):
        if state_data.dim() == 2:
            if method == 'max':
                return state_data.max(dim=0)[0].cpu()
            else:
                return state_data.mean(dim=0).cpu()
        return state_data.view(-1).cpu()
    else:
        try:
            return torch.tensor(state_data, dtype=torch.float32).view(-1)
        except:
            return torch.zeros(32, dtype=torch.float32)

def get_feature_dim(env, gs, task_complex_index):
    if isinstance(task_complex_index, (list, tuple, np.ndarray)):
        task_complex_index = int(getattr(env, 'task_complex_index', 0))
    else:
        task_complex_index = int(task_complex_index)
    try:
        dummy_state = gs.get_graph_state_new(env, (0, 0), task_complex_index)
        vec = graph_state_to_vector(dummy_state)
        return vec.shape[0]
    except Exception as e:
        return 32

def get_graph_cache(user_num, subgraph_num, basegraph_num, project_root):
    return None

def generate_arrival_plan(seed: int, max_steps: int, stop_step: int, base_prob: float, burst_prob: float, burst_min: int, burst_max: int):
    """Initialize evenly spaced arrivals, then relocate future arrivals earlier.

    Shuffle user IDs with a local random.Random(seed), and place user i
    at floor(i * horizon / user_count), where horizon is stop_step clipped
    to [1, max_steps]. At each initialization slot, a Bernoulli burst_prob
    trial requests randint(burst_min, burst_max) future arrivals. Scan
    backward from the horizon, moving at most one user per visited future
    slot, until the requested count or available future users are exhausted.
    Each user is moved, never duplicated. Actual arrivals can occupy a
    much shorter span than the initialization horizon. Existing arrivals
    in a burst slot are additional to its relocated users.

    base_prob is an unused compatibility argument. The standard inputs
    are 150 users, horizon 2000, burst_prob=0.2, and burst bounds 27..55.
    All random draws use the local RNG; global RNG state is unaffected."""
    from utils.constant import para
    target_users = int(para['user_num'])
    rng = random.Random(seed)
    plan = [0] * max_steps
    schedule = [[] for _ in range(max_steps)]
    uids = list(range(target_users))
    rng.shuffle(uids)
    horizon = max(1, min(stop_step, max_steps))
    for i, uid in enumerate(uids):
        slot = int(i * horizon / target_users)
        schedule[slot].append(uid)
        plan[slot] += 1
    if burst_prob > 0:
        for slot in range(horizon):
            if rng.random() < burst_prob:
                extra = rng.randint(burst_min, burst_max)
                pulled = 0
                pull_slot = horizon - 1
                while pulled < extra and pull_slot > slot:
                    while pull_slot > 0 and (not schedule[pull_slot]) and (pulled < extra):
                        pull_slot -= 1
                    if pull_slot > slot and schedule[pull_slot]:
                        uid = schedule[pull_slot].pop()
                        plan[pull_slot] -= 1
                        schedule[slot].append(uid)
                        plan[slot] += 1
                        pulled += 1
                    pull_slot -= 1
    return {'plan': plan, 'schedule': schedule, 'env_seed': seed}

def make_offline_arrival_plan(seed: int, max_steps: int, target_users: int=None):
    from utils.constant import para
    if target_users is None:
        target_users = int(para['user_num'])
    plan = [0] * max_steps
    schedule = [[] for _ in range(max_steps)]
    plan[0] = target_users
    schedule[0] = list(range(target_users))
    return {'plan': plan, 'schedule': schedule, 'env_seed': seed}

def save_arrival_plan_to_run_dir(run_dir: str, seed_offset: int, arrival_plan: dict, env_seed: int=None, max_steps: int=None, stop_step: int=0, base_prob: float=0.0, burst_prob: float=0.0, burst_min: int=0, burst_max: int=0):
    p = Path(run_dir) / 'arrivals' / f'arrival_seed{seed_offset}.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {'env_seed': env_seed if env_seed is not None else arrival_plan.get('env_seed', 0), 'arrival_plan': arrival_plan, 'max_steps': max_steps, 'stop_step': stop_step, 'base_prob': base_prob, 'burst_prob': burst_prob, 'burst_min': burst_min, 'burst_max': burst_max}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(p)

def apply_arrival_plan(ts, slot, plan):
    if ts.decision_timer is not None:
        ts.decision_timer.finish()
    if plan is None:
        return
    if isinstance(plan, dict):
        if 'arrival_plan_dict' in plan and isinstance(plan['arrival_plan_dict'], dict):
            plan = plan['arrival_plan_dict']
        elif 'arrival_plan' in plan:
            plan = plan['arrival_plan']
    if isinstance(plan, dict):
        schedule = plan.get('schedule', None)
        if isinstance(schedule, list):
            if 0 <= slot < len(schedule) and schedule[slot]:
                ts.new_arrival(1, slot, fixed_uids=schedule[slot])
            return
        counts = plan.get('plan', None)
        if isinstance(counts, list):
            if 0 <= slot < len(counts):
                k = int(counts[slot])
                for _ in range(k):
                    ts.new_arrival(1, slot)
            return
        return
    if isinstance(plan, list):
        if 0 <= slot < len(plan):
            k = int(plan[slot])
            for _ in range(k):
                ts.new_arrival(1, slot)
        return

def get_arrived_apps(ts):
    return {uid for uid in range(ts.user_num) if np.isfinite(ts.enter_time[uid])}

def all_arrived_done(ts):
    arrived = get_arrived_apps(ts)
    done = set(getattr(ts, 'application_finished', set())) | set(getattr(ts, 'application_timeout_finished', set()))
    return len(arrived) > 0 and len(arrived - done) == 0

def safe_rest_tasks_total(rest_tasks):
    if isinstance(rest_tasks, (list, tuple, np.ndarray)):
        return int(np.sum(rest_tasks))
    elif isinstance(rest_tasks, dict):
        return int(sum(rest_tasks.values()))
    else:
        return 0

def _to_int_safe(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s == '':
            return 0
        try:
            return int(float(s))
        except Exception:
            return 0
    if isinstance(x, (list, tuple, np.ndarray)):
        return int(len(x))
    return 0

def load_arrival_plan(run_dir: str, seed_offset: int):
    p = Path(run_dir) / 'arrivals' / f'arrival_seed{seed_offset}.json'
    if p.exists():
        data = json.loads(p.read_text(encoding='utf-8'))
        if isinstance(data, list):
            return {'arrival_plan': data, 'env_seed': None}
        plan = data.get('arrival_plan', [])
        if isinstance(plan, dict):
            if 'plan' in plan:
                data['arrival_plan_dict'] = plan
                data['arrival_plan'] = [_to_int_safe(v) for v in plan['plan']]
            else:
                data['arrival_plan_dict'] = plan
                data['arrival_plan'] = []
        elif isinstance(plan, list):
            data['arrival_plan_dict'] = {'plan': plan}
            data['arrival_plan'] = [_to_int_safe(v) for v in plan]
        else:
            pass
            data['arrival_plan_dict'] = {'plan': []}
            data['arrival_plan'] = []
        return data
    else:
        return None

def make_run_dir(base_dir: str, algo_name: str=None, seed: int=None) -> Path:
    ts = datetime.now().strftime('%m%d%H%M%S')
    if algo_name and seed is not None:
        run_dir = Path(base_dir) / f'run{algo_name}_seed{seed}_{ts}'
    else:
        run_dir = Path(base_dir) / f'run_{ts}'
    (run_dir / 'curves').mkdir(parents=True, exist_ok=True)
    (run_dir / 'traces').mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (run_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (run_dir / 'arrivals').mkdir(parents=True, exist_ok=True)
    (run_dir / 'configs').mkdir(parents=True, exist_ok=True)
    return run_dir
CURVE_FIELDS = ['episode', 'phase', 'reward', 'energy', 'delay', 'app_timeout_rate', 'task_timeout_rate', 'score', 'utility_score', 'epsilon', 'avg_loss', 'note']

def safe_append_curve_row(path: Path, row: dict, retry: int=10, sleep: float=0.15):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    full = {k: row.get(k, '') for k in CURVE_FIELDS}
    header_needed = not path.exists() or path.stat().st_size == 0
    for i in range(retry):
        try:
            with open(path, 'a', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
                if header_needed:
                    w.writeheader()
                    header_needed = False
                w.writerow(full)
            return True
        except PermissionError:
            time.sleep(sleep * (i + 1))
        except Exception as e:
            return False
    return False

def append_curve_row(path: Path, row: dict, max_retries=3):
    for attempt in range(max_retries):
        try:
            header = not path.exists()
            df = pd.DataFrame([row])
            df.to_csv(path, mode='a', header=header, index=False)
            return
        except PermissionError as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
            else:
                pass
                pass
                pass
        except Exception as e:
            break

def atomic_to_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False, float_format='%.4f')
    try:
        if path.exists():
            path.unlink()
        tmp.replace(path)
    except PermissionError:
        pass
    except Exception as e:
        pass

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def init_worker(seed_offset, para, CONFIG):
    seed = CONFIG['SEED'] + seed_offset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if 'local_power' not in para or len(para.get('local_power', [])) < para['user_num']:
        env_seed = CONFIG['SEED']
        rng = random.Random(env_seed)
        if 'local_power_range' in para:
            low, high = para['local_power_range']
            para['local_power'] = [rng.uniform(low, high) * 10 ** 9 for _ in range(para['user_num'])]
        else:
            para['local_power'] = [rng.uniform(300000000.0, 600000000.0) for _ in range(para['user_num'])]

def safe_rest_tasks_total(rest_tasks) -> int:
    try:
        v = np.sum(rest_tasks)
        if isinstance(v, np.ndarray):
            v = v.item()
        return int(v)
    except Exception:
        try:
            return int(rest_tasks)
        except Exception:
            return 0

def safe_action_to_int(action) -> int:
    try:
        if isinstance(action, (list, tuple, np.ndarray)):
            if len(action) > 0:
                return int(action[0])
            else:
                return 0
        else:
            return int(action)
    except Exception:
        return 0

def diagnose_timeout(ts, algo_name=''):
    timeout_info = calc_timeout_rate(ts)
    sys.stdout.flush()

def torch_load_compat(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def strip_prefix(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def weight_fingerprint(model):
    try:
        for p in model.parameters():
            w = p.detach().float().cpu()
            return (float(w.mean()), float(w.std()), float(w.abs().max()))
    except:
        return (0.0, 0.0, 0.0)

def load_model_bundle(bundle_path, device):
    bundle_path = Path(bundle_path)
    if bundle_path.is_dir():
        pt_files = list(bundle_path.glob('*.pt'))
        if pt_files:
            bundle_path = pt_files[0]
        else:
            return None
    if bundle_path.suffix != '.pt':
        return None
    try:
        bundle = torch.load(bundle_path, map_location=device, weights_only=False)
        if isinstance(bundle, dict) and 'algo' in bundle and ('model_state' in bundle):
            return bundle
    except Exception as e:
        pass
    return None

def generate_tight_deadline_config(seed: int, user_num: int, tight_ratio: float=0.3, deadline_slot: int=130, factor_range=(0.6, 0.9)):
    from utils.constant import para
    deadline_slot = para.get('deadline_slot', deadline_slot)
    rng = random.Random(seed)
    num_tight_users = int(user_num * tight_ratio)
    tight_deadline_users = rng.sample(range(user_num), num_tight_users)
    deadline_slot_per_user = [deadline_slot] * user_num
    app_deadline_slots = {}
    for idx, uid in enumerate(tight_deadline_users):
        min_factor, max_factor = factor_range
        factor = min_factor + idx % 4 * 0.1
        tight_deadline = int(round(deadline_slot * factor))
        app_deadline_slots[uid] = tight_deadline
        deadline_slot_per_user[uid] = tight_deadline
    return {'tight_user_ids': tight_deadline_users, 'app_deadline_slots': app_deadline_slots, 'deadline_slot_per_user': deadline_slot_per_user, 'default_deadline_slot': deadline_slot}

def save_deadline_config(run_dir: str, seed: int, tight_ratio: float=0.3):
    from utils.constant import para
    deadline_config = generate_tight_deadline_config(seed=seed, user_num=para['user_num'], tight_ratio=tight_ratio, deadline_slot=para['deadline_slot'])
    config_dir = Path(run_dir) / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    deadline_path = config_dir / 'deadlines.json'
    with open(deadline_path, 'w', encoding='utf-8') as f:
        json.dump(deadline_config, f, indent=2)
    return deadline_config

def load_deadline_config(run_dir: str):
    from utils.constant import para
    deadline_path = Path(run_dir) / 'configs' / 'deadlines.json'
    if deadline_path.exists():
        with open(deadline_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        if 'deadline_slot_per_user' in cfg:
            slots = cfg['deadline_slot_per_user']
            if len(slots) != para['user_num']:
                raise ValueError(f"deadline_slot_per_user 长度不匹配: {len(slots)} != {para['user_num']}")
            slots = [int(x) for x in slots]
            cfg['deadline_slot_per_user'] = slots
            if 'app_deadline_slots' not in cfg or len(cfg['app_deadline_slots']) == 0:
                default_slot = cfg.get('default_deadline_slot', para['deadline_slot'])
                app_deadline_slots = {}
                for uid, slot in enumerate(slots):
                    if slot != default_slot:
                        app_deadline_slots[uid] = slot
                cfg['app_deadline_slots'] = app_deadline_slots
        return cfg
    return None

def estimate_costs_helper(env, ts, task_complex_index, uid, sid, slot, task_size_bytes):
    now = slot * para['slot_interval']
    bw = float(min(para['uplink_range']))
    f_local = float(env.device_list[uid].local_power)
    from Environment import computation
    _, t_exec_l = computation.execute_consumption(task_size_bytes, f_local, task_complex_index, 'l')
    t_queue_l = max(0.0, float(ts.devices_exe_useful[uid]) - now)
    cost_local = t_queue_l + float(t_exec_l)
    edge_costs = []
    for eid in range(para['edge_num']):
        f_edge = float(env.edges[eid].edge_power * env.edges[eid].calculate_parameter)
        dist = float(env.device_list[uid].edge_distances[eid])
        _, t_up_e = computation.upload_consumption([task_size_bytes, dist, bw], 1, 'e')
        _, t_exec_e = computation.execute_consumption(task_size_bytes, f_edge, task_complex_index, 'e')
        t_queue_e = max(0.0, float(ts.remain_times[eid]) - now)
        t_queue_up = max(0.0, float(ts.devices_upload_useful[uid]) - now)
        cost_edge = t_queue_up + float(t_up_e) + t_queue_e + float(t_exec_e)
        edge_costs.append(cost_edge)
    nearest = int(np.argmin(env.device_list[uid].edge_distances))
    dist0 = float(env.device_list[uid].edge_distances[nearest])
    _, t_up1 = computation.upload_consumption([task_size_bytes, dist0, bw], 1, 'e')
    _, t_up2 = computation.upload_consumption(task_size_bytes, 1, 'c')
    fc = float(env.cloud.cloud_power)
    _, t_exec_c = computation.execute_consumption(task_size_bytes, fc, task_complex_index, 'c')
    wan = float(para.get('cloud_wan_rtt', 0.0))
    t_queue_up = max(0.0, float(ts.devices_upload_useful[uid]) - now)
    cost_cloud = t_queue_up + float(t_up1) + float(t_up2) + float(t_exec_c) + wan
    return (cost_local, cost_cloud, edge_costs)

def augment_state_with_lookahead_and_current(state_data, node2idx, env, ts, task_complex_index, uid, sid, slot, task_size_bytes):
    LOOKAHEAD_DIM = 2 + para['edge_num']
    size_log = torch.log1p(torch.tensor(task_size_bytes, dtype=torch.float32)).view(1, 1)
    size_norm = (size_log - 10.0) / 10.0
    size_feat = size_norm.repeat(state_data.x.size(0), 1).to(state_data.x.device)
    cost_vec = [0.0] * LOOKAHEAD_DIM
    try:
        c_local, c_cloud, c_edges = estimate_costs_helper(env, ts, task_complex_index, uid, sid, slot, task_size_bytes)
        max_cost = 3.0
        cost_vec = [c_local, c_cloud] + c_edges
        cost_vec = [min(1.0, max(0.0, c / max_cost)) for c in cost_vec]
    except Exception as e:
        pass
    global_cost = torch.tensor(cost_vec, dtype=state_data.x.dtype).view(1, -1)
    global_cost = global_cost.repeat(state_data.x.size(0), 1)
    is_cur = torch.zeros((state_data.x.size(0), 1), dtype=state_data.x.dtype)
    if sid in node2idx:
        is_cur[node2idx[sid], 0] = 1.0
    state_data.x = torch.cat([state_data.x, size_feat, global_cost, is_cur], dim=1)
    return state_data
_current_mod = sys.modules.get(__name__)
