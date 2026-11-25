import random
import time
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
experiments_dir = os.path.join(project_root, 'experiments')
if experiments_dir not in sys.path:
    sys.path.insert(0, experiments_dir)
from Environment import computation
from utils.constant import para
from scheduler.task_selector import TaskSelector
try:
    from experiments.exp_utils import CONFIG
    _HAS_CONFIG = True
except:
    CONFIG = {'TOPK_TASKS': 30}
    _HAS_CONFIG = False

def _as_float(val, default=0.0):
    try:
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(val[0]) if len(val) > 0 else default
        return float(val)
    except:
        return default

class Benchmark:

    def __init__(self, env, gs, ts, complex_index, effective=False, seed=None, task_order_mode=None):
        self.env = env
        self.gs = gs
        self.ts = ts
        self.complex_index = complex_index
        self.effective = effective
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random
        self.ESTIMATION_ERROR = 0.2
        self.VISIBLE_K = 3
        self.task_order_mode = task_order_mode or 'none'
        self.task_selector = TaskSelector(mode=self.task_order_mode, seed=seed or 0)

    def reset(self):
        pass

    def _get_noisy_value(self, true_value):
        if true_value == 0:
            return 0
        noise = 1.0 + self.rng.uniform(-self.ESTIMATION_ERROR, self.ESTIMATION_ERROR)
        return true_value * noise

    def _get_safe_task_size(self, user_id, subtask_id):
        s = self.ts.task_size
        try:
            if hasattr(self.ts, 'get_task_size_bytes'):
                return self.ts.get_task_size_bytes(user_id, subtask_id)
            if isinstance(s, (list, tuple, np.ndarray)) and len(s) > user_id:
                user_data = s[user_id]
                if isinstance(user_data, (list, tuple, np.ndarray)) and len(user_data) > subtask_id:
                    return float(user_data[subtask_id])
            if isinstance(s, dict):
                if (user_id, subtask_id) in s:
                    return float(s[user_id, subtask_id])
                if subtask_id in s:
                    return float(s[subtask_id])
                if str(subtask_id) in s:
                    return float(s[str(subtask_id)])
                return float(list(s.values())[0])
            if isinstance(s, (list, tuple, np.ndarray)) and len(s) > subtask_id:
                return float(s[subtask_id])
            return 200000.0
        except:
            return 200000.0

    def get_actions(self, type, slot, preslot):
        """Propose nonlearning actions from an adaptively limited task list.

        Start with GraphScheduler readiness/CP-slack triage, apply the optional
        TaskSelector order (default preserves input order), then adaptive
        pick_topk_tasks. The helper overrides the supplied k under its default
        adaptive=True policy. Local-only, Edge-only, Greedy, Genetic, and
        HybridPSOGA all enter through this path. Placement proposals follow
        each strategy; step checks feasibility against current resources."""
        tasks = self.gs.get_tasks(slot, sort_tasks=False)
        if not tasks:
            return ([], preslot)
        tasks = self.task_selector.order(self.gs, slot, tasks)
        k = CONFIG.get('TOPK_TASKS', 30)
        if hasattr(self.gs, 'pick_topk_tasks'):
            try:
                tasks = self.gs.pick_topk_tasks(slot, tasks, k=min(k, len(tasks)))
            except Exception:
                pass
        if type == 0:
            return ([[task, 0] for task in tasks], preslot)
        elif type == 1:
            return ([[task, 1] for task in tasks], preslot)
        elif type == 2:
            edge_num = para['edge_num']
            return_list = []
            for task in tasks:
                edge_id = self.rng.randint(0, edge_num - 1)
                return_list.append([task, edge_id + 2])
            return (return_list, preslot)
        elif type == 4:
            return_list = []
            for task in tasks:
                user_id, subtask_id = task
                real_task_size = self._get_safe_task_size(user_id, subtask_id)
                est_task_size = self._get_noisy_value(real_task_size)
                f_local = _as_float(self.env.device_list[user_id].local_power, 1000000000.0)
                _, t_local = computation.execute_consumption(est_task_size, f_local, self.complex_index, 'l')
                best_action = 0
                min_time = t_local
                est_bw_cloud = self._get_noisy_value(np.mean(para['uplink_range']))
                nearest_edge_idx = int(np.argmin(self.env.device_list[user_id].edge_distances))
                dist = float(self.env.device_list[user_id].edge_distances[nearest_edge_idx])
                _, t_up1 = computation.upload_consumption([est_task_size, dist, est_bw_cloud], 1, 'e')
                _, t_up2 = computation.upload_consumption(est_task_size, 1, 'c')
                fc = _as_float(self.env.cloud.cloud_power, 6000000000.0)
                _, t_exe_c = computation.execute_consumption(est_task_size, fc, self.complex_index, 'c')
                t_cloud = t_up1 + t_up2 + t_exe_c
                if t_cloud < min_time:
                    min_time = t_cloud
                    best_action = 1
                dists = self.env.device_list[user_id].edge_distances
                visible_edges = list(np.argsort(dists))[:self.VISIBLE_K]
                for e_id in visible_edges:
                    est_bw_edge = self._get_noisy_value(np.mean(para['uplink_range']))
                    f_edge = self.env.edges[e_id].edge_power * self.env.edges[e_id].calculate_parameter
                    dist_e = float(dists[e_id])
                    _, t_up_e = computation.upload_consumption([est_task_size, dist_e, est_bw_edge], 1, 'e')
                    _, t_exe_e = computation.execute_consumption(est_task_size, f_edge, self.complex_index, 'e')
                    t_edge = t_up_e + t_exe_e
                    if t_edge < min_time:
                        min_time = t_edge
                        best_action = e_id + 2
                return_list.append([task, best_action])
            return (return_list, preslot)
        elif type == 6:
            now = slot * float(para.get('slot_interval', 0.01))

            def _avg_wait(times):
                if not times:
                    return 0.0
                ws = []
                for t in times:
                    try:
                        ws.append(max(0.0, float(t) - now))
                    except Exception:
                        ws.append(0.0)
                return float(sum(ws) / max(1, len(ws)))
            avg_local_wait = _avg_wait(getattr(self.ts, 'devices_exe_useful', []))
            avg_upload_wait = _avg_wait(getattr(self.ts, 'devices_upload_useful', []))
            avg_edge_wait = 0.0
            edge_useful = getattr(self.ts, 'edge_useful', None)
            if isinstance(edge_useful, (list, tuple)) and len(edge_useful) > 0:
                per_edge_best = []
                for edge_cores in edge_useful:
                    try:
                        core_times = [max(0.0, float(t) - now) for t in edge_cores]
                        per_edge_best.append(min(core_times) if core_times else 0.0)
                    except Exception:
                        per_edge_best.append(0.0)
                if per_edge_best:
                    avg_edge_wait = float(sum(per_edge_best) / len(per_edge_best))
            edge_num = int(para['edge_num'])
            bw_choices = para.get('uplink_range', [para['uplink_range'][0]])
            rng = self.rng
            action_max = 1 + edge_num
            population_size = 4
            generations = 1
            mutation_rate = 0.2
            crossover_rate = 0.8

            def _app_deadline_abs(uid: int) -> float:
                slot_interval = float(para.get('slot_interval', 0.01))
                default_slots = int(para.get('deadline_slot', 180))
                enter = getattr(self.ts, 'enter_time', None)
                enter_t = float(enter[uid]) if enter is not None and uid < len(enter) and (enter[uid] != float('inf')) else now
                if hasattr(self.ts, 'get_app_deadline_slot'):
                    try:
                        dslots = int(self.ts.get_app_deadline_slot(uid))
                    except Exception:
                        dslots = default_slots
                else:
                    dslots = default_slots
                return enter_t + dslots * slot_interval

            def _estimate_energy_delay(uid: int, task_size_bytes: float, action: int):
                if action == 0:
                    f_local = _as_float(self.env.device_list[uid].local_power, 1000000000.0)
                    e_ex, t_ex = computation.execute_consumption(task_size_bytes, f_local, self.complex_index, 'l')
                    return (float(e_ex), float(avg_local_wait + t_ex))
                if action == 1:
                    fc = _as_float(self.env.cloud.cloud_power, 6000000000.0)
                    nearest_edge = int(np.argmin(self.env.device_list[uid].edge_distances))
                    dist = float(self.env.device_list[uid].edge_distances[nearest_edge])
                    bw = float(rng.choice(bw_choices))
                    e1, d1 = computation.upload_consumption([task_size_bytes, dist, bw], 1, 'e')
                    e2, d2 = computation.upload_consumption(task_size_bytes, 1, 'c')
                    e3, d3 = computation.execute_consumption(task_size_bytes, fc, self.complex_index, 'c')
                    return (float(e1 + e2 + e3), float(avg_upload_wait + d1 + d2 + d3))
                edge_id = int(action - 2)
                edge_id = max(0, min(edge_id, edge_num - 1))
                f_edge = self.env.edges[edge_id].edge_power * self.env.edges[edge_id].calculate_parameter
                dist_e = float(self.env.device_list[uid].edge_distances[edge_id])
                bw = float(rng.choice(bw_choices))
                e_up, d_up = computation.upload_consumption([task_size_bytes, dist_e, bw], 1, 'e')
                e_ex, d_ex = computation.execute_consumption(task_size_bytes, f_edge, self.complex_index, 'e')
                return (float(e_up + e_ex), float(avg_upload_wait + avg_edge_wait + d_up + d_ex))

            def _late_penalty(uid: int, est_delay: float) -> float:
                deadline_abs = _app_deadline_abs(uid)
                remaining = float(deadline_abs - now)
                if remaining <= 0.0:
                    return 20.0
                overflow = max(0.0, est_delay - remaining)
                return 10.0 * overflow

            def _heuristic_seed_action(uid: int, subtask_id: int) -> int:
                task_size = float(self._get_safe_task_size(uid, subtask_id))
                local_e, local_d = _estimate_energy_delay(uid, task_size, 0)
                cloud_e, cloud_d = _estimate_energy_delay(uid, task_size, 1)
                nearest_eid = int(np.argmin(self.env.device_list[uid].edge_distances))
                edge_action = 2 + nearest_eid
                edge_e, edge_d = _estimate_energy_delay(uid, task_size, edge_action)
                deadline_abs = _app_deadline_abs(uid)
                remaining = float(deadline_abs - now)
                if remaining > 0 and local_d < remaining * 0.5:
                    return 0
                if cloud_d <= edge_d:
                    return 1
                else:
                    return edge_action
            tasks_by_app = {}
            for task in tasks:
                user_id, subtask_id = task
                app_id = user_id
                if app_id not in tasks_by_app:
                    tasks_by_app[app_id] = []
                tasks_by_app[app_id].append(task)
            task_best_actions = {}
            for uid, app_tasks in tasks_by_app.items():
                if not app_tasks:
                    continue
                n = len(app_tasks)

                def fitness(individual):
                    total_energy = 0.0
                    total_delay = 0.0
                    for i, task in enumerate(app_tasks):
                        user_id, subtask_id = task
                        action = individual[i]
                        real_task_size = float(self._get_safe_task_size(user_id, subtask_id))
                        noisy_task_size = self._get_noisy_value(real_task_size)
                        e, d = _estimate_energy_delay(uid, noisy_task_size, action)
                        total_energy += e
                        total_delay += d
                    penalty = _late_penalty(uid, total_delay)
                    return 0.5 * total_energy + 0.5 * total_delay + penalty

                def selection(pop, fits):
                    selected = []
                    for _ in range(len(pop)):
                        i1, i2 = rng.sample(range(len(pop)), 2)
                        selected.append(pop[i1][:] if fits[i1] < fits[i2] else pop[i2][:])
                    return selected

                def crossover(p1, p2):
                    if rng.random() > crossover_rate or len(p1) < 2:
                        return (p1[:], p2[:])
                    pt = rng.randint(1, len(p1) - 1)
                    return (p1[:pt] + p2[pt:], p2[:pt] + p1[pt:])

                def mutation(ind):
                    if rng.random() < mutation_rate:
                        idx = rng.randint(0, len(ind) - 1)
                        ind[idx] = rng.randint(0, action_max)
                    return ind
                population = []
                seed = [_heuristic_seed_action(t[0], t[1]) for t in app_tasks]
                population.append(seed)
                while len(population) < population_size:
                    ind = [rng.randint(0, action_max) for _ in range(n)]
                    population.append(ind)
                for _gen in range(generations):
                    fits = [fitness(ind) for ind in population]
                    parents = selection(population, fits)
                    next_pop = []
                    while len(next_pop) < population_size:
                        p1, p2 = rng.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
                        c1, c2 = crossover(p1, p2)
                        next_pop.append(mutation(c1))
                        if len(next_pop) < population_size:
                            next_pop.append(mutation(c2))
                    population = next_pop
                final_fits = [fitness(ind) for ind in population]
                best_idx = int(np.argmin(final_fits))
                best_ind = population[best_idx]
                for i, task in enumerate(app_tasks):
                    task_best_actions[task] = int(best_ind[i])
            return_list = []
            for task in tasks:
                if task in task_best_actions:
                    return_list.append([task, task_best_actions[task]])
                else:
                    return_list.append([task, 0])
            return (return_list, preslot)
        elif type == 8:
            now = slot * float(para.get('slot_interval', 0.01))
            tasks_by_app = {}
            for task in tasks:
                uid, sid = task
                if uid not in tasks_by_app:
                    tasks_by_app[uid] = []
                tasks_by_app[uid].append(task)
            task_best_actions = {}
            pop_size = 5
            max_iter = 2
            w = 0.7
            c1, c2 = (1.2, 1.2)
            mutation_rate = 0.1
            edge_num = int(para['edge_num'])
            action_max = 1 + edge_num
            bw_avg = np.mean(para['uplink_range'])
            bw_choices = para.get('uplink_range', [bw_avg])

            def _avg_wait(times):
                if not times:
                    return 0.0
                ws = []
                for t in times:
                    try:
                        ws.append(max(0.0, float(t) - now))
                    except Exception:
                        ws.append(0.0)
                return float(sum(ws) / max(1, len(ws)))
            avg_local_wait = _avg_wait(getattr(self.ts, 'devices_exe_useful', []))
            avg_upload_wait = _avg_wait(getattr(self.ts, 'devices_upload_useful', []))
            per_edge_wait = []
            avg_edge_wait = 0.0
            edge_useful = getattr(self.ts, 'edge_useful', None)
            if isinstance(edge_useful, (list, tuple)) and len(edge_useful) > 0:
                for edge_cores in edge_useful:
                    try:
                        core_times = [max(0.0, float(t) - now) for t in edge_cores]
                        per_edge_wait.append(min(core_times) if core_times else 0.0)
                    except Exception:
                        per_edge_wait.append(0.0)
                if per_edge_wait:
                    avg_edge_wait = float(sum(per_edge_wait) / len(per_edge_wait))
            for uid, app_tasks in tasks_by_app.items():
                if not app_tasks:
                    continue
                n = len(app_tasks)
                deadline_abs = self.ts.enter_time[uid] + self.ts.get_app_deadline_slot(uid) * para['slot_interval']

                def _est_cost(action, noisy_size, task_idx):
                    q_local = avg_local_wait
                    q_up = avg_upload_wait
                    if action == 0:
                        f_local = _as_float(self.env.device_list[uid].local_power, 1000000000.0)
                        e, d = computation.execute_consumption(noisy_size, f_local, self.complex_index, 'l')
                        return (e, q_local + d)
                    elif action == 1:
                        fc = _as_float(self.env.cloud.cloud_power, 6000000000.0)
                        dist = float(self.env.device_list[uid].edge_distances[int(np.argmin(self.env.device_list[uid].edge_distances))])
                        bw = self._get_noisy_value(bw_avg)
                        e1, d1 = computation.upload_consumption([noisy_size, dist, bw], 1, 'e')
                        e2, d2 = computation.upload_consumption(noisy_size, 1, 'c')
                        e3, d3 = computation.execute_consumption(noisy_size, fc, self.complex_index, 'c')
                        return (e1 + e2 + e3, q_up + d1 + d2 + d3)
                    else:
                        eid = int(action - 2)
                        f_e = self.env.edges[eid].edge_power * self.env.edges[eid].calculate_parameter
                        dist = float(self.env.device_list[uid].edge_distances[eid])
                        if 0 <= eid < len(per_edge_wait):
                            q_edge = per_edge_wait[eid]
                        else:
                            q_edge = avg_edge_wait
                        bw = self._get_noisy_value(bw_avg)
                        e_up, d_up = computation.upload_consumption([noisy_size, dist, bw], 1, 'e')
                        e_ex, d_ex = computation.execute_consumption(noisy_size, f_e, self.complex_index, 'e')
                        return (e_up + e_ex, q_up + d_up + q_edge + d_ex)

                def fitness(particle_pos):
                    total_e, max_d, total_d = (0, 0, 0)
                    edge_loads = {}
                    for i, act_raw in enumerate(particle_pos):
                        act = max(0, min(action_max, int(round(act_raw))))
                        if act >= 2:
                            eid = act - 2
                            dists = self.env.device_list[uid].edge_distances
                            visible_edges = list(np.argsort(dists))[:self.VISIBLE_K]
                            if eid not in visible_edges:
                                act = visible_edges[0] + 2
                                eid = act - 2
                        else:
                            eid = -1
                        real_s = self._get_safe_task_size(app_tasks[i][0], app_tasks[i][1])
                        noisy_s = self._get_noisy_value(real_s)
                        e, d = _est_cost(act, noisy_s, i)
                        total_e += e
                        total_d += d
                        max_d = max(max_d, d)
                        if eid >= 0:
                            edge_loads[eid] = edge_loads.get(eid, 0) + 1
                    if len(edge_loads) > 1:
                        loads = list(edge_loads.values())
                        mean_load = sum(loads) / len(edge_loads)
                        load_var = sum(((x - mean_load) ** 2 for x in loads)) / len(edge_loads)
                    else:
                        load_var = 0.0
                    overflow = max(0.0, now + max_d - deadline_abs)
                    penalty = 2.5 * overflow + (2.5 if overflow > 0 else 0.0)
                    return 0.5 * total_e + 0.5 * max_d + penalty
                particles = []
                dists = self.env.device_list[uid].edge_distances
                visible_edges = list(np.argsort(dists))[:self.VISIBLE_K]
                visible_actions = [0, 1] + [e + 2 for e in visible_edges]
                if n >= 2 and len(visible_edges) >= 2:
                    sorted_eids = sorted(visible_edges, key=lambda x: per_edge_wait[x] if 0 <= x < len(per_edge_wait) else 0.0)
                    seed_dispersion = [float(2 + sorted_eids[i % len(sorted_eids)]) for i in range(n)]
                else:
                    seed_dispersion = [float(self.rng.choice(visible_actions)) for _ in range(n)]
                if visible_edges:
                    nearest_eid = visible_edges[0]
                    seed_nearest = [float(2 + nearest_eid) for _ in range(n)]
                else:
                    seed_nearest = [float(self.rng.choice(visible_actions)) for _ in range(n)]
                seed_local = [0.0 for _ in range(n)]
                for _p_idx in range(pop_size):
                    if _p_idx == 0:
                        pos = seed_dispersion
                    elif _p_idx == 1:
                        pos = seed_nearest
                    elif _p_idx == 2:
                        pos = seed_local
                    else:
                        pos = [float(self.rng.choice(visible_actions)) for _ in range(n)]
                    particles.append({'pos': [float(act) for act in pos], 'vel': [self.rng.uniform(-1, 1) for _ in range(n)], 'best_pos': pos[:], 'best_fit': float('inf')})
                g_best_pos = particles[0]['pos'][:]
                g_best_fit = float('inf')
                for _iter in range(max_iter):
                    for p in particles:
                        fit = fitness(p['pos'])
                        if fit < p['best_fit']:
                            p['best_fit'] = fit
                            p['best_pos'] = p['pos'][:]
                        if fit < g_best_fit:
                            g_best_fit = fit
                            g_best_pos = p['pos'][:]
                    for p in particles:
                        for i in range(n):
                            r1, r2 = (self.rng.random(), self.rng.random())
                            p['vel'][i] = w * p['vel'][i] + c1 * r1 * (p['best_pos'][i] - p['pos'][i]) + c2 * r2 * (g_best_pos[i] - p['pos'][i])
                            new_pos_raw = p['pos'][i] + p['vel'][i]
                            new_pos = max(0, min(action_max, new_pos_raw))
                            p['pos'][i] = new_pos
                    if self.rng.random() < mutation_rate:
                        idx = self.rng.randint(0, pop_size - 1)
                        gene_idx = self.rng.randint(0, n - 1)
                        dists = self.env.device_list[uid].edge_distances
                        visible_edges = list(np.argsort(dists))[:self.VISIBLE_K]
                        visible_actions = [0, 1] + [e + 2 for e in visible_edges]
                        particles[idx]['pos'][gene_idx] = float(self.rng.choice(visible_actions))
                for i, task in enumerate(app_tasks):
                    act_raw = int(round(g_best_pos[i]))
                    if act_raw >= 2:
                        eid = act_raw - 2
                        dists = self.env.device_list[uid].edge_distances
                        visible_edges = list(np.argsort(dists))[:self.VISIBLE_K]
                        if eid not in visible_edges:
                            act_raw = visible_edges[0] + 2
                    task_best_actions[task] = act_raw
            return ([[t, task_best_actions.get(t, 0)] for t in tasks], preslot)
        else:
            return ([], preslot)

    def step(self, actions):
        """Execute proposals that pass a fresh simulator feasibility mask.

        Check each action immediately before committing it, so earlier actions
        in the batch affect later checks. Out-of-range or masked actions are
        skipped with zero execution energy/delay and no resource allocation.
        Rejection does not trigger a replacement placement. Readiness and
        application filtering are the caller's task-selection responsibility."""
        total_energy = 0.0
        total_delay = 0.0
        for task, action in actions:
            user_id, subtask_id = task
            mask = self.ts.get_action_mask(user_id, self.ts.get_task_size_bytes(user_id, subtask_id), self.ts.current_time)
            if action < 0 or action >= len(mask) or (not mask[action]):
                continue
            transition_start = time.perf_counter()
            if hasattr(self.ts, 'offload_task'):
                try:
                    energy, delay = self.ts.offload_task(user_id, subtask_id, action)
                except TypeError:
                    energy, delay = self.ts.offload_task(subtask_id, user_id, action)
            elif hasattr(self.ts, 'add_task'):
                energy, delay = self.ts.add_task(user_id, subtask_id, action)
            else:
                raise AttributeError('TaskScheduler has neither offload_task nor add_task')
            if self.ts.decision_timer is not None:
                self.ts.decision_timer.excluded += time.perf_counter() - transition_start
                self.ts.decision_timer.launches += 1
            total_energy += float(energy)
            total_delay += float(delay)
        reward = -(0.5 * total_energy + 0.5 * total_delay)
        info = {'step_energy': total_energy, 'step_delay': total_delay}
        return (reward, info)
