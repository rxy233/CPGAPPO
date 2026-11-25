import networkx as nx
from Environment import computation
from Environment.Graph import BaseGraph
from Environment.environment import Environment
from utils.constant import para
from torch_geometric.data import Data
import numpy as np
import torch

class GraphScheduler:

    def __init__(self, basegraph: BaseGraph, subgraph_list, task_scheduler, device=None):
        self.basegraph = basegraph
        self.subgraph_list = subgraph_list
        self.ts = task_scheduler
        self.bfs_order = self.get_bfs_order()
        self.rank_u = {}
        self.num_edges = para['edge_num']
        self.num_users = para['user_num']
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.static_edge_index = self._build_static_edge_index()

    def get_bfs_order(self):
        bfs_tree = nx.DiGraph()
        for source in self.basegraph.enter_nodes:
            tree = nx.bfs_tree(self.basegraph.nx_graph, source=source)
            bfs_tree.add_edges_from(tree.edges())
        return list(bfs_tree.nodes())

    def _estimate_w(self, env, uid, node_id, task_complex_index):
        bytes_ = float(self.ts.task_size[node_id])
        f_local = float(env.device_list[uid].local_power)
        _, t_local = computation.execute_consumption(bytes_, f_local, task_complex_index, 'l')
        best_edge_t = 1000000000.0
        for eid in range(para['edge_num']):
            f_edge = env.edges[eid].edge_power * env.edges[eid].calculate_parameter
            dist = float(env.device_list[uid].edge_distances[eid])
            if not np.isfinite(dist) or dist > para['edge_radius']:
                continue
            bw = env.device_list[uid].local_bw
            _, t_up = computation.upload_consumption([bytes_, dist, bw], 1, 'e', local_trans=env.device_list[uid].local_trans)
            _, t_ex = computation.execute_consumption(bytes_, f_edge, task_complex_index, 'e')
            best_edge_t = min(best_edge_t, t_up + t_ex)
        fc = float(env.cloud.cloud_power)
        nearest = int(np.argmin(env.device_list[uid].edge_distances))
        dist = float(env.device_list[uid].edge_distances[nearest])
        bw = env.device_list[uid].local_bw
        _, t1 = computation.upload_consumption([bytes_, dist, bw], 1, 'e', local_trans=env.device_list[uid].local_trans)
        _, t2 = computation.upload_consumption(bytes_, 1, 'c')
        _, t3 = computation.execute_consumption(bytes_, fc, task_complex_index, 'c')
        wan = float(para.get('cloud_wan_rtt', 0.0))
        t_cloud = t1 + t2 + t3 + wan
        return min(t_local, best_edge_t, t_cloud)

    def _compute_rank_u_for_app(self, env, uid, task_complex_index):
        g = self.ts.subgraph_list[uid].nx_graph
        topo = list(nx.topological_sort(g))
        rank = {}
        for v in reversed(topo):
            wv = self._estimate_w(env, uid, v, task_complex_index)
            succs = list(g.successors(v))
            if not succs:
                rank[v] = wv
            else:
                rank[v] = wv + max((rank[s] for s in succs))
        self.rank_u[uid] = rank

    def _build_static_edge_index(self):
        adj = nx.to_numpy_array(self.basegraph.nx_graph)
        edge_coords = np.array(adj.nonzero())
        valid_mask = edge_coords[0] != edge_coords[1]
        edge_coords = edge_coords[:, valid_mask]
        num_graph_nodes = adj.shape[0]
        if edge_coords.size > 0:
            valid_mask = (edge_coords[0] < num_graph_nodes) & (edge_coords[1] < num_graph_nodes)
            edge_coords = edge_coords[:, valid_mask]
        if edge_coords.size == 0:
            num_self_loops = min(num_graph_nodes, 10)
            edge_coords = np.stack([np.arange(num_self_loops), np.arange(num_self_loops)], axis=0)
        return torch.tensor(edge_coords, dtype=torch.long, device=self.device)

    def get_graph_state_new(self, env: Environment, task, complex_index: int, slot=None):
        if isinstance(task, (list, tuple)) and len(task) >= 2:
            current_user_id = int(task[0])
            current_subtask_id = int(task[1])
        else:
            current_user_id, current_subtask_id = (0, 0)
        if isinstance(complex_index, (list, tuple, np.ndarray)):
            complex_index = int(getattr(env, 'task_complex_index', 0))
        else:
            complex_index = int(complex_index)
        uid = current_user_id
        if uid not in self.rank_u:
            self._compute_rank_u_for_app(env, uid, complex_index)
        rank_u_u = self.rank_u.get(uid, {})
        max_rank_u = max(rank_u_u.values()) if rank_u_u else 1.0
        max_rank_u = max(max_rank_u, 1e-09)
        deadline_u = float(self.ts.get_app_deadline_slot(uid)) * float(para['slot_interval'])
        deadline_u = max(deadline_u, 1e-09)
        if slot is not None:
            now = para['slot_interval'] * slot
        elif hasattr(env, 'slot'):
            now = para['slot_interval'] * env.slot
        else:
            now = 0.0
            pass
        u_upload = self.ts.devices_upload_useful
        u_exe = self.ts.devices_exe_useful
        curr_upload_wait = max(0.0, float(u_upload[uid]) - now)
        curr_exe_wait = max(0.0, float(u_exe[uid]) - now)
        edge_waits = [min(2.0, max(0.0, float(t) - now)) / 2.0 for t in self.ts.remain_times]
        global_features = [min(1.0, curr_upload_wait), min(1.0, curr_exe_wait)] + edge_waits
        edge_distances = env.device_list[uid].edge_distances
        norm_dists = [min(1.0, float(d) / 500.0) for d in edge_distances]
        global_features = global_features + norm_dists
        feature_len = 5 + len(global_features)
        g_u = self.ts.subgraph_list[uid].nx_graph
        nodes_u = set(g_u.nodes())
        raw_data = []
        for node_id, size_norm in enumerate(self.ts.task_size_normalized):
            in_app = 1.0 if node_id in nodes_u else 0.0
            is_current = 1.0 if node_id == current_subtask_id and in_app > 0.5 else 0.0
            ru = float(rank_u_u.get(node_id, 0.0))
            rank_u_norm = ru / max_rank_u * in_app
            urgency_norm = float(np.clip(ru / deadline_u, 0.0, 1.0)) * in_app
            node_feats = [in_app, float(size_norm), is_current, rank_u_norm, urgency_norm] + global_features
            if len(node_feats) < feature_len:
                node_feats += [0.0] * (feature_len - len(node_feats))
            elif len(node_feats) > feature_len:
                node_feats = node_feats[:feature_len]
            raw_data.append(node_feats)
        x = torch.tensor(raw_data, dtype=torch.float32, device=self.device)
        num_nodes = x.shape[0]
        edge_index = self.static_edge_index
        if edge_index.numel() > 0:
            max_idx = int(edge_index.max().item())
            if max_idx >= num_nodes:
                valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                edge_index = edge_index[:, valid_mask]
                if edge_index.numel() == 0:
                    device = x.device
                    num_self_loops = min(num_nodes, 10)
                    edge_index = torch.stack([torch.arange(num_self_loops, device=device), torch.arange(num_self_loops, device=device)], dim=0)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        return Data(x=x, edge_index=edge_index, batch=batch)

    def _get_user_by_subtask(self, subtask_id, user_to_subtasks=None):
        if user_to_subtasks is None:
            user_to_subtasks = {}
            for user_id in range(len(self.subgraph_list)):
                user_to_subtasks[user_id] = set(self.subgraph_list[user_id].nx_graph.nodes())
        for user_id, subtasks in user_to_subtasks.items():
            if subtask_id in subtasks:
                return user_id
        return -1

    def get_tasks(self, slot, sort_tasks: bool=False, enable_triage: bool=True, triage_margin: float=0.2):
        """Return ready tasks after application and earliest-start filtering.

        Active applications exclude both completed and timeout-finished sets.
        Default CP-slack triage defers applications below -triage_margin;
        when the list is empty it retries the deferred application with greatest
        slack. sort_tasks=True orders by descending upward rank, then IDs.
        This method applies no Top-K limit. Callers may subsequently use
        pick_topk_tasks, which has a different ordering and count policy."""
        if self.ts.decision_timer is not None:
            self.ts.decision_timer.start(slot)
        tasks = []
        active_users = self.ts.application_started - self.ts.application_finished - getattr(self.ts, 'application_timeout_finished', set())
        EPS = 1e-09
        now = para['slot_interval'] * slot
        user_slack_cp = {}
        if enable_triage:
            for user in active_users:
                enter_time = self.ts.enter_time[user]
                if enter_time == float('inf'):
                    user_slack_cp[user] = float('inf')
                    continue
                user_slack_cp[user] = self.get_slack_cp(user, slot, now=now)
        tasks_all = []
        for user in active_users:
            enter_time = self.ts.enter_time[user]
            if enter_time == float('inf'):
                continue
            if enable_triage and user_slack_cp.get(user, float('inf')) < -triage_margin:
                continue
            app_deadline_abs = enter_time + self.ts.get_app_deadline_slot(user) * para['slot_interval']
            for subtask, start_t in self.ts.start_time[user].items():
                if start_t == float('inf'):
                    continue
                if self.ts.finish_time[user][subtask] != float('inf'):
                    continue
                if start_t > now + EPS:
                    continue
                start_if_local = max(start_t, self.ts.devices_exe_useful[user])
                start_if_offld = max(start_t, self.ts.devices_upload_useful[user])
                best_possible_start = min(start_if_local, start_if_offld)
                if best_possible_start > app_deadline_abs + EPS:
                    continue
                tasks.append((user, subtask))
        if enable_triage and (not tasks):
            triage_users = [(u, s) for u, s in user_slack_cp.items() if s < -triage_margin]
            if triage_users:
                triage_users.sort(key=lambda x: x[1], reverse=True)
                best_user = triage_users[0][0]
                enter_time = self.ts.enter_time[best_user]
                if enter_time != float('inf'):
                    app_deadline_abs = enter_time + self.ts.get_app_deadline_slot(best_user) * para['slot_interval']
                    for subtask, start_t in self.ts.start_time[best_user].items():
                        if start_t == float('inf'):
                            continue
                        if self.ts.finish_time[best_user][subtask] != float('inf'):
                            continue
                        if start_t > now + EPS:
                            continue
                        start_if_local = max(start_t, self.ts.devices_exe_useful[best_user])
                        start_if_offld = max(start_t, self.ts.devices_upload_useful[best_user])
                        best_possible_start = min(start_if_local, start_if_offld)
                        if best_possible_start > app_deadline_abs + EPS:
                            continue
                        tasks.append((best_user, subtask))
        if tasks and sort_tasks:
            task_complex_index = self.ts.env.task_complex_index
            for u, _ in tasks:
                if u not in self.rank_u:
                    self._compute_rank_u_for_app(self.ts.env, u, task_complex_index)

            def get_slack_cp(user_id):
                enter_time = self.ts.enter_time[user_id]
                if enter_time == float('inf'):
                    return float('inf')
                app_deadline_abs = enter_time + self.ts.get_app_deadline_slot(user_id) * para['slot_interval']
                user_tasks = [t for t in tasks if t[0] == user_id]
                if not user_tasks:
                    return float('inf')
                cp_remain = max((self.rank_u[user_id].get(t[1], 0.0) for t in user_tasks))
                slack_cp = app_deadline_abs - now - cp_remain
                return float(slack_cp)
            tasks.sort(key=lambda pair: (-self.rank_u[pair[0]].get(pair[1], 0.0), pair[0], pair[1]))
        return tasks

    def get_task_priority_info(self, task, slot, all_tasks=None):
        uid, sid = task
        now = para['slot_interval'] * slot
        if uid not in self.rank_u:
            task_complex_index = self.ts.env.task_complex_index
            self._compute_rank_u_for_app(self.ts.env, uid, task_complex_index)
        rank_u = self.rank_u.get(uid, {}).get(sid, 0.0)
        enter_time = self.ts.enter_time[uid]
        if enter_time == float('inf'):
            slack = float('inf')
        else:
            app_deadline_abs = enter_time + self.ts.get_app_deadline_slot(uid) * para['slot_interval']
            if all_tasks is not None:
                user_tasks = [t for t in all_tasks if t[0] == uid]
                if user_tasks:
                    cp_remain = max((self.rank_u[uid].get(t[1], 0.0) for t in user_tasks))
                else:
                    cp_remain = rank_u
            else:
                cp_remain = rank_u
            deadline_remain = app_deadline_abs - now
            slack = deadline_remain - cp_remain
        s = self.ts.task_size
        if isinstance(s, (list, tuple, np.ndarray)) and len(s) > 0 and isinstance(s[0], (list, tuple, np.ndarray)):
            task_size = float(s[uid][sid])
        elif isinstance(s, dict):
            task_size = float(s.get(sid, 0))
        else:
            task_size = float(s[sid])
        return {'rank_u': rank_u, 'slack': slack, 'task_size': task_size, 'enter_time': enter_time}

    def get_tasks_priority_info_batch(self, tasks, slot):
        now = para['slot_interval'] * slot
        task_complex_index = self.ts.env.task_complex_index
        for uid, _ in tasks:
            if uid not in self.rank_u:
                self._compute_rank_u_for_app(self.ts.env, uid, task_complex_index)
        tasks_by_app = {}
        for task in tasks:
            uid = task[0]
            if uid not in tasks_by_app:
                tasks_by_app[uid] = []
            tasks_by_app[uid].append(task)
        cp_remain_by_app = {}
        for uid, app_tasks in tasks_by_app.items():
            cp_remain_by_app[uid] = max((self.rank_u[uid].get(t[1], 0.0) for t in app_tasks))
        s = self.ts.task_size
        result = {}
        for task in tasks:
            uid, sid = task
            rank_u = self.rank_u[uid].get(sid, 0.0)
            cp_remain = cp_remain_by_app[uid]
            enter_time = self.ts.enter_time[uid]
            if enter_time == float('inf'):
                slack = float('inf')
            else:
                app_deadline_abs = enter_time + self.ts.get_app_deadline_slot(uid) * para['slot_interval']
                deadline_remain = app_deadline_abs - now
                slack = deadline_remain - cp_remain
            if isinstance(s, (list, tuple, np.ndarray)) and len(s) > 0 and isinstance(s[0], (list, tuple, np.ndarray)):
                task_size = float(s[uid][sid])
            elif isinstance(s, dict):
                task_size = float(s.get(sid, 0))
            else:
                task_size = float(s[sid])
            result[task] = {'rank_u': rank_u, 'slack': slack, 'task_size': task_size, 'enter_time': enter_time}
        return result

    def pick_topk_tasks(self, slot, ready_tasks, k=30, adaptive=True):
        """Select a ranked candidate subset from the supplied task list.

        With adaptive=True, the supplied k is overridden: n <= 30 uses
        min(TOPK_MIN, n), 30 < n <= 50 uses 30, and n > 50 uses
        min(TOPK_MAX, n). Current CONFIG limits are 15 and 30. SMART_TOPK
        and ADAPTIVE_TOPK in CONFIG are compatibility fields; this function
        uses its adaptive argument. adaptive=False uses min(k, n).

        Sort ascending by (application CP slack - 0.5 * task_bytes / 1e6,
        negative upward rank, recorded task start time), preserving input
        order on exact ties. Application CP slack uses the maximum upward
        rank among that application's tasks in the supplied list.

        Benchmark.get_actions uses this for the nonlearning methods.
        GATDQN uses it after removing tasks with no feasible placement.
        The limit is per candidate selection, not a per-slot dispatch cap."""
        if not ready_tasks:
            return []
        num_ready = len(ready_tasks)
        if adaptive:
            topk_min = 20
            topk_max = 50
            try:
                from experiments.exp_utils import CONFIG
                topk_min = CONFIG.get('TOPK_MIN', 20)
                topk_max = CONFIG.get('TOPK_MAX', 50)
            except:
                pass
            if num_ready <= 30:
                k = min(topk_min, num_ready)
            elif num_ready <= 50:
                k = min(30, num_ready)
            else:
                k = min(topk_max, num_ready)
        else:
            k = min(k, num_ready)
        now = para['slot_interval'] * slot
        task_complex_index = self.ts.env.task_complex_index
        for u, _ in ready_tasks:
            if u not in self.rank_u:
                self._compute_rank_u_for_app(self.ts.env, u, task_complex_index)
        tasks_by_user = {}
        for u, s in ready_tasks:
            tasks_by_user.setdefault(u, []).append(s)
        slack_cp_by_user = {}
        task_size_map = {}
        for u, subtasks in tasks_by_user.items():
            enter_time = self.ts.enter_time[u]
            if enter_time == float('inf'):
                slack_cp_by_user[u] = float('inf')
                continue
            app_deadline_abs = enter_time + self.ts.get_app_deadline_slot(u) * para['slot_interval']
            cp_remain = max((self.rank_u[u].get(s, 0.0) for s in subtasks)) if subtasks else 0.0
            slack_cp_by_user[u] = float(app_deadline_abs - now - cp_remain)
            for s in subtasks:
                try:
                    t_size = float(self.ts.task_size[u][s]) if isinstance(self.ts.task_size, (list, np.ndarray)) else float(self.ts.task_size.get((u, s), 200000))
                    task_size_map[u, s] = t_size
                except:
                    task_size_map[u, s] = 200000.0

        def sort_key(t):
            u, s = t
            slack = slack_cp_by_user.get(u, float('inf'))
            t_size = task_size_map.get((u, s), 200000.0)
            slack_adjusted = slack - t_size / 1000000.0 * 0.5
            ru = float(self.rank_u[u].get(s, 0.0))
            st = float(self.ts.start_time[u].get(s, float('inf')))
            return (slack_adjusted, -ru, st)
        return sorted(ready_tasks, key=sort_key)[:k]

    def get_slack_cp(self, uid, slot, now=None):
        if now is None:
            now = para['slot_interval'] * slot
        enter = self.ts.enter_time[uid]
        if enter == float('inf'):
            return float('inf')
        app_deadline_abs = enter + self.ts.get_app_deadline_slot(uid) * para['slot_interval']
        if uid not in self.rank_u:
            self._compute_rank_u_for_app(self.ts.env, uid, self.ts.env.task_complex_index)
        unfinished_tasks = [n for n in self.ts.subgraph_list[uid].nx_graph.nodes if self.ts.finish_time[uid][n] == float('inf') and self.ts._calculate_est(uid, n) <= now + 1e-09]
        if not unfinished_tasks:
            return float('inf')
        cp_remain = max((self.rank_u[uid].get(n, 0.0) for n in unfinished_tasks))
        return app_deadline_abs - now - cp_remain

    def get_app_dag_data(self, env, uid: int, slot: int, complex_index: int, current_subtask_id=None):
        g = self.ts.subgraph_list[uid].nx_graph
        nodes = list(g.nodes())
        node2idx = {nid: i for i, nid in enumerate(nodes)}
        if g.number_of_edges() > 0:
            edges = list(g.edges())
            src = [node2idx[a] for a, b in edges]
            dst = [node2idx[b] for a, b in edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        now = para['slot_interval'] * slot
        enter = self.ts.enter_time[uid]
        local_upload_wait = max(0.0, self.ts.devices_upload_useful[uid] - now)
        local_exe_wait = max(0.0, self.ts.devices_exe_useful[uid] - now)
        edge_waits = [max(0.0, float(t) - now) for t in self.ts.remain_times]
        edge_loads = []
        for eid in range(para['edge_num']):
            eid_load = sum((1 for t in self.ts.edge_useful[eid] if t != 0))
            edge_loads.append(min(1.0, eid_load / max(1, self.ts.edge_core)))
        dists = [min(1.0, float(d) / 1000.0) for d in env.device_list[uid].edge_distances]
        global_feats = [min(1.0, local_upload_wait / 2.0), min(1.0, local_exe_wait / 2.0)] + [min(1.0, w / 2.0) for w in edge_waits] + edge_loads + dists
        if uid not in self.rank_u:
            self._compute_rank_u_for_app(env, uid, complex_index)
        rank_u = self.rank_u.get(uid, {})
        max_ru = max(rank_u.values()) if rank_u else 1.0
        max_ru = max(max_ru, 1e-09)
        if enter != float('inf'):
            app_deadline_abs = enter + self.ts.get_app_deadline_slot(uid) * para['slot_interval']
            app_deadline_slots = self.ts.get_app_deadline_slot(uid)
        else:
            app_deadline_abs = 0.0
            app_deadline_slots = 100.0
        ready_nodes = []
        for nid in nodes:
            st = self.ts.start_time[uid].get(nid, float('inf'))
            ft = self.ts.finish_time[uid].get(nid, float('inf'))
            if st != float('inf') and ft == float('inf') and (st <= now):
                ready_nodes.append(nid)
        cp_remain = max((rank_u.get(n, 0.0) for n in ready_nodes)) if ready_nodes else 0.0
        potential_costs = []
        if current_subtask_id is not None:
            try:
                curr_task_size = self.ts.get_task_size_bytes(uid, current_subtask_id)
                for eid in range(para['edge_num']):
                    e_power = env.edges[eid].edge_power * env.edges[eid].calculate_parameter
                    exec_time = curr_task_size / e_power
                    queue_time = max(0.0, self.ts.remain_times[eid] - now)
                    total = queue_time + exec_time
                    potential_costs.append(min(1.0, total / 2.0))
            except:
                potential_costs = [0.0] * para['edge_num']
        else:
            potential_costs = [0.0] * para['edge_num']
        global_feats = global_feats + potential_costs
        x_rows = []
        for nid in nodes:
            st = self.ts.start_time[uid].get(nid, float('inf'))
            ft = self.ts.finish_time[uid].get(nid, float('inf'))
            is_ready = st != float('inf') and ft == float('inf') and (st <= now)
            is_done = ft != float('inf')
            ru = float(rank_u.get(nid, 0.0))
            ru_norm = ru / max_ru
            size_norm = float(self.ts.task_size_normalized[nid])
            if app_deadline_slots > 0:
                slack = app_deadline_abs - now
                slack_norm = min(1.0, slack / max(1e-06, app_deadline_slots * para['slot_interval']))
            else:
                slack_norm = 1.0
            if app_deadline_slots > 0:
                cp_slack = app_deadline_abs - now - cp_remain
                cp_slack_norm = min(1.0, cp_slack / max(1e-06, app_deadline_slots * para['slot_interval']))
            else:
                cp_slack_norm = 1.0
            x_rows.append([size_norm, float(is_ready), float(is_done), ru_norm, slack_norm, cp_slack_norm] + global_feats)
        x = torch.tensor(x_rows, dtype=torch.float32)
        batch = torch.zeros(x.size(0), dtype=torch.long)
        return (Data(x=x, edge_index=edge_index, batch=batch), node2idx)
