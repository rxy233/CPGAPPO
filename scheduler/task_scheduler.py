import random
import os
import numpy as np
import torch
from utils.constant import para
from Environment.environment import Environment
from Environment import computation
import networkx as nx
from utils.tools import *
VERBOSE = os.environ.get('VERBOSE', '0') == '1'

def vprint(*args, **kwargs):
    if VERBOSE:
        pass

class TaskScheduler:
    TIMEOUT_EPSILON = 1e-05

    def __init__(self, user_num, subgraph_num, basegraph_num, env: Environment, tasksize=para['task_size_range'], tight_deadline_config=None, seed=None):
        from decision_timing import DecisionTimer
        self.decision_timer = DecisionTimer() if getattr(env, 'decision_timing', False) else None
        self.current_time = 0.0
        self.pipeline_dispatch_times = [[float('-inf'), float('-inf')] for _ in range(user_num)]
        self.user_num = user_num
        self.subgraph_num = subgraph_num
        self.basegraph_num = basegraph_num
        self.edge_core = para['edgecore_limit']
        self.env = env
        self.subgraph_list = self.env.subgraph_list
        self.rng = random.Random(seed) if seed is not None else random
        self.task_size = self.rng.choices(tasksize, k=basegraph_num)
        if len(self.task_size) > 0:
            sample_sizes = self.task_size[:min(5, len(self.task_size))]
            sample_kb = np.array(sample_sizes) / 1024
        self.task_size_normalized = minmax_normalize(self.task_size)
        self.overtime = 0
        self.total_energy = 0.0
        self.using_Algorithm = 0
        self.total_subtasks = sum((len(sub.nx_graph.nodes) for sub in self.subgraph_list))
        self.application_waiting = set([i for i in range(self.user_num)])
        self.application_started = set()
        self.application_finished = set()
        self.application_timeout_finished = set()
        self.application_task_timeout = set()
        self.remain_times = [0.0] * para['edge_num']
        self.route_stats = {'local': 0, 'cloud': 0, 'edge': 0, 'recent': []}
        self.devices_exe_useful = [0 for i in range(self.user_num)]
        self.devices_upload_useful = [0 for i in range(self.user_num)]
        self.edge_core_finish = [[0.0] * self.edge_core for _ in self.env.edges]
        self.edge_jobs = [[] for _ in self.env.edges]
        self.edge_useful = []
        self.core_remaining_work = []
        for _ in range(para['edge_num']):
            self.core_remaining_work.append([])
            self.edge_useful.append([0] * self.edge_core)
        self.rest_tasks = [len(sub.nx_graph.nodes) for sub in self.subgraph_list]
        self.energy = [0 for i in range(self.user_num)]
        self.de_wait_energy = [0 for i in range(self.user_num)]
        self.upload_energy = [0 for _ in range(self.user_num)]
        self.enter_time = [float('inf') for i in range(self.user_num)]
        self.exit_time = [float('inf') for i in range(self.user_num)]
        self.local_time = [0 for i in range(self.user_num)]
        self.upload_time = [0 for i in range(self.user_num)]
        if tight_deadline_config is not None:
            if 'deadline_slot_per_user' in tight_deadline_config:
                self.deadline_slot_per_user = list(tight_deadline_config['deadline_slot_per_user'])
                self.app_deadline_slots = tight_deadline_config.get('app_deadline_slots', {})
                default_slot = para['deadline_slot']
                tight_users = [uid for uid, slot in enumerate(self.deadline_slot_per_user) if slot != default_slot]
                num_tight_users = len(tight_users)
                if tight_users:
                    tight_slots = [self.deadline_slot_per_user[uid] for uid in tight_users]
                    min_slot = min(tight_slots)
                    max_slot = max(tight_slots)
                else:
                    min_slot = max_slot = default_slot
            else:
                tight_deadline_users = tight_deadline_config.get('tight_user_ids', [])
                self.app_deadline_slots = tight_deadline_config.get('app_deadline_slots', {})
                self.deadline_slot_per_user = [para['deadline_slot']] * self.user_num
                for uid, slot in self.app_deadline_slots.items():
                    self.deadline_slot_per_user[uid] = slot
                num_tight_users = len(tight_deadline_users)
                min_factor = min([self.app_deadline_slots.get(uid, para['deadline_slot']) / para['deadline_slot'] for uid in tight_deadline_users]) if tight_deadline_users else 1.0
                max_factor = max([self.app_deadline_slots.get(uid, para['deadline_slot']) / para['deadline_slot'] for uid in tight_deadline_users]) if tight_deadline_users else 1.0
                pass
        else:
            TIGHT_DEADLINE_RATIO = 0.3
            deadline_rng = random.Random(42)
            num_tight_users = int(self.user_num * TIGHT_DEADLINE_RATIO)
            tight_deadline_users = deadline_rng.sample(range(self.user_num), num_tight_users)
            self.app_deadline_slots = {}
            self.deadline_slot_per_user = [para['deadline_slot']] * self.user_num
            for idx, uid in enumerate(tight_deadline_users):
                factor = 0.6 + idx % 4 * 0.1
                tight_deadline = int(round(para['deadline_slot'] * factor))
                self.app_deadline_slots[uid] = tight_deadline
                self.deadline_slot_per_user[uid] = tight_deadline
            min_factor = 0.6
            max_factor = 0.9
            pass
            self._tight_deadline_config = {'tight_user_ids': tight_deadline_users, 'app_deadline_slots': self.app_deadline_slots.copy(), 'deadline_slot_per_user': self.deadline_slot_per_user.copy()}
        self.app_deadline_slots_critical_path = self._compute_critical_path_deadlines()
        self.finish_time = None
        self.start_time = None
        self.est_time = None
        self.ast_time = None
        self.cp_tasks = None
        self.init_times()
        self.remain_times = [0] * para['edge_num']

    def init_times(self):
        self.start_time = [{} for _ in range(self.user_num)]
        self.finish_time = [{} for _ in range(self.user_num)]
        self.est_time = [{} for _ in range(self.user_num)]
        self.ast_time = [{} for _ in range(self.user_num)]
        self.cp_tasks = [set() for _ in range(self.user_num)]
        for i, sub in enumerate(self.subgraph_list):
            for node_id in sub.nx_graph.nodes:
                self.start_time[i][node_id] = float('inf')
                self.finish_time[i][node_id] = float('inf')
                self.est_time[i][node_id] = None
                self.ast_time[i][node_id] = None
        self._init_critical_path_marks()

    def _compute_critical_path_deadlines(self):
        app_deadline_slots = {}
        slack_factor = para.get('app_deadline_slack_factor', 0.25)
        app_deadline_alpha = para.get('app_deadline_alpha', 0.35)
        tight_app_slack_factor = para.get('tight_app_slack_factor', 0.0)
        base_deadline_slot = para['deadline_slot']
        for user_id in range(self.user_num):
            subgraph = self.subgraph_list[user_id]
            try:
                cp_length = nx.dag_longest_path_length(subgraph.nx_graph) + 1
            except:
                cp_length = len(subgraph.nx_graph.nodes)
            app_deadline_slot = int(cp_length * base_deadline_slot * (1.0 + slack_factor) * app_deadline_alpha)
            app_deadline_slots[user_id] = app_deadline_slot
        if hasattr(self, 'deadline_slot_per_user') and self.deadline_slot_per_user is not None:
            for user_id in range(self.user_num):
                tight_slot = self.deadline_slot_per_user[user_id]
                subgraph = self.subgraph_list[user_id]
                try:
                    cp_length = nx.dag_longest_path_length(subgraph.nx_graph) + 1
                except:
                    cp_length = len(subgraph.nx_graph.nodes)
                tight_app_deadline = int(cp_length * tight_slot * (1.0 + tight_app_slack_factor) * app_deadline_alpha)
                app_deadline_slots[user_id] = min(app_deadline_slots[user_id], tight_app_deadline)
        elif hasattr(self, 'app_deadline_slots') and self.app_deadline_slots:
            for user_id, tight_slot in self.app_deadline_slots.items():
                subgraph = self.subgraph_list[user_id]
                try:
                    cp_length = nx.dag_longest_path_length(subgraph.nx_graph) + 1
                except:
                    cp_length = len(subgraph.nx_graph.nodes)
                tight_app_deadline = int(cp_length * tight_slot * (1.0 + tight_app_slack_factor) * app_deadline_alpha)
                app_deadline_slots[user_id] = min(app_deadline_slots[user_id], tight_app_deadline)
        if user_id >= 0:
            cp_lengths = []
            app_deadlines = []
            for uid in range(self.user_num):
                subgraph = self.subgraph_list[uid]
                try:
                    cp_len = nx.dag_longest_path_length(subgraph.nx_graph) + 1
                except:
                    cp_len = len(subgraph.nx_graph.nodes)
                cp_lengths.append(cp_len)
                app_deadlines.append(app_deadline_slots[uid])
        return app_deadline_slots

    def get_app_deadline_slot(self, user_id):
        if hasattr(self, 'app_deadline_slots_critical_path') and self.app_deadline_slots_critical_path is not None:
            return self.app_deadline_slots_critical_path[user_id]
        return self.app_deadline_slots.get(user_id, para['deadline_slot'])

    def get_task_deadline_slot(self, user_id):
        total_app_slot = 0
        if hasattr(self, 'deadline_slot_per_user') and self.deadline_slot_per_user is not None:
            total_app_slot = self.deadline_slot_per_user[user_id]
        else:
            total_app_slot = self.app_deadline_slots.get(user_id, para['deadline_slot'])
        try:
            subgraph = self.subgraph_list[user_id]
            cp_length = len(subgraph.nx_graph.nodes)
            cp_length = max(cp_length, 1)
        except:
            cp_length = 8
        task_deadline_slot = total_app_slot
        return task_deadline_slot

    def _init_critical_path_marks(self):
        for user_id in range(self.user_num):
            subgraph = self.subgraph_list[user_id]
            try:
                longest_path = nx.dag_longest_path(subgraph.nx_graph)
                self.cp_tasks[user_id] = set(longest_path)
            except Exception as e:
                self.cp_tasks[user_id] = set()

    def _calculate_est(self, user_id, subtask_id):
        subgraph = self.subgraph_list[user_id]
        if subtask_id in subgraph.enter_nodes:
            return self.enter_time[user_id]
        est = 0.0
        for pred in subgraph.nx_graph.predecessors(subtask_id):
            if pred in self.finish_time[user_id] and self.finish_time[user_id][pred] != float('inf'):
                est = max(est, self.finish_time[user_id][pred])
            else:
                return float('inf')
        return est

    def compute_cp_metrics(self, eps=1e-09):
        cp_wait_list = []
        cp_ratio_list = []
        for user_id in range(self.user_num):
            if user_id not in self.application_finished:
                continue
            cp_task_set = self.cp_tasks[user_id]
            if not cp_task_set:
                continue
            cp_wait = 0.0
            for task_id in cp_task_set:
                if task_id not in self.finish_time[user_id] or task_id not in self.ast_time[user_id] or task_id not in self.est_time[user_id]:
                    continue
                ast = self.ast_time[user_id][task_id]
                est = self.est_time[user_id][task_id]
                if ast is not None and est is not None:
                    cp_wait += max(0.0, ast - est)
            if task_id not in self.finish_time[user_id]:
                continue
            makespan = self.finish_time[user_id][task_id] - self.enter_time[user_id]
            cp_ratio = cp_wait / max(makespan, eps)
            cp_wait_list.append(cp_wait)
            cp_ratio_list.append(cp_ratio)
        cp_wait_avg = float(np.mean(cp_wait_list)) if cp_wait_list else 0.0
        cp_ratio_avg = float(np.mean(cp_ratio_list)) if cp_ratio_list else 0.0
        return (cp_wait_avg, cp_ratio_avg)

    def get_gantt_data(self, user_id):
        if user_id >= len(self.subgraph_list):
            return []
        gantt_data = []
        cp_set = self.cp_tasks[user_id]
        for task_id in self.subgraph_list[user_id].nx_graph.nodes:
            if task_id not in self.finish_time[user_id] or task_id not in self.ast_time[user_id]:
                continue
            ft = self.finish_time[user_id][task_id]
            ast = self.ast_time[user_id][task_id]
            if ast is not None and ft != float('inf'):
                duration = ft - ast
                task_size = self.get_task_size_bytes(user_id, task_id)
                f_local = float(self.env.device_list[user_id].local_power)
                _, local_exec_time = computation.execute_consumption(task_size, f_local, self.env.task_complex_index, 'l')
                if duration <= local_exec_time * 1.1:
                    channel = 'local'
                else:
                    channel = 'up'
                gantt_data.append({'task_id': task_id, 'channel': channel, 'ast': ast, 'ft': ft, 'is_cp': task_id in cp_set, 'action': 0 if channel == 'local' else 1})
        return gantt_data

    def new_arrival(self, num_to_select, slot, fixed_uids=None):
        if not self.application_waiting:
            return []
        if fixed_uids is not None:
            chosen = set()
            for uid in fixed_uids:
                if uid in self.application_waiting:
                    chosen.add(uid)
                else:
                    continue
        elif num_to_select >= len(self.application_waiting):
            chosen = self.application_waiting.copy()
        else:
            k = min(num_to_select, len(self.application_waiting))
            chosen = set(random.sample(list(self.application_waiting), k=k))
        self.application_waiting -= chosen
        self.application_started = self.application_started | chosen
        for c in chosen:
            self.enter_time[c] = slot * para['slot_interval']
            for en in self.subgraph_list[c].enter_nodes:
                self.start_time[c][en] = slot * para['slot_interval']
        return chosen

    def check_prev_all_finished(self, user_id, subtask_id):
        predecessors = self.subgraph_list[user_id].nx_graph.predecessors(subtask_id)
        for prev in predecessors:
            if prev in self.finish_time[user_id] and self.finish_time[user_id][prev] == float('inf'):
                return False
        return True

    def calculate_start_time(self, user_id, subtask_id, local):
        est = self._calculate_est(user_id, subtask_id)
        self.est_time[user_id][subtask_id] = est
        device_available_time = self.devices_exe_useful[user_id] if local else self.devices_upload_useful[user_id]
        ast = max(est, device_available_time)
        self.ast_time[user_id][subtask_id] = ast
        return ast

    def edge_update(self, task_info, task_size, upload_consumption, execute_consumption, edge_id):
        """Equation (9): non-preemptive execution on the first available core."""
        upload_e, upload_d = upload_consumption
        execute_e, execute_d = execute_consumption
        start = self.calculate_start_time(*task_info, False)
        core = int(np.argmin(self.edge_core_finish[edge_id]))
        execute_start = max(start + upload_d, self.edge_core_finish[edge_id][core])
        finish = execute_start + execute_d
        energy = upload_e + execute_e
        self.update(task_info, energy, finish - start, local=False, upload_duration=upload_d, exec_duration=execute_d)
        self.edge_core_finish[edge_id][core] = finish
        self.edge_jobs[edge_id].append((core, execute_start, finish, task_size))
        self._refresh_edge_resources(self.current_time)
        uid, sid = task_info
        delay = self.finish_time[uid][sid] - self.start_time[uid][sid]
        return (energy, delay, self.remain_times[edge_id])

    def _refresh_edge_resources(self, now):
        """Release completed jobs; retain cycles and storage for queued work."""
        for eid, edge in enumerate(self.env.edges):
            jobs = [job for job in self.edge_jobs[eid] if job[2] > now]
            self.edge_jobs[eid] = jobs
            work = [0.0] * self.edge_core
            frequency = edge.edge_power * edge.calculate_parameter
            for core, begin, end, size in jobs:
                work[core] += max(0.0, end - max(now, begin)) * frequency
            self.core_remaining_work[eid] = sorted((w for w in work if w > 0))
            self.edge_useful[eid] = sorted(self.edge_core_finish[eid])
            self.remain_times[eid] = min(self.edge_useful[eid])
            edge.task_count = len(jobs)
            edge.total_task_size = sum((job[3] for job in jobs))
            edge.used_storage = edge.total_task_size * para['task_storage_index']
            edge.total_difficulty = sum(work)

    def is_core_remaining_empty(self, edge_id):
        return not self.core_remaining_work[edge_id]

    def edge_doTask_inslot(self):
        self._refresh_edge_resources(self.current_time)

    def update(self, task_info, energy, delay, local=False, upload_duration=0, exec_duration=0):
        user_id, subtask_id = task_info
        task_start_time = self.calculate_start_time(user_id, subtask_id, local)
        task_deadline_abs = self.enter_time[user_id] + self.get_task_deadline_slot(user_id) * para['slot_interval']
        app_deadline_abs = self.enter_time[user_id] + self.get_app_deadline_slot(user_id) * para['slot_interval']
        estimated_finish = task_start_time + delay
        if estimated_finish > task_deadline_abs + self.TIMEOUT_EPSILON:
            if not hasattr(self, 'application_task_timeout'):
                self.application_task_timeout = set()
            self.application_task_timeout.add(user_id)
        if estimated_finish > app_deadline_abs + self.TIMEOUT_EPSILON:
            pass
        self.energy[user_id] += energy
        self.total_energy += energy
        if not local:
            self.devices_upload_useful[user_id] = max(task_start_time, self.current_time) + upload_duration
            self.pipeline_dispatch_times[user_id][1] = self.current_time
        else:
            self.devices_exe_useful[user_id] = max(task_start_time, self.current_time) + exec_duration
            self.pipeline_dispatch_times[user_id][0] = self.current_time
        self.finish_time[user_id][subtask_id] = task_start_time + delay
        self.rest_tasks[user_id] -= 1
        if self.rest_tasks[user_id] < 0:
            self.rest_tasks[user_id] = 0
        task_deadline_abs = self.enter_time[user_id] + self.get_task_deadline_slot(user_id) * para['slot_interval']
        finish_abs = self.finish_time[user_id][subtask_id]
        EPS = 1e-06
        if self.enter_time[user_id] != float('inf') and finish_abs > task_deadline_abs + EPS:
            self.overtime += 1
            if hasattr(self, 'application_task_timeout'):
                self.application_task_timeout.add(user_id)
        successors = self.subgraph_list[user_id].nx_graph.successors(subtask_id)
        for succ in successors:
            if succ in self.start_time[user_id] and self.check_prev_all_finished(user_id, succ):
                max_pred_finish = 0
                for pred in self.subgraph_list[user_id].nx_graph.predecessors(succ):
                    if pred in self.finish_time[user_id]:
                        max_pred_finish = max(max_pred_finish, self.finish_time[user_id][pred])
                self.start_time[user_id][succ] = max_pred_finish
        if self.is_application_done(user_id):
            self.exit_time[user_id] = max(self.finish_time[user_id].values())
            self.application_finished.add(user_id)
            deadline_abs = self.enter_time[user_id] + self.get_app_deadline_slot(user_id) * para['slot_interval']
            EPS = 1e-06
            if self.enter_time[user_id] != float('inf') and self.exit_time[user_id] > deadline_abs + EPS:
                self.application_timeout_finished.add(user_id)
        delay = self.finish_time[user_id][subtask_id] - self.start_time[user_id][subtask_id]
        return (energy, delay)

    def is_application_done(self, user_id):
        for v in self.finish_time[user_id].values():
            if v == float('inf'):
                return False
        return True

    def is_done(self):
        return len(self.application_finished) == self.user_num

    def finalize_episode(self, end_slot):
        if self.decision_timer is not None:
            self.decision_timer.finish()
        end_time = end_slot * para['slot_interval']
        if not hasattr(self, 'deadline_debug_count'):
            self.deadline_debug_count = 0
        for uid in range(self.user_num):
            if self.enter_time[uid] == float('inf'):
                continue
            app_deadline_slot = self.get_app_deadline_slot(uid)
            deadline_abs = self.enter_time[uid] + app_deadline_slot * para['slot_interval']
            if self.deadline_debug_count < 10:
                finish_time = self.exit_time[uid] if self.exit_time[uid] != float('inf') else 0
                is_timeout = finish_time > deadline_abs + self.TIMEOUT_EPSILON if self.is_application_done(uid) else True
                self.deadline_debug_count += 1
            if not self.is_application_done(uid):
                self.application_timeout_finished.add(uid)
                if self.exit_time[uid] == float('inf'):
                    self.exit_time[uid] = min(end_time, deadline_abs)
                continue
            if self.exit_time[uid] == float('inf'):
                self.exit_time[uid] = max(self.finish_time[uid].values())
            if self.exit_time[uid] > deadline_abs + self.TIMEOUT_EPSILON:
                self.application_timeout_finished.add(uid)

    def check_timeouts(self, current_slot):
        current_time = current_slot * para['slot_interval']
        self.current_time = current_time
        self._refresh_edge_resources(current_time)
        for uid in range(self.user_num):
            if uid in self.application_finished or uid in self.application_timeout_finished:
                continue
            if self.enter_time[uid] == float('inf'):
                continue
            app_deadline_abs = self.enter_time[uid] + self.get_app_deadline_slot(uid) * para['slot_interval']
            task_deadline_abs = self.enter_time[uid] + self.get_task_deadline_slot(uid) * para['slot_interval']
            for subtask_id in self.finish_time[uid]:
                if self.finish_time[uid][subtask_id] != float('inf'):
                    if self.finish_time[uid][subtask_id] > task_deadline_abs + self.TIMEOUT_EPSILON:
                        if not hasattr(self, 'application_task_timeout'):
                            self.application_task_timeout = set()
                        self.application_task_timeout.add(uid)
            if current_time > app_deadline_abs + self.TIMEOUT_EPSILON and (not self.is_application_done(uid)):
                self.application_timeout_finished.add(uid)
                self.exit_time[uid] = app_deadline_abs
                self.rest_tasks[uid] = 0

    def reset(self):
        if self.decision_timer is not None:
            self.decision_timer.reset()
        self.current_time = 0.0
        self.remain_times = [0.0] * para['edge_num']
        self.route_stats = {'local': 0, 'cloud': 0, 'edge': 0, 'recent': []}
        self.pipeline_dispatch_times = [[float('-inf'), float('-inf')] for _ in range(self.user_num)]
        self.application_waiting = set([i for i in range(self.user_num)])
        self.application_started = set()
        self.application_finished = set()
        self.application_timeout_finished = set()
        self.application_task_timeout = set()
        self.overtime = 0
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
        if self._reset_count == 0:
            if hasattr(self, 'deadline_slot_per_user') and self.deadline_slot_per_user is not None:
                default_slot = para['deadline_slot']
                tight_users = [uid for uid, slot in enumerate(self.deadline_slot_per_user) if slot != default_slot]
                num_tight_users = len(tight_users)
                if tight_users:
                    tight_slots = [self.deadline_slot_per_user[uid] for uid in tight_users]
                    min_slot = min(tight_slots)
                    max_slot = max(tight_slots)
                else:
                    min_slot = max_slot = default_slot
            else:
                num_tight_users = len([uid for uid in range(self.user_num) if uid in self.app_deadline_slots])
                factors = [self.app_deadline_slots.get(uid, para['deadline_slot']) / para['deadline_slot'] for uid in range(self.user_num) if uid in self.app_deadline_slots]
                min_factor = min(factors) if factors else 1.0
                max_factor = max(factors) if factors else 1.0
        else:
            pass
        self._reset_count += 1
        self.devices_exe_useful = [0 for i in range(self.user_num)]
        self.devices_upload_useful = [0 for i in range(self.user_num)]
        self.local_time = [0 for i in range(self.user_num)]
        self.upload_time = [0 for i in range(self.user_num)]
        self.rest_tasks = [len(sub.nx_graph.nodes) for sub in self.subgraph_list]
        self.edge_core_finish = [[0.0] * self.edge_core for _ in self.env.edges]
        self.edge_jobs = [[] for _ in self.env.edges]
        self.edge_useful = []
        self.core_remaining_work = []
        for _ in range(para['edge_num']):
            self.core_remaining_work.append([])
            self.edge_useful.append([0] * para['edgecore_limit'])
        for edge_id in range(para['edge_num']):
            target_edge = self.env.edges[edge_id]
            target_edge.task_count = 0
            target_edge.total_task_size = 0
            target_edge.total_difficulty = 0
            target_edge.used_storage = 0
            target_edge.current_remainTime = 0
            target_edge.calculate_parameter = 1
        self.energy = [0 for i in range(self.user_num)]
        self.de_wait_energy = [0 for i in range(self.user_num)]
        self.total_energy = 0.0
        self.upload_energy = [0 for _ in range(self.user_num)]
        self.enter_time = [float('inf') for i in range(self.user_num)]
        self.exit_time = [float('inf') for i in range(self.user_num)]
        self.init_times()

    def get_avg_results(self, only_successful=False, timeout_charge='2x_deadline'):
        """Average UE energy and latency charges over arrived applications.

        Paper evaluation uses only_successful=False and timeout_charge='deadline'.
        Finite exits outside the timeout-finished set incur recorded elapsed
        time; other arrivals incur their deadline budget. Late completions
        can be in both terminal sets and thus receive a deadline charge.
        The only_successful compatibility option filters latency to finite
        exits outside the timeout-finished set; energy still uses all arrivals."""
        for uid in range(self.user_num):
            application_time = self.exit_time[uid] - self.enter_time[uid]
        arrived = [uid for uid in range(self.user_num) if np.isfinite(self.enter_time[uid])]
        avg_energy = sum((self.energy[uid] for uid in arrived)) / max(1, len(arrived))
        valid_delays = []
        deadline_limit = para['deadline_slot'] * para['slot_interval']
        for uid in arrived:
            app_deadline = self.enter_time[uid] + self.get_app_deadline_slot(uid) * para['slot_interval']
            app_deadline_budget = self.get_app_deadline_slot(uid) * para['slot_interval']
            if only_successful:
                if uid not in self.application_timeout_finished and self.exit_time[uid] != float('inf'):
                    delay = self.exit_time[uid] - self.enter_time[uid]
                    valid_delays.append(delay)
            else:
                if uid not in self.application_timeout_finished and self.exit_time[uid] != float('inf'):
                    delay = self.exit_time[uid] - self.enter_time[uid]
                elif timeout_charge == 'deadline':
                    delay = app_deadline_budget
                else:
                    delay = deadline_limit * 2
                valid_delays.append(delay)
        if len(valid_delays) > 0:
            avg_delay = sum(valid_delays) / len(valid_delays)
        elif timeout_charge == 'deadline':
            avg_delay = deadline_limit
        else:
            avg_delay = deadline_limit * 2
        return (avg_energy, avg_delay)

    def get_sum_energy(self):
        return sum(self.energy)

    def get_delays(self):
        return [exi - ent for exi, ent in zip(self.exit_time, self.enter_time)]

    def get_finish_times(self):
        flattened_list = [element for sublist in self.finish_time for element in sublist.values()]
        return flattened_list

    def get_system_state(self):
        freq = [d.local_power for d in self.env.device_list]
        useful = self.devices_exe_useful
        rest = self.rest_tasks
        freq.extend([self.env.edge.edge_power * self.env.edge.calculate_parameter, self.env.cloud.cloud_power])
        useful.extend([self.edge_useful[0] if self.edge_useful else 0, len(self.edge_useful)])
        rest.extend([0, 0])
        freq = normalize_list(freq)
        useful = normalize_list(useful)
        rest = normalize_list(rest)
        res = [[d, u, r] for d, u, r in zip(freq, useful, rest)]
        res = [item for sublist in res for item in sublist]
        return torch.Tensor([res])

    def get_action_mask(self, user_id, task_size, current_time=None):
        """Return resource-feasible placements at the decision boundary.

        Check local/uplink occupancy, same-slot launches, edge reachability,
        cloud availability, and edge storage. Application readiness and
        CP/deadline-risk filtering are handled by dispatch callers."""
        now = self.current_time if current_time is None else float(current_time)
        device = self.env.device_list[user_id]
        mask = torch.ones(para['edge_num'] + 2, dtype=torch.bool)
        launches = self.pipeline_dispatch_times[user_id]
        mask[0] = device.local_power > 0 and self.devices_exe_useful[user_id] <= now + 1e-09 and (launches[0] < now)
        uplink_idle = self.devices_upload_useful[user_id] <= now + 1e-09 and launches[1] < now
        reachable = [bool(np.isfinite(d) and d <= para['edge_radius']) for d in device.edge_distances]
        mask[1] = uplink_idle and any(reachable) and self.env.cloud.is_available()
        for eid, edge in enumerate(self.env.edges):
            mask[eid + 2] = uplink_idle and reachable[eid] and (edge.used_storage + task_size * para['task_storage_index'] <= edge.max_storage)
        return mask

    def get_complete_infos(self, task_info, local_delay):
        user_id, subtask_id = task_info
        deadline_time = self.get_app_deadline_slot(user_id) * para['slot_interval']
        deadline = deadline_time + self.enter_time[user_id]
        remain_rate = self.rest_tasks[user_id] / len(self.subgraph_list[user_id].nx_graph.nodes)
        complete_rate = 1 - remain_rate
        if remain_rate < 0:
            input()
        current_time = self.finish_time[user_id][subtask_id]
        time_remaining = deadline - current_time
        if deadline < current_time:
            self.overtime += 1
            if hasattr(self, 'application_task_timeout'):
                self.application_task_timeout.add(user_id)
        time_ratio = time_remaining / deadline_time
        if time_ratio > 0.2:
            deadline_rate = time_ratio
        elif time_ratio < 0:
            deadline_rate = 10 * time_ratio - 10
        else:
            deadline_rate = 1 - (1 - time_ratio) * 2
        return (complete_rate, deadline_rate)

    def get_task_size_bytes(self, user_id: int, subtask_id: int) -> float:
        s = getattr(self, 'task_size', None)
        if s is None:
            return 0.0
        try:
            if isinstance(s, (list, tuple, np.ndarray)) and len(s) > 0 and isinstance(s[0], (list, tuple, np.ndarray)):
                uid = int(user_id)
                if uid < 0 or uid >= len(s):
                    uid = max(0, min(uid, len(s) - 1))
                row = s[uid]
                if row is None or len(row) == 0:
                    return 0.0
                sid = int(subtask_id)
                if 0 <= sid < len(row):
                    return float(row[sid])
                sid = max(0, min(sid, len(row) - 1))
                return float(row[sid])
            if isinstance(s, (list, tuple, np.ndarray)):
                sid = int(subtask_id)
                if 0 <= sid < len(s):
                    return float(s[sid])
                sid = max(0, min(sid, len(s) - 1))
                return float(s[sid])
            if isinstance(s, dict):
                if subtask_id in s:
                    return float(s[subtask_id])
                return float(next(iter(s.values())))
        except Exception:
            pass
        return 0.0

    def add_task(self, user_id, subtask_id, node_id=None, *args, **kwargs):
        """Commit a placement already checked by the dispatch caller.

        Normal training/evaluation entry points call Benchmark.step, which
        rechecks get_action_mask immediately before this method. This low-level
        primitive assumes a ready task and a valid, feasible placement."""
        if hasattr(self.env, 'task_complex_index'):
            task_complex_index = self.env.task_complex_index
        else:
            task_complex_index = 0
        task_size = self.get_task_size_bytes(user_id, subtask_id)
        if task_size == 0:
            task_size = 200000.0
        task_info = (user_id, subtask_id)
        if node_id == 0:
            from Environment import computation
            f_local = float(self.env.device_list[user_id].local_power)
            upload_e, upload_d = (0.0, 0.0)
            exec_e, exec_d = computation.execute_consumption(task_size, f_local, task_complex_index, 'l')
            e, d = self.update(task_info, exec_e, exec_d, local=True, upload_duration=0, exec_duration=exec_d)
            return (e, d)
        elif node_id == 1:
            from Environment import computation
            nearest_edge = int(np.argmin(self.env.device_list[user_id].edge_distances))
            distance = float(self.env.device_list[user_id].edge_distances[nearest_edge])
            bandwidth = self.env.device_list[user_id].local_bw
            upload_e1, upload_d1 = computation.upload_consumption([task_size, distance, bandwidth], 1, 'e', local_trans=self.env.device_list[user_id].local_trans)
            upload_e2, upload_d2 = computation.upload_consumption(task_size, 1, 'c')
            fc = float(self.env.cloud.cloud_power)
            exec_e, exec_d = computation.execute_consumption(task_size, fc, task_complex_index, 'c')
            total_energy = upload_e1 + upload_e2 + exec_e
            total_delay = upload_d1 + upload_d2 + exec_d
            self.update(task_info, total_energy, total_delay, local=False, upload_duration=upload_d1, exec_duration=exec_d)
            return (total_energy, total_delay)
        elif node_id >= 2:
            edge_id = node_id - 2
            from Environment import computation
            distance = float(self.env.device_list[user_id].edge_distances[edge_id])
            bandwidth = self.env.device_list[user_id].local_bw
            upload_e, upload_d = computation.upload_consumption([task_size, distance, bandwidth], 1, 'e', local_trans=self.env.device_list[user_id].local_trans)
            f_edge = self.env.edges[edge_id].edge_power * self.env.edges[edge_id].calculate_parameter
            exec_e, exec_d = computation.execute_consumption(task_size, f_edge, task_complex_index, 'e')
            e, d, remain = self.edge_update(task_info, task_size, (upload_e, upload_d), (exec_e, exec_d), edge_id)
            return (e, d)
        else:
            from Environment import computation
            f_local = float(self.env.device_list[user_id].local_power)
            exec_e, exec_d = computation.execute_consumption(task_size, f_local, task_complex_index, 'l')
            e, d = self.update(task_info, exec_e, exec_d, local=True, upload_duration=0, exec_duration=exec_d)
            return (e, d)
