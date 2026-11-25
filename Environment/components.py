from utils.constant import *
import random
import numpy as np
import traceback
_DEVICE_WARN_PRINTED = False

class Device:

    def __init__(self, number, x, y, edge_distances, edge_scores, edge_inrange, task_complex_index=0, local_power=None, local_wait=None, local_trans=None):
        self.number = number
        self.task_complex_index = task_complex_index
        if local_power is not None:
            self.local_power = float(local_power)
        elif 'local_power_range' in para:
            global _DEVICE_WARN_PRINTED
            if not _DEVICE_WARN_PRINTED:
                _DEVICE_WARN_PRINTED = True
                traceback.print_stack(limit=15)
            import warnings
            warnings.warn('CPU frequency not supplied; using deterministic device parameters')
            low, high = para['local_power_range']
            rng = np.random.default_rng(number)
            self.local_power = float(rng.uniform(low, high))
        elif 'local_power' in para and len(para['local_power']) > 0:
            power_idx = number % len(para['local_power'])
            self.local_power = float(para['local_power'][power_idx])
        else:
            raise KeyError(f'Device[{number}]: local_power 参数未传入，且 para 中未定义 local_power_range 或 local_power')
        if local_wait is not None:
            self.local_wait = float(local_wait)
        else:
            self.local_wait = para.get('local_wait', 0.1)
        if local_trans is not None:
            self.local_trans = float(local_trans)
        else:
            self.local_trans = para.get('local_trans', 1.5)
        self.pos_x = x
        self.pos_y = y
        self.edge_distances = edge_distances
        self.edge_scores = edge_scores
        self.edge_inrange = edge_inrange

    def is_available(self):
        if hasattr(self, 'local_power') and self.local_power <= 0:
            return False
        return True

    def get_compute_power(self):
        if hasattr(self, 'local_power'):
            return float(self.local_power)
        return 0.0

    def get_idle_power(self):
        if hasattr(self, 'local_wait'):
            return float(self.local_wait)
        return para.get('local_wait', 0.1)

    def get_transmission_power(self):
        if hasattr(self, 'local_trans'):
            return float(self.local_trans)
        return para.get('local_trans', 1.5)

class EdgeServer:

    def __init__(self, number, x, y, fe, task_complex_index=0):
        self.number = number
        self.pos_x = x
        self.pos_y = y
        self.edge_power = fe
        self.edge_near = []
        self.task_complex_index = task_complex_index
        self.current_remainTime = 0
        self.task_count = 0
        self.total_task_size = 0
        self.total_difficulty = 0
        self.used_storage = 0
        self.max_storage = 6000 * 2 ** 10
        self.calculate_parameter = 1
        self.bw_occupy = {}
        self.bw_occupy_index = []

    def add_new_task(self, task_size, complex_index):
        self.task_count += 1
        self.total_task_size += task_size
        self.total_difficulty += task_size * complex_index
        self.used_storage += task_size * para['task_storage_index']

    def update_core_speeds(self, load_rates):
        threshold = 0.5
        if load_rates < threshold:
            self.calculate_parameter = 1
        else:
            self.calculate_parameter = 1

    def is_available(self):
        if hasattr(self, 'max_storage') and hasattr(self, 'used_storage'):
            if self.used_storage >= self.max_storage:
                return False
        if hasattr(self, 'task_count'):
            max_tasks = para.get('edgecore_limit', 5) * 10
            if self.task_count >= max_tasks:
                return False
        return True

    def get_current_load(self):
        if hasattr(self, 'task_count'):
            max_tasks = para.get('edgecore_limit', 5)
            return min(1.0, self.task_count / max(max_tasks, 1))
        return 0.0

    def get_queue_time(self):
        if hasattr(self, 'current_remainTime'):
            return max(0.0, float(self.current_remainTime))
        return 0.0

    def get_available_storage(self):
        if hasattr(self, 'max_storage') and hasattr(self, 'used_storage'):
            return max(0.0, float(self.max_storage - self.used_storage))
        return float('inf')

    def get_compute_power(self):
        base_power = float(self.edge_power) if hasattr(self, 'edge_power') else 0.0
        calc_param = float(self.calculate_parameter) if hasattr(self, 'calculate_parameter') else 1.0
        return base_power * calc_param

class Cloud:

    def __init__(self, task_complex_index=0):
        self.cloud_power = para['cloud_power']
        self.task_complex_index = task_complex_index

    def is_available(self):
        return True

    def get_current_load(self):
        return 0.0

    def get_compute_power(self):
        if hasattr(self, 'cloud_power'):
            return float(self.cloud_power)
        return para.get('cloud_power', 10000000000.0)

    def get_latency_overhead(self):
        return para.get('cloud_wan_rtt', 0.0)
