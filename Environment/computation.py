"""Communication and computation costs, in seconds and joules."""
import math
from utils.constant import para

def upload_consumption(data, target_load_num, type='e', local_trans=None, local_wait=None):
    """Equation (8) and UE radio energy; wired forwarding has no UE energy."""
    power = para['local_trans'] if local_trans is None else local_trans
    if type == 'e':
        size, distance, bandwidth_mhz = data
        if size < 0 or distance <= 0 or bandwidth_mhz <= 0 or (power <= 0):
            raise ValueError('Invalid uplink physical parameters')
        rate = bandwidth_mhz * 1000000.0 * math.log2(1 + power * distance ** (-para['theta']) / para['yita2'])
        delay = 8 * size / rate
        return (power * delay, delay)
    if type == 'c':
        return (0.0, 8 * data / para['R_ec'])
    raise ValueError(f'Unknown transmission type: {type}')

def execute_consumption(data_size, f, task_complex_index, type='l', local_wait=None):
    """Runtime = bytes * cycles/byte / Hz; only UE computation consumes UE energy."""
    if data_size < 0 or f <= 0 or (not math.isfinite(f)):
        raise ValueError('Invalid execution size or frequency')
    if type not in ('l', 'e', 'c'):
        raise ValueError(f'Unknown execution type: {type}')
    delay = data_size * para['task_complex'][task_complex_index] / f
    energy = para['kappa'] * f ** 3 * delay if type == 'l' else 0.0
    return (energy, delay)
