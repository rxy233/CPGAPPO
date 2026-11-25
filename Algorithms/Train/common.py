import os
import sys
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
experiments_dir = os.path.join(project_root, 'experiments')
if experiments_dir not in sys.path:
    sys.path.insert(0, experiments_dir)
print_lock = None

def set_print_lock(lock):
    global print_lock
    print_lock = lock

def decode_action(action: int) -> tuple:
    if action == 0:
        return ('Local', -1)
    if action == 1:
        return ('Cloud', -1)
    return ('Edge', action - 2)

def ensure_trace_dir(run_dir: str) -> Path:
    p = Path(run_dir) / 'traces'
    p.mkdir(parents=True, exist_ok=True)
    return p
