import random
from dataclasses import dataclass

@dataclass
class TaskSelector:
    mode: str = 'cp'
    seed: int = 0
    callback: object = None

    def order(self, gs, slot: int, tasks):
        if not tasks:
            return []
        m = (self.mode or 'cp').lower()
        if m == 'cp':
            task_infos = gs.get_tasks_priority_info_batch(tasks, slot)

            def cp_sort_key(task):
                info = task_infos[task]
                return (info['slack'], -info['rank_u'])
            return sorted(tasks, key=cp_sort_key)
        if m == 'cp_rev':
            task_infos = gs.get_tasks_priority_info_batch(tasks, slot)

            def cp_rev_sort_key(task):
                info = task_infos[task]
                return (-info['slack'], info['rank_u'])
            return sorted(tasks, key=cp_rev_sort_key)
        if m == 'fifo':
            return sorted(tasks, key=lambda t: gs.ts.start_time[t[0]].get(t[1], float('inf')))
        if m == 'slack':
            task_infos = gs.get_tasks_priority_info_batch(tasks, slot)
            return sorted(tasks, key=lambda t: task_infos[t]['slack'])
        if m == 'rank_u':
            task_infos = gs.get_tasks_priority_info_batch(tasks, slot)
            return sorted(tasks, key=lambda t: -task_infos[t]['rank_u'])
        if m == 'random':
            rng = random.Random((self.seed + 1) * 1000003 + slot)
            tasks2 = list(tasks)
            rng.shuffle(tasks2)
            return tasks2
        if m == 'custom' and self.callback is not None:
            return list(self.callback(gs, slot, list(tasks)))
        return list(tasks)
