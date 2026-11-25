"""Wall-clock decision timing per active logical slot."""
import time
import torch

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class DecisionTimer:

    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = []
        self.slot = None
        self.started = None
        self.excluded = 0.0
        self.launches = 0

    def start(self, slot):
        if self.slot == slot:
            return
        self.finish()
        synchronize()
        self.slot = int(slot)
        self.started = time.perf_counter()
        self.excluded = 0.0
        self.launches = 0

    def finish(self):
        if self.started is not None:
            synchronize()
            elapsed = time.perf_counter() - self.started - self.excluded
            if self.launches:
                self.samples.append(dict(slot=self.slot, decisions=self.launches, decision_ms=1000 * max(0, elapsed)))
        self.started = None
        self.slot = None
