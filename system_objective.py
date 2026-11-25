"""System-level accounting, separate from the PPO surrogate objective."""
import math

def instantaneous_power(ts, uid):
    """Equation (5), immediately after dispatch at the current slot boundary."""
    from utils.constant import para
    d = ts.env.device_list[uid]
    local = para['kappa'] * d.local_power ** 3 if ts.devices_exe_useful[uid] > ts.current_time else 0
    uplink = d.local_trans if ts.devices_upload_useful[uid] > ts.current_time else 0
    return local + uplink

def jsop_cost(ts, w_delay, w_energy, w_sla):
    """Evaluate Eq. (15) with strict application deadline violations.

    Late recorded completion incurs a deadline violation here while
    counting as complete in Eq. (36) AppTimeout. This system objective
    is separate from the PPO surrogate and checkpoint-selection cost."""
    from utils.constant import para
    if min(w_delay, w_energy, w_sla) < 0:
        raise ValueError('System-objective weights must be nonnegative')
    arrived = [i for i in range(ts.user_num) if math.isfinite(ts.enter_time[i])]
    charges = []
    violations = 0
    for i in arrived:
        budget = ts.get_app_deadline_slot(i) * para['slot_interval']
        recorded = ts.exit_time[i]
        completed = i in ts.application_finished
        violations += int(not completed or recorded > ts.enter_time[i] + budget)
        charge = recorded - ts.enter_time[i] if math.isfinite(recorded) and i not in ts.application_timeout_finished else budget
        charges.append(charge)
    return w_delay * sum(charges) + w_energy * sum((ts.energy[i] for i in arrived)) + w_sla * violations
