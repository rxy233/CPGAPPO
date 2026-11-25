import torch
from torch_geometric.data import Data
import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from utils.constant import para
from Environment import computation
NODE_FEATURE_DIM = 27

def _estimate_placement_costs(ts, uid, task_size_bytes, task_complex_index, now):
    env = ts.env
    device = env.device_list[uid]
    local_freq = device.local_power
    _, t_local = computation.execute_consumption(task_size_bytes, local_freq, task_complex_index, 'l', local_wait=device.local_wait)
    local_queue = max(0.0, float(ts.devices_exe_useful[uid]) - now)
    est_local_delay = local_queue + t_local
    upload_bw = device.local_bw if hasattr(device, 'local_bw') else np.mean(para['uplink_range'])
    cloud_freq = env.cloud.cloud_power
    _, t_cloud_exec = computation.execute_consumption(task_size_bytes, cloud_freq, task_complex_index, 'c')
    nearest_edge = int(np.argmin(device.edge_distances))
    nearest_dist = float(device.edge_distances[nearest_edge])
    _, t_cloud_upload_ue = computation.upload_consumption([task_size_bytes, nearest_dist, upload_bw], 0, 'e', local_trans=device.local_trans if hasattr(device, 'local_trans') else para['local_trans'], local_wait=device.local_wait)
    _, t_cloud_upload_ec = computation.upload_consumption(task_size_bytes, 0, 'c', local_trans=device.local_trans if hasattr(device, 'local_trans') else para['local_trans'], local_wait=device.local_wait)
    upload_queue = max(0.0, float(ts.devices_upload_useful[uid]) - now)
    est_cloud_delay = upload_queue + t_cloud_upload_ue + t_cloud_upload_ec + t_cloud_exec
    edge_num = para['edge_num']
    best_edge_delay = float('inf')
    best_edge_eid = -1
    best_edge_queue = float('inf')
    best_edge_storage_frac = 1.0
    for eid in range(edge_num):
        dist = device.edge_distances[eid]
        if dist == float('inf') or dist > para['edge_radius']:
            continue
        edge_freq = float(env.edges[eid].edge_power * env.edges[eid].calculate_parameter)
        _, t_edge_exec = computation.execute_consumption(task_size_bytes, edge_freq, task_complex_index, 'e')
        _, t_edge_upload = computation.upload_consumption([task_size_bytes, dist, upload_bw], 0, 'e', local_trans=device.local_trans if hasattr(device, 'local_trans') else para['local_trans'], local_wait=device.local_wait)
        edge_queue = 0.0
        if eid < len(ts.core_remaining_work) and ts.core_remaining_work[eid]:
            total_work = sum(ts.core_remaining_work[eid])
            edge_queue = total_work / (edge_freq * para['slot_interval']) * para['slot_interval']
            if ts.core_remaining_work[eid]:
                earliest_work = ts.core_remaining_work[eid][0]
                edge_queue = earliest_work / (edge_freq * para['slot_interval']) * para['slot_interval']
        total_edge_delay = upload_queue + t_edge_upload + edge_queue + t_edge_exec
        if total_edge_delay < best_edge_delay:
            best_edge_delay = total_edge_delay
            best_edge_eid = eid
            edge_core = para.get('edgecore_limit', 4)
            if eid < len(ts.core_remaining_work):
                best_edge_queue = len(ts.core_remaining_work[eid]) / max(1, edge_core)
            ms = env.edges[eid].max_storage if hasattr(env.edges[eid], 'max_storage') else 1.0
            us = env.edges[eid].used_storage if hasattr(env.edges[eid], 'used_storage') else 0.0
            best_edge_storage_frac = min(1.0, float(us) / max(1.0, float(ms)))
    reachable = 0
    for eid in range(edge_num):
        dist = device.edge_distances[eid]
        if dist != float('inf') and dist <= para['edge_radius']:
            reachable += 1
    return {'est_local_delay': est_local_delay, 'est_cloud_delay': est_cloud_delay, 'best_edge_delay': best_edge_delay if best_edge_eid >= 0 else est_cloud_delay, 'best_edge_eid': best_edge_eid, 'best_edge_queue': best_edge_queue if best_edge_eid >= 0 else 1.0, 'best_edge_storage_frac': best_edge_storage_frac, 'reachable_edge_count': reachable, 'edge_num': edge_num, 'local_power': local_freq}

def _get_route_structure_features(ts, uid, now):
    stats = getattr(ts, 'route_stats', None)
    if stats is None:
        return [0.0] * 10
    total = max(1, stats.get('local', 0) + stats.get('edge', 0) + stats.get('cloud', 0))
    route_ratio_local = stats.get('local', 0) / total
    route_ratio_edge = stats.get('edge', 0) / total
    route_ratio_cloud = stats.get('cloud', 0) / total
    recent = stats.get('recent', [])
    if len(recent) == 0:
        recent_ratio_local = 0.0
        recent_ratio_edge = 0.0
        recent_ratio_cloud = 0.0
    else:
        rtot = len(recent)
        recent_ratio_local = sum((1 for a in recent if a == 0)) / rtot
        recent_ratio_cloud = sum((1 for a in recent if a == 1)) / rtot
        recent_ratio_edge = sum((1 for a in recent if a >= 2)) / rtot
    edge_pressures = []
    if hasattr(ts, 'core_remaining_work'):
        for eid in range(min(len(ts.core_remaining_work), para['edge_num'])):
            try:
                edge_pressures.append(len(ts.core_remaining_work[eid]))
            except Exception:
                edge_pressures.append(0.0)
    avg_edge_queue_pressure = float(np.mean(edge_pressures)) / 10.0 if edge_pressures else 0.0
    cloud_pressure_proxy = route_ratio_cloud
    ready_local_feasible_ratio = max(0.0, 1.0 - route_ratio_cloud)
    ready_edge_feasible_ratio = max(0.0, 1.0 - route_ratio_local)
    return [route_ratio_local, route_ratio_edge, route_ratio_cloud, recent_ratio_local, recent_ratio_edge, recent_ratio_cloud, avg_edge_queue_pressure, cloud_pressure_proxy, ready_local_feasible_ratio, ready_edge_feasible_ratio]

def _node_features(ts, uid, sid, now, task_complex_index):
    task_complex_val = para['task_complex']
    if isinstance(task_complex_val, (list, tuple)):
        safe_idx = max(0, min(int(task_complex_index), len(task_complex_val) - 1))
        comp_val = float(task_complex_val[safe_idx])
    else:
        safe_idx = 0
        comp_val = float(task_complex_val)
    task_size_bytes = float(ts.get_task_size_bytes(uid, sid))
    task_size = task_size_bytes
    task_complex_index = safe_idx
    release = float(ts.enter_time[uid])
    deadline_abs = (release if np.isfinite(release) else now) + ts.get_app_deadline_slot(uid) * para['slot_interval']
    time_left = max(0, deadline_abs - now)
    local_queue = max(0, float(ts.devices_exe_useful[uid]) - now)
    upload_queue = max(0, float(ts.devices_upload_useful[uid]) - now)
    costs = _estimate_placement_costs(ts, uid, task_size_bytes, task_complex_index, now)
    size_norm = task_size_bytes / 1000000.0
    comp_norm = comp_val / 100.0
    time_left_norm = time_left / 10.0
    local_queue_norm = local_queue / 5.0
    upload_queue_norm = upload_queue / 5.0
    est_local_norm = costs['est_local_delay'] / 5.0
    est_cloud_norm = costs['est_cloud_delay'] / 5.0
    best_edge_delay_norm = min(costs['best_edge_delay'], 10.0) / 5.0
    edge_cloud_gap = (costs['best_edge_delay'] - costs['est_cloud_delay']) / 5.0
    edge_local_gap = (costs['best_edge_delay'] - costs['est_local_delay']) / 5.0
    cloud_local_gap = (costs['est_cloud_delay'] - costs['est_local_delay']) / 5.0
    best_edge_queue = costs['best_edge_queue']
    best_edge_stor = costs['best_edge_storage_frac']
    reachable_count = costs['reachable_edge_count'] / max(1, costs['edge_num'])
    best_edge_idx = costs['best_edge_eid'] / max(1, costs['edge_num'] - 1) if costs['best_edge_eid'] >= 0 else 0.5
    local_power_norm = costs['local_power'] / 1000000000.0
    route_struct_feat = _get_route_structure_features(ts, uid, now)
    base_feat = [size_norm, comp_norm, time_left_norm, local_queue_norm, upload_queue_norm, est_local_norm, est_cloud_norm, best_edge_delay_norm, edge_cloud_gap, edge_local_gap, best_edge_queue, best_edge_stor, reachable_count, best_edge_idx, local_power_norm, cloud_local_gap] + route_struct_feat
    return base_feat

def encode_application_state(ts, task_tuple, slot=None, now_time=None, task_complex_index=0):
    uid, sid = task_tuple
    now = float(now_time) if now_time is not None else float(slot or 0) * para['slot_interval']
    g = ts.subgraph_list[uid].nx_graph
    nodes = list(g.nodes())
    mapping = {node: i for i, node in enumerate(nodes)}
    rows = [_node_features(ts, uid, node, now, task_complex_index) + [float(node == sid)] for node in nodes]
    edges = [(mapping[a], mapping[b]) for a, b in g.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    mask = ts.get_action_mask(uid, ts.get_task_size_bytes(uid, sid), now)
    return (Data(x=torch.tensor(rows, dtype=torch.float32), edge_index=edge_index), mask.tolist())

def encode_dag_gat_ppo_state(ts, task_tuple, slot=None, now_time=None, task_complex_index=0):
    """Equation (19): disjoint union of every arrived application's DAG."""
    uid, sid = task_tuple
    now = float(now_time) if now_time is not None else float(slot or 0) * para['slot_interval']
    if not np.isfinite(ts.enter_time[uid]):
        return encode_application_state(ts, task_tuple, slot, now_time, task_complex_index)
    features, edges, offset = ([], [], 0)
    for app in range(ts.user_num):
        if not np.isfinite(ts.enter_time[app]):
            continue
        graph = ts.subgraph_list[app].nx_graph
        nodes = list(graph.nodes())
        mapping = {node: offset + i for i, node in enumerate(nodes)}
        for node in nodes:
            features.append(_node_features(ts, app, node, now, task_complex_index) + [float((app, node) == tuple(task_tuple))])
        edges.extend(((mapping[a], mapping[b]) for a, b in graph.edges()))
        offset += len(nodes)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    mask = ts.get_action_mask(uid, ts.get_task_size_bytes(uid, sid), now)
    return (Data(x=torch.tensor(features, dtype=torch.float32), edge_index=edge_index), mask.tolist())
