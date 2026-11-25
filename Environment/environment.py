import math
import random
from Environment.components import *
from utils.constant import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from utils.tools import *
from Environment.Graph import BaseGraph, SubGraph

class Environment:

    def __init__(self, user_num=20, subgraph_num=10, basegraph_num=30, task_complex_index=3, fe=para['edge_power']):
        self.task_complex_index = task_complex_index
        self.cloud = Cloud(task_complex_index=task_complex_index)
        self.basegraph = BaseGraph(basegraph_num)
        edges_num = 4
        self.edges = []
        edge_order = 0
        for i, position in enumerate(para['edge_positions']):
            x, y = position
            if isinstance(para['edge_power'], list):
                power = para['edge_power'][i % len(para['edge_power'])]
            else:
                power = para['edge_power']
            self.edges.append(EdgeServer(edge_order, x, y, power, task_complex_index=task_complex_index))
            edge_order += 1
        self.device_list = []
        self.subgraph_list = []
        self.task_list = random.choices(para['task_size_range'], k=user_num)
        self.user_num = user_num
        self.subtask_num = subgraph_num
        self.radius = para['edge_radius']

    def generate_components(self, seed=None):
        self.device_list = []
        self.subgraph_list = []
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            if 'local_power_range' in para:
                z = rng.uniform(0.0, 1.0, size=self.user_num)

                def lerp(a, b, z):
                    return a + (b - a) * z
                local_powers = lerp(200000000.0, 600000000.0, z)
                local_waits = lerp(0.06, 0.18, z)
                local_trans_powers = lerp(1.3, 2.3, z)
                local_powers *= rng.lognormal(mean=0.0, sigma=0.05, size=self.user_num)
                local_waits *= rng.lognormal(mean=0.0, sigma=0.05, size=self.user_num)
                local_trans_powers *= rng.lognormal(mean=0.0, sigma=0.05, size=self.user_num)
                local_powers = np.clip(local_powers, 200000000.0, 600000000.0)
                local_waits = np.clip(local_waits, 0.05, 0.2)
                local_trans_powers = np.clip(local_trans_powers, 1.0, 2.8)
            else:
                local_powers = None
                local_waits = None
                local_trans_powers = None
        else:
            rng = None
            local_powers = None
            local_waits = None
            local_trans_powers = None
        cnt = 0
        while cnt < self.user_num:
            temp_x = round(random.normalvariate(250, 200))
            temp_y = round(random.normalvariate(250, 200))
            if temp_x < 0 or temp_x > 500 or temp_y < 0 or (temp_y > 500):
                continue
            in_range = False
            too_close = False
            for edge in self.edges:
                if temp_x == edge.pos_x and temp_y == edge.pos_y:
                    too_close = True
                    break
            if too_close:
                continue
            distances = [float('inf')] * len(self.edges)
            edge_scores = [float('inf')] * len(self.edges)
            for index, edge in enumerate(self.edges):
                temp_dist0 = math.sqrt((temp_x - edge.pos_x) ** 2 + (temp_y - edge.pos_y) ** 2)
                if temp_dist0 < distances[index]:
                    distances[index] = temp_dist0
                if temp_dist0 <= self.radius:
                    in_range = True
            if in_range:
                edge_inrange = []
                for edge_idx in range(len(self.edges)):
                    if distances[edge_idx] <= self.radius:
                        edge_inrange.append(self.edges[edge_idx])
                        edge_scores[edge_idx] = para['edge_load_weights'][0] * distances[edge_idx]
                device_power = local_powers[cnt] if local_powers is not None else None
                device_wait = local_waits[cnt] if local_waits is not None else None
                device_trans = local_trans_powers[cnt] if local_trans_powers is not None else None
                self.device_list.append(Device(cnt, temp_x, temp_y, distances, edge_scores, edge_inrange, self.task_complex_index, local_power=device_power, local_wait=device_wait, local_trans=device_trans))
                self.device_list[-1].local_bw = float(rng.choice(para['uplink_range'])) if rng is not None else float(para['uplink_range'][0])
                self.subgraph_list.append(SubGraph(self.subtask_num, self.basegraph))
                cnt += 1

    def plot_env(self):
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        for edge in self.edges:
            circle = Circle((edge.pos_x, edge.pos_y), radius=self.radius, facecolor='white', edgecolor='cornflowerblue', alpha=0.8, fill=False)
            ax.add_patch(circle)
        ax.set_xlim(-50, 550)
        ax.set_ylim(-50, 550)
        square = Rectangle(xy=(0, 0), width=500, height=500, linewidth=2, edgecolor='r', facecolor='none')
        ax = plt.gca()
        ax.add_patch(square)
        user_position_x = []
        user_position_y = []
        for item in self.device_list:
            user_position_x.append(item.pos_x)
            user_position_y.append(item.pos_y)
        plt.scatter(user_position_x, user_position_y, color='orange')
        for position in para['edge_positions']:
            x, y = position
            plt.scatter(x, y, color='green', marker='^', s=64)
        plt.xlabel('Meters', fontsize=13)
        plt.ylabel('Meters', fontsize=13)
        plt.show()

    def set_fe(self, fe):
        self.edge.edge_power = fe

    def reset_env(self):
        pass

    def calculate_average_distance(self):
        total_distance = 0
        for device in self.device_list:
            min_distance = float('inf')
            for edge in self.edges:
                distance = math.sqrt((device.pos_x - edge.pos_x) ** 2 + (device.pos_y - edge.pos_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
            total_distance += min_distance
        average_distance = total_distance / len(self.device_list)
        return average_distance
