import random
import networkx as nx

class BaseGraph:

    def __init__(self, num):
        self.num = num
        self.nx_graph = self.parse_dag()
        self.enter_nodes = [n for n, d in self.nx_graph.in_degree() if d == 0]
        self.exit_nodes = [n for n, d in self.nx_graph.out_degree() if d == 0]
        self.reachable_nodes = self.get_reachable_nodes()

    def get_reachable_nodes(self):
        g = self.nx_graph
        dist_to_exit = {}
        topo = list(nx.topological_sort(g))
        for v in reversed(topo):
            succs = list(g.successors(v))
            if not succs:
                dist_to_exit[v] = 0
            else:
                dist_to_exit[v] = 1 + max((dist_to_exit.get(succ, 0) for succ in succs))
        for exit_node in self.exit_nodes:
            if exit_node in dist_to_exit:
                del dist_to_exit[exit_node]
        return dist_to_exit

    def _generate_random_dag(self, num_nodes):
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        edge_count = int(num_nodes * 1.5)
        for _ in range(edge_count):
            source = random.randint(0, num_nodes - 2)
            dest = random.randint(source + 1, num_nodes - 1)
            if not G.has_edge(source, dest):
                G.add_edge(source, dest)
        if not nx.is_weakly_connected(G):
            components = list(nx.weakly_connected_components(G))
            for i in range(len(components) - 1):
                nodes1 = sorted(list(components[i]))
                nodes2 = sorted(list(components[i + 1]))
                if nodes1 and nodes2:
                    G.add_edge(min(nodes1), min(nodes2))
        return G

    def parse_dag(self):
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'matrix', f'matrix_{self.num}.txt')
        _override = os.environ.get('MATRIX_OVERRIDE_PATH')
        if _override and os.path.exists(_override):
            path = _override
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, encoding='utf-8') as file:
            lines = file.readlines()
        edges = []
        for line in lines:
            if '->' in line:
                source, dest = line.split('[size =')[0].split(' -> ')
                edges.append((int(source) - 1, int(dest) - 1))
        G = nx.DiGraph(edges)
        return G

class SubGraph:

    def __init__(self, num: int, basegraph: BaseGraph):
        self.num = num
        self.nx_graph = self.generate_subgraph(basegraph)
        self.enter_nodes = [n for n, d in self.nx_graph.in_degree() if d == 0]
        self.exit_nodes = [n for n, d in self.nx_graph.out_degree() if d == 0]

    def generate_subgraph_old(self, basegraph: BaseGraph):
        G = basegraph.nx_graph
        subgraph_nodes = set()
        while len(subgraph_nodes) < self.num:
            node = random.choice(list(G.nodes))
            subgraph_nodes.add(node)
            reachable_nodes = set(nx.descendants(G, node))
            for reachable_node in reachable_nodes:
                if len(subgraph_nodes) < self.num:
                    subgraph_nodes.add(reachable_node)
                else:
                    break
        S = basegraph.nx_graph.subgraph(subgraph_nodes)
        return S

    def generate_subgraph(self, basegraph: BaseGraph):
        G = basegraph.nx_graph
        subgraph_nodes = set()
        distances = basegraph.reachable_nodes
        sorted_nodes = sorted(distances.items(), key=lambda x: x[1], reverse=True)
        top_n = len(G.nodes) // 6
        farthest_nodes = [node for node, distance in sorted_nodes[:top_n]]
        initial_node = random.choice(farthest_nodes)
        new_nodes = {initial_node}
        new_neighbors = set()
        while len(subgraph_nodes) < self.num and new_nodes:
            if new_neighbors:
                node = random.choice(list(new_neighbors))
            else:
                node = random.choice(list(new_nodes))
            new_nodes.remove(node)
            subgraph_nodes.add(node)
            neighbors = list(G.neighbors(node))
            new_neighbors = set()
            for neighbor in neighbors:
                if neighbor not in subgraph_nodes:
                    new_nodes.add(neighbor)
                    new_neighbors.add(neighbor)
        S = G.subgraph(subgraph_nodes)
        return S
