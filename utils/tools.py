import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def plot_graph(g):
    subax1 = plt.subplot(111)
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()

def averange(l):
    return sum(l) / len(l)

def normalize_list(input_list):
    return mean_normalize(input_list)

def normalize_dict_values(input_dict):
    min_val = min(input_dict.values())
    max_val = max(input_dict.values())
    if min_val == max_val:
        return {key: 0.5 for key in input_dict}
    normalized_dict = {key: (value - min_val) / (max_val - min_val) for key, value in input_dict.items()}
    return normalized_dict

def minmax_normalize(input_list):
    min_val = min(input_list)
    max_val = max(input_list)
    if min_val == max_val:
        return input_list
    normalized_list = [(x - min_val) / (max_val - min_val) for x in input_list]
    return normalized_list

def mean_normalize(input_list):
    mean_value = sum(input_list) / len(input_list)
    normalized_list = [x - mean_value for x in input_list]
    return normalized_list

def m_normalize(data):
    data = [data]
    scaler = MinMaxScaler()
    scaler.fit(data)
    normalized_data = scaler.transform(data)
    return normalized_data[0]
