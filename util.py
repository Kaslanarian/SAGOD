from math import ceil
import numpy as np
import torch
from torch_geometric.data import Data


def struct_ano_injection(G: Data, n: int, m: int):
    '''
    结构异常注入，选择mn个节点，形成n个簇，每个簇有m个节点.
    '''
    ano_index = np.random.choice(G.num_nodes, m * n, replace=False)
    edge_index = G.edge_index.numpy().T.tolist()
    edge_index = [tuple(edge) for edge in edge_index]
    for i in range(n):
        cluster_index = ano_index[m * i:m * (i + 1)]
        for j in range(m):
            for k in range(m):
                new_edge = (cluster_index[j], cluster_index[k])
                if j != k and new_edge not in edge_index:
                    edge_index.append(new_edge)
    G.y[ano_index] = 1
    G.edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)
    return G


def attr_ano_injection(G: Data, p: int, k: int = 50):
    '''
    属性异常注入，选择p个节点为属性异常点，对每个节点随机选择k个点，
    欧氏距离最大的点对应属性进行覆盖.
    '''
    ano_index = np.random.choice(G.num_nodes, p, replace=False)
    G.y[ano_index] = 1
    for i in ano_index:
        rnd_choice = np.random.choice(G.num_nodes, k, replace=False)
        j = torch.argmax((G.x[rnd_choice] - G.x[i]).square().sum(1))
        G.x[i] = G.x[j]
    return G


def predict_by_score(score: np.ndarray, contamination: float):
    pred = np.zeros_like(score)
    pred[score > np.percentile(score, 1 - contamination)] = 1
    return pred
