from typing import List, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Sequential


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
        G.x[i] = G.x[rnd_choice[j]]
    return G


def predict_by_score(
    score: np.ndarray,
    contamination: float,
    return_threshold: bool = False,
):
    pred = np.zeros_like(score)
    threshold = np.percentile(score, 1 - contamination)
    pred[score > threshold] = 1
    if return_threshold:
        return pred, threshold
    return pred


def MLP(
    n_layers: int,
    n_input: int,
    n_hidden: Union[int, List[int], Tuple[int]],
    n_output: int,
    act: nn.modules,
    last_act: bool = True,
):
    if n_layers < 0:
        raise ValueError("Parameter 'n_layers' must be non-negative!")
    elif n_layers == 0:
        return nn.Identity()

    if type(n_hidden) not in {list, tuple}:
        n_hidden = [n_hidden] * max(n_layers - 1, 0)

    n_per_layer = [n_input] + list(n_hidden) + [n_output]
    assert len(n_per_layer) == n_layers + 1
    module_list = []
    for i in range(n_layers):
        module_list.extend([
            nn.Linear(n_per_layer[i], n_per_layer[i + 1]),
            act(),
        ])
    if not last_act:
        module_list.pop()
    return nn.Sequential(*module_list)


def GCN(
    n_layers: int,
    n_input: int,
    n_hidden: Union[int, List[int], Tuple[int]],
    n_output: int,
    act: nn.modules,
    last_act: bool = True,
    conv_layer: nn.Module = GCNConv,
):
    if n_layers < 0:
        raise ValueError("Parameter 'n_layers' must be non-negative!")
    elif n_layers == 0:
        return Sequential('x, edge_index', [(nn.Identity(), 'x -> x')])

    if type(n_hidden) not in {list, tuple}:
        n_hidden = [n_hidden] * max(n_layers - 1, 0)

    n_per_layer = [n_input] + n_hidden + [n_output]
    assert len(n_per_layer) == n_layers + 1
    module_list = []
    for i in range(n_layers):
        module_list.extend([
            (
                conv_layer(n_per_layer[i], n_per_layer[i + 1]),
                'x, edge_index -> x',
            ),
            act(),
        ])

    if not last_act:
        module_list.pop()
    return Sequential('x, edge_index', module_list)
