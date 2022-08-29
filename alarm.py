import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import Data

from sklearn.base import OutlierMixin
from sklearn.metrics import roc_auc_score
from typing import List, Literal
from util import predict_by_score


def multilayer_gcn(
    n_layers: int,
    n_input: int,
    n_hidden: int,
    n_output: int,
    act,
):
    assert n_layers >= 1
    if n_layers == 1:
        return Sequential('x, edge_index', [(
            GCNConv(n_input, n_output),
            'x, edge_index -> x',
        ), act()])
    module_list = [(
        GCNConv(n_input, n_output),
        'x, edge_index -> x',
    ), act()]
    for i in range(n_layers - 2):
        module_list.extend([(
            GCNConv(n_hidden, n_hidden),
            'x, edge_index -> x',
        ), act()])
    module_list.extend([(
        GCNConv(n_hidden, n_output),
        'x, edge_index -> x',
    ), act()])
    return Sequential('x, edge_index', module_list)


class ALARM_MODEL(nn.Module):
    def __init__(
        self,
        n_inputs: List[int],
        n_hidden: int,
        n_layers: int,
        act,
        aggregator,
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.n_graph = len(n_inputs)
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.act = act
        self.aggregator = aggregator

        self.gcn_list = [
            multilayer_gcn(n_layers, n_input, n_hidden, n_hidden, act)
            for n_input in n_inputs
        ]
        self.attr_dec = nn.Sequential(
            nn.Linear(
                n_hidden if aggregator == 'weight_sum' else n_hidden *
                self.n_graph,
                sum(n_inputs),
            ),
            nn.ReLU(),
        )
        if aggregator == 'weight_sum':
            self.weight = nn.parameter.Parameter(
                torch.full((self.n_graph, 1, 1), 1. / self.n_graph))
            self.forward = self.weight_sum_forward
        else:
            self.forward = self.concat_forward

    def concat_forward(self, x_list, edge_index):
        Z = torch.concat(
            [
                self.gcn_list[i](x_list[i], edge_index)
                for i in range(self.n_graph)
            ],
            dim=1,
        )
        return Z @ Z.T, self.attr_dec(Z)

    def weight_sum_forward(self, x_list, edge_index):
        output = torch.concat([
            self.gcn_list[i](x_list[i], edge_index).unsqueeze(0)
            for i in range(self.n_graph)
        ])
        Z = (self.weight * output).sum(0) / self.weight.sum()
        return Z @ Z.T, self.attr_dec(Z)


class ALARM(OutlierMixin):
    def __init__(
        self,
        views: List[int],
        n_hidden: int = 32,
        n_layers: int = 3,
        act=nn.ReLU,
        aggregator: Literal['concat', 'weight_sum'] = 'weight_sum',
        epoch: int = 5,
        lr: float = 0.01,
        contamination: float = 0.1,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.views = views
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        assert n_layers >= 1
        self.act = act
        assert aggregator in {'concat', 'weight_sum'}
        self.aggregator = aggregator

        self.epoch = epoch
        self.lr = lr
        self.contamination = contamination
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        A = torch.zeros((G.num_nodes, ) * 2)
        A[G.edge_index[0], G.edge_index[1]] = 1
        self.model = ALARM_MODEL(
            self.views,
            self.n_hidden,
            self.n_layers,
            self.act,
            self.aggregator,
        )

        x_list = torch.split(G.x, self.views, dim=1)
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(1, self.epoch + 1):
            A_hat, X_hat = self.model(x_list, G.edge_index)
            stru_score = (A_hat - A).square().sum(1)
            attr_score = (X_hat - G.x).square().sum(1)
            score = stru_score + attr_score
            loss = score.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    auc = roc_auc_score(y, score.detach().numpy())
                    log += ", AUC={:6f}".format(auc)
                print(log)

        self.score = self.decision_function(G)
        self.prediction = predict_by_score(self.score, self.contamination)
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        A = torch.zeros((G.num_nodes, ) * 2)
        A[G.edge_index[0], G.edge_index[1]] = 1
        x_list = torch.split(G.x, self.views, dim=1)
        A_hat, X_hat = self.model(x_list, G.edge_index)
        stru_score = (A_hat - A).square().sum(1)
        attr_score = (X_hat - G.x).square().sum(1)
        score = stru_score + attr_score

        return score.numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)
