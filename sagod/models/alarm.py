import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from typing import List, Literal
from ..utils import predict_by_score, GCN


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
            GCN(n_layers, n_input, n_hidden, n_hidden, act)
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


class ALARM(BaseDetector):
    def __init__(
        self,
        views: List[int],
        gamma: float = 1.,
        n_hidden: int = 32,
        n_layers: int = 3,
        act=nn.ReLU,
        aggregator: Literal['concat', 'weight_sum'] = 'weight_sum',
        epoch: int = 5,
        lr: float = 0.01,
        contamination: float = 0.1,
        verbose: bool = False,
    ) -> None:
        super().__init__(contamination)
        self.views = views
        self.gamma = gamma
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        assert n_layers >= 1
        self.act = act
        assert aggregator in {'concat', 'weight_sum'}
        self.aggregator = aggregator

        self.epoch = epoch
        self.lr = lr
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        self.model = ALARM_MODEL(
            self.views,
            self.n_hidden,
            self.n_layers,
            self.act,
            self.aggregator,
        )
        weight = torch.ones_like(A)
        weight[A == 1] = self.gamma

        x_list = torch.split(G.x, self.views, dim=1)
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        import torch.nn.functional as F

        self.model.train()
        for epoch in range(1, self.epoch + 1):
            A_hat, X_hat = self.model(x_list, G.edge_index)
            stru_loss = F.binary_cross_entropy_with_logits(
                A_hat,
                A,
                reduction='none',
                weight=weight,
            ).sum(1)
            attr_score = (X_hat - G.x).square().sum(1)
            loss = stru_loss.mean() + attr_score.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    with torch.no_grad():
                        stru_score = (A_hat - A).square().sum(1)
                        score = stru_score + attr_score
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                print(log)

        self.decision_scores_ = self.decision_function(G)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        x_list = torch.split(G.x, self.views, dim=1)
        A_hat, X_hat = self.model(x_list, G.edge_index)
        stru_score = (A_hat - A).square().sum(1)
        attr_score = (X_hat - G.x).square().sum(1)
        score = stru_score + attr_score

        return score.numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)
