import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from functools import partial
from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score

from ..utils import predict_by_score, GCN


class SADAG(BaseDetector):
    def __init__(
        self,
        reg: float = 1.,
        num_layers: int = 2,
        n_hidden: int = 32,
        act: nn.Module = nn.ReLU,
        bias: bool = False,
        epoch: int = 100,
        lr: float = 0.01,
        contamination: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__(contamination)
        self.reg = reg
        self.num_layers = num_layers
        self.n_hidden = n_hidden
        self.act = act
        self.bias = bias
        self.epoch = epoch
        self.lr = lr
        self.verbose = verbose

    def fit(self, G: Data, y, mask=None):
        arange = torch.arange(G.num_nodes)
        if mask is None:
            mask = torch.zeros(arange.shape, dtype=bool)
        assert not torch.all(
            mask), "Mask all labels is not allowed in semi-supervised task!"
        A = arange[torch.logical_and(
            G.y == 1,
            torch.logical_not(mask),
        )]
        N = arange[torch.logical_and(
            G.y == 0,
            torch.logical_not(mask),
        )]

        self.model = GCN(
            self.num_layers,
            G.num_features,
            self.n_hidden,
            self.n_hidden,
            self.act,
            conv_layer=partial(GCNConv, bias=self.bias),
        )
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(1, self.epoch + 1):
            H = self.model(G.x, G.edge_index)
            c = H[N].mean(0)
            score = (H - c).square().sum(1)
            score_nor, score_ano = score[N], score[A]
            L_nor = score_nor.mean()
            R_auc = torch.sigmoid(score_ano - score_nor.reshape(-1, 1)).mean()
            loss = L_nor - self.reg * R_auc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                auc = roc_auc_score(y, score.detach())
                log += ", AUC={:6f}".format(auc)
                print(log)

        self.c = c
        self.decision_scores_ = self.decision_function(G)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        return (self.model(G.x, G.edge_index) - self.c).square().sum(1)

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)
