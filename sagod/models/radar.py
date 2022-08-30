import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from ..utils import predict_by_score


def l21_norm(x: torch.Tensor):
    return x.square().sum(1).sqrt().sum()


class Radar_MODEL(nn.Module):
    def __init__(
        self,
        n: int,
        d: int,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.W = torch.nn.parameter.Parameter(torch.rand(n, n))
        self.R = torch.nn.parameter.Parameter(torch.rand(n, d))

    def forward(self, x, laplacian):
        l1 = torch.norm(x - self.W.T @ x - self.R).square()
        l2 = self.alpha * l21_norm(self.W)
        l3 = self.beta * l21_norm(self.R)
        l4 = self.gamma * torch.trace(self.R.T @ laplacian @ self.R)
        return (l1 + l2 + l3 + l4) / x.shape[0]


class Radar(BaseDetector):
    def __init__(
        self,
        alpha: float = 1.,
        beta: float = 1.,
        gamma: float = 1.,
        epoch: int = 100,
        lr: float = 0.01,
        contamination: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__(contamination)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epoch = epoch
        self.lr = lr
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        self.model = Radar_MODEL(
            G.num_nodes,
            G.num_features,
            self.alpha,
            self.beta,
            self.gamma,
        )

        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        L = torch.diag(A.sum(1)) - A

        optim = Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(1, self.epoch + 1):
            loss = self.model.forward(G.x, L)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    with torch.no_grad():
                        score = self.model.R.square().sum(1).sqrt()
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                print(log)

        with torch.no_grad():
            self.decision_scores_ = self.model.R.square().sum(1).sqrt().numpy()
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )

        return self

    def decision_function(self, G: Data = None):
        if G is not None:
            print("Radar is transductive only!")
        return self.decision_scores_

    def predict(self, G: Data = None):
        if G is not None:
            print("Radar is transductive only!")
        return self.labels_