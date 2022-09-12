import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from ..utils import predict_by_score


class ANOMALOUS_MODEL(nn.Module):
    def __init__(
        self,
        n: int,
        d: int,
        alpha: float = 1.,
        beta: float = 1.,
        gamma: float = 1.,
        phi: float = 1.,
    ) -> None:
        super().__init__()
        self.n = n
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi
        self.W = torch.nn.parameter.Parameter(torch.rand((n, d)))
        self.R = torch.nn.parameter.Parameter(torch.rand((d, n)))

    def forward(self, x, laplacian):
        term1 = (x - x @ self.W @ x - self.R).square().sum()
        term2 = self.alpha * self.W.square().sum(1).sqrt().sum()
        term3 = self.beta * self.W.square().sum(0).sqrt().sum()
        term4 = self.gamma * self.R.square().sum(0).sqrt().sum()
        term5 = self.phi * torch.trace(self.R @ laplacian @ self.R.T)
        return (term1 + term2 + term3 + term4 + term5) / self.n


class ANOMALOUS(BaseDetector):
    def __init__(
        self,
        alpha: float = 1.,
        beta: float = 1.,
        gamma: float = 1.,
        phi: float = 1.,
        epoch: int = 100,
        lr: float = 0.01,
        contamination: float = 0.1,
        verbose: bool = False,
    ) -> None:
        super().__init__(contamination)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi
        self.epoch = epoch
        self.lr = lr
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        self.model = ANOMALOUS_MODEL(
            G.num_nodes,
            G.num_node_features,
            self.alpha,
            self.beta,
            self.gamma,
            self.phi,
        )

        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        L = torch.diag(A.sum(1)) - A

        optim = Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(1, self.epoch + 1):
            loss = self.model.forward(G.x.T, L)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    with torch.no_grad():
                        score = self.model.R.square().sum(0).sqrt()
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                print(log)

        with torch.no_grad():
            self.decision_scores_ = self.model.R.square().sum(0).sqrt().numpy()
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )

        return self

    def decision_function(self, G: Data = None):
        if G is not None:
            print("ANOMALOUS is transductive only!")
        return self.decision_scores_

    def predict(self, G: Data = None):
        if G is not None:
            print("ANOMALOUS is transductive only!")
        return self.labels_
