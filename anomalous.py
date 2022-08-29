import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data

from sklearn.base import OutlierMixin
from sklearn.metrics import roc_auc_score
from util import predict_by_score


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
        return (term1 + term2 + term3 + term4 - term5) / self.n


class ANOMALOUS(OutlierMixin):
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
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi
        self.epoch = epoch
        self.lr = lr
        self.contamination = contamination
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

        A = torch.zeros((G.num_nodes, G.num_nodes))
        edge_index = G.edge_index
        A[edge_index[0], edge_index[1]] = 1
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

        self.score = self.decision_function()
        self.prediction = predict_by_score(self.score, self.contamination)

        return self

    @torch.no_grad()
    def decision_function(self, G: Data = None):
        if G is not None:
            print("ANOMALOUS is transductive only!")
        return self.model.R.square().sum(0).sqrt()

    def predict(self, G: Data = None):
        if G is not None:
            print("ANOMALOUS is transductive only!")
        return self.prediction
