import torch
import torch.nn as nn
from torch.linalg import solve
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from ..utils import predict_by_score, l21_norm



class Radar_MODEL:
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
    '''
    Interface of "Residual Anlysis for Anomaly Detection in Attributed Networks"(Radar) model. The original paper provide a well-designed optimization rule and we use it.

    Parameters
    ----------
    alpha : float, default=1.
        The weight to control the row sparsity of W.
    beta : float, default=1.
        The weight to control the row sparsity of R.
    gamma : float, default=1.
        The weight of control correlation between network structure and node attributes.
    epoch : int, default=100
        Training epoches of Radar.
    verbose : bool, default=False
        Whether to print training log, including training epoch and training loss (and ROC_AUC if pass label when fitting model).
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function.
    '''
    def __init__(
        self,
        alpha: float = 1.,
        beta: float = 1.,
        gamma: float = 1.,
        epoch: int = 100,
        verbose: bool = False,
        contamination: float = 0.1,
    ):
        super().__init__(contamination)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        I = torch.eye
        X = G.x
        N = G.num_nodes
        A = to_dense_adj(G.edge_index, max_num_nodes=N)[0]
        L = torch.diag(A.sum(1)) - A
        Dr, Dw = I(N), I(N)
        R = solve(I(N) + self.beta * Dr + self.gamma * L, X)
        XXT = X @ X.T

        for epoch in range(1, self.epoch + 1):
            W = solve(XXT + self.alpha * Dw, XXT - X @ R.T)
            Dw[range(N), range(N)] = 1 / (2 * W.square().sum(1).sqrt() + 1e-8)
            R = solve(I(N) + self.beta * Dr + self.gamma * L, X - W.T @ X)
            Dr[range(N), range(N)] = 1 / (2 * R.square().sum(1).sqrt() + 1e-8)

            l1 = torch.norm(X - W.T @ X - R).square()
            l2 = self.alpha * l21_norm(W)
            l3 = self.beta * l21_norm(R)
            l4 = self.gamma * torch.trace(R.T @ L @ R)
            loss = l1 + l2 + l3 + l4

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    score = R.square().sum(1).sqrt()
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                print(log)

        self.decision_scores_ = R.square().sum(1).sqrt().numpy()
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