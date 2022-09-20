import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from typing import Union, List, Tuple
from ..utils import predict_by_score, GCN


class OCGNN(BaseDetector):
    '''
    Interface of "One-Clas Graph Neural Network"(OCGNN) model.
    
    Parameters
    ----------
    n_hidden : Union[List[int], Tuple[int], int], default=64
        Size of hidden layers. `n_hidden` can be list or tuple of `int`, or just `int`, which means all hidden layers has same size.
    n_layers : int, default=4
        Number of GCN layers.
    act : default=nn.ReLU
        Activation function of each layer. Class name should be pass just like the default parameter `nn.ReLU`.
    beta : float, default=0.1
        The parameter for controling the radius of hyper-sphere.
    phi : int, defaul=10
        OCGNN update the radius and center of hypersphere each phi epoches.
    lr : float, default=0.001
        The learning rate of optimizer (Adam).
    weight_decay : float, default=1e-4.
        The weight decay parameter of optimizer (Adam).
    epoch : int, default=100
        Training epoches of OCGNN.
    verbose : bool, default=False
        Whether to print training log, including training epoch and training loss (and ROC_AUC if pass label when fitting model).
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function.
    '''
    def __init__(
        self,
        n_hidden: Union[List[int], Tuple[int], int] = 64,
        n_layers: int = 4,
        act=nn.ReLU,
        beta: float = 0.1,
        phi: int = 10,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        epoch: int = 100,
        verbose: bool = False,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination)
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.act = act
        self.beta = beta
        self.phi = phi
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        self.model = GCN(
            self.n_layers,
            G.num_features,
            self.n_hidden,
            self.n_hidden,
            self.act,
            last_act=False,
        )
        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        with torch.no_grad():
            r = 0.
            c = self.model(G.x, G.edge_index).mean(0)

        self.model.train()
        for epoch in range(1, self.epoch + 1):
            self.model.train()
            dV = (self.model(G.x, G.edge_index) - c).square().sum(1)
            loss = torch.relu(dV - r**2).mean() / self.beta

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    with torch.no_grad():
                        score = (self.model(G.x, G.edge_index) -
                                 c).square().sum(1)
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                print(log)

            if epoch % self.phi == 0:
                with torch.no_grad():
                    r = torch.quantile(dV, 1 - self.beta).item()
                    c = self.model(G.x, G.edge_index).mean(0)

        with torch.no_grad():
            self.r = torch.quantile(dV, 1 - self.beta).item()
            self.c = self.model(G.x, G.edge_index).mean(0)

        self.decision_scores_ = self.decision_function(G)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        score = (self.model(G.x, G.edge_index) - self.c).square().sum(1)
        return score.numpy() - self.r**2

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)
