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
    '''
    Interface of "Semi-supervised Anomaly Detection on Attributed Graphs"(SADAG) model.
    
    Parameters
    ----------
    n_layers : int, default=2
        Number of GCN layers.
    n_hidden : int, default=32,
        Hidden size in each GCN layer.
    bias : bool, default=False
        Whether to add bias to GCN layers.
    act : default=nn.ReLU
        Activation function of each layer. Class name should be pass just like the default parameter `nn.ReLU`.
    reg : float, default=1.
        weight of AUC regularization term.
    lr : float, default=0.01
        The learning rate of optimizer (Adam).
    weight_decay : float, default=0.
        The weight decay parameter of optimizer (Adam).
    epoch : int, default=100
        Training epoches of SADAG.
    verbose : bool, default=False
        Whether to print training log, including training epoch and training loss (and ROC_AUC if pass label when fitting model).
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
    '''
    def __init__(
        self,
        n_layers: int = 2,
        n_hidden: int = 32,
        bias: bool = False,
        act: nn.Module = nn.ReLU,
        reg: float = 1.,
        lr: float = 0.01,
        weight_decay: float = 0.,
        epoch: int = 100,
        verbose: bool = False,
        contamination: float = 0.1,
    ):
        super().__init__(contamination)
        self.reg = reg
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.act = act
        self.bias = bias
        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay
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
            self.n_layers,
            G.num_features,
            self.n_hidden,
            self.n_hidden,
            self.act,
            conv_layer=partial(GCNConv, bias=self.bias),
        )
        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

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
