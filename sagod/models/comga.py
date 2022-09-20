import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from typing import Union, List, Tuple
from ..utils import predict_by_score, GCN, MLP


class ComGA_MODEL(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        n_layers: int,
        n_hidden: int,
        embed_dim: int,
        act: nn.Module,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.embed_dim = embed_dim
        self.act = act

        self.aeEnc = MLP(
            n_layers,
            num_nodes,
            n_hidden,
            embed_dim,
            act,
        )
        self.aeDec = MLP(
            n_layers,
            embed_dim,
            n_hidden,
            num_nodes,
            act,
        )

        n_per_layer = [num_features] + (
            [n_hidden] * max(n_layers - 1, 0) if type(n_hidden)
            not in {list, tuple} else n_hidden) + [embed_dim, embed_dim]
        self.tGCN = []
        for i in range(len(n_per_layer) - 1):
            s = Sequential('x, edge_index', [
                (
                    GCNConv(n_per_layer[i], n_per_layer[i + 1]),
                    'x, edge_index -> x',
                ),
                act(),
            ])
            self.__setattr__("tGCN_{}".format(i), s)
            self.tGCN.append(s)

        self.attrDec = GCN(
            1,
            embed_dim,
            ...,
            num_features,
            act,
        )

    def forward(self, x, edge_index, B):
        h_list = []
        h = B
        for i in range(self.n_layers):
            h = self.aeEnc[2 * i:2 * i + 2](h)
            h_list.append(h)
        B_hat = self.aeDec(h)

        z = self.tGCN[0](x, edge_index)
        for i in range(1, self.n_layers + 1):
            z = self.tGCN[i](z + h_list[i - 1], edge_index)
        A_hat = z @ z.T
        X_hat = self.attrDec(z, edge_index)

        return A_hat, B_hat, X_hat, h_list[-1], z


class ComGA(BaseDetector):
    '''
    Interface of "Community-aware attributed graph anomaly detection framework"(ComGA) model.

    Parameters
    ----------
    embed_dim : int, default=32
        Embedding dimension of model.
    n_hidden : Union[List[int], Tuple[int], int], default=128
        Size of hidden layers. `n_hidden` can be list or tuple of `int`, or just `int`, which means all hidden layers has same size.
    n_layers : int, default=3
        Number of network layers.
    act : default=nn.ReLU
        Activation function of each layer. Class name should be pass just like the default parameter `nn.ReLU`.
    alpha : float, default=0.5
        The weight of structural anomaly score, 1-alpha is the weight of attributed anomaly score correspondingly.
    lr : float, default=0.005
        The learning rate of optimizer (Adam).
    weight_decay : float, default=0.
        The weight decay parameter of optimizer (Adam).
    epoch : int, default=100
        Training epoches of ComGA.
    verbose : bool, default=False
        Whether to print training log, including training epoch and training loss (and ROC_AUC if pass label when fitting model).
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
    '''
    def __init__(
        self,
        embed_dim: int = 32,
        n_hidden: Union[List[int], Tuple[int], int] = 128,
        n_layers: int = 3,
        act=nn.ReLU,
        alpha: int = 0.5,
        lr: float = 0.005,
        weight_decay: float = 0.,
        epoch: int = 100,
        verbose: bool = False,
        contamination: float = 0.1,
    ):
        super().__init__(contamination)
        self.alpha = alpha
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.embed_dim = embed_dim
        self.act = act
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        self.model = ComGA_MODEL(
            G.num_nodes,
            G.num_features,
            self.n_layers,
            self.n_hidden,
            self.embed_dim,
            self.act,
        )
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        K = A.sum(1)
        B = A - K.reshape(-1, 1) * K / K.sum()
        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        kl_loss = nn.KLDivLoss(reduction='sum')

        for epoch in range(1, self.epoch + 1):
            A_hat, B_hat, X_hat, H, Z = self.model(G.x, G.edge_index, B)
            stru_score = (A - A_hat).square().sum(1)
            attr_score = (G.x - X_hat).square().sum(1)
            score = self.alpha * stru_score + (1 - self.alpha) * attr_score

            L_res = (B - B_hat).square().mean()
            L_gui = kl_loss(
                torch.softmax(H, dim=1),
                torch.softmax(Z, dim=1),
            ) / G.num_nodes
            L_rec = score.mean()
            loss = L_res + L_rec + L_gui

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    auc = roc_auc_score(y, score.detach().numpy())
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
        K = A.sum(1)
        B = A - K.reshape(-1, 1) * K / K.sum()
        A_hat, _, X_hat, _, _ = self.model(G.x, G.edge_index, B)
        stru_score = (A - A_hat).square().sum(1)
        attr_score = (G.x - X_hat).square().sum(1)
        score = self.alpha * stru_score + (1 - self.alpha) * attr_score
        return score.numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)