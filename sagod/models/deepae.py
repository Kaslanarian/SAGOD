import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.nn.conv.gcn_conv import GCNConv, gcn_norm

from functools import partial
from sklearn.metrics import roc_auc_score
from pyod.models.base import BaseDetector
from typing import Union, List, Tuple
from ..utils import GCN, predict_by_score


class SharpenGCN(GCNConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        coef: float = 1.,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            improved,
            cached,
            add_self_loops,
            normalize,
            bias,
            **kwargs,
        )
        self.coef = coef

    def forward(self, x, edge_index, edge_weight=None):
        if self.normalize:
            if isinstance(edge_index, torch.Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, torch.SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        out = self.propagate(edge_index,
                             x=x,
                             edge_weight=edge_weight,
                             size=None)

        # Sharpen GCNConv
        out = (1 - self.coef) * x + self.coef * out

        if self.bias is not None:
            out += self.bias

        return out


class DeepAE_MODEL(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_input: int,
        n_hidden: int,
        gamma=0.5,
        act=nn.ReLU,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.gamma = gamma
        self.act = act

        self.encoder = GCN(n_layers, n_input, n_hidden, n_hidden, act)
        self.decoder = GCN(
            n_layers,
            n_hidden,
            n_hidden,
            n_input,
            act,
            conv_layer=partial(SharpenGCN, coef=gamma),
        )

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        return h, self.decoder(h, edge_index)


class DeepAE(BaseDetector):
    '''
    Interface of "Deep Graph Autoencoders on Attributed Networks"(DeepAE) model.
    
    Parameters
    ----------
    n_hidden : Union[List[int], Tuple[int], int], default=32
        Size of hidden layers. `n_hidden` can be list or tuple of `int`, or just `int`, which means all hidden layers has same size.
    n_layers : int, default=2
        Number of network layers.
    act : default=nn.ReLU
        Activation function of each layer. Class name should be pass just like the default parameter `nn.ReLU`.
    gamma : float=0.5
        The parameter for sharpen GCN. When `alpha`=1, Sharpen GCM becomes normal GCN.
    lr : float, default=0.005
        The learning rate of optimizer (Adam).
    weight_decay : float, default=0.
        The weight decay parameter of optimizer (Adam).
    epoch : int, default=100
        Training epoches of DeepAE.
    verbose : bool, default=False
        Whether to print training log, including training epoch and training loss (and ROC_AUC if pass label when fitting model).
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
    '''
    def __init__(
        self,
        n_hidden: Union[List[int], Tuple[int], int] = 32,
        n_layers: int = 2,
        act=nn.ReLU,
        beta: float = 1.,
        gamma: float = 0.5,
        lr: float = 0.001,
        weight_decay: float = 0.,
        epoch: int = 100,
        verbose=False,
        contamination=0.1,
    ):
        super().__init__(contamination)
        self.beta = beta
        self.gamma = gamma
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.act = act
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        self.model = DeepAE_MODEL(
            self.n_layers,
            G.num_features,
            self.n_hidden,
            self.gamma,
            self.act,
        )
        X = G.x
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]

        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        for epoch in range(1, self.epoch + 1):
            H, X_hat = self.model(X, G.edge_index)
            A_hat = H @ H.T

            recon_attr = (X_hat - X).square().sum(1)
            recon_stru = (A_hat - A).square().sum(1)
            score = self.beta * recon_attr + recon_stru

            Lh = score.sum()
            L = torch.diag(A_hat.sum(1)) - A_hat
            Lf = 2 * torch.trace(H.T @ L @ H)
            norm_X = X / X.square().sum(1, keepdims=True).sqrt()
            sim = norm_X @ norm_X.T
            sim /= sim.sum()
            Ls = -(sim * torch.log(A_hat + 1e-16)).sum()

            loss = (Lh + Lf + Ls) / G.num_nodes

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
        X = G.x
        A = to_dense_adj(
            G.edge_index,
            max_num_nodes=G.num_nodes,
        )[0]
        H, X_hat = self.model(X, G.edge_index)
        A_hat = H @ H.T

        recon_attr = (X_hat - X).square().sum(1)
        recon_stru = (A_hat - A).square().sum(1)
        score = self.beta * recon_attr + recon_stru
        return score.numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)
