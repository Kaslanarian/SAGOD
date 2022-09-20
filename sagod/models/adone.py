from typing import Union, Tuple, List
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from ..utils import predict_by_score, MLP


class AdONE_MODEL(nn.Module):
    def __init__(
        self,
        n_node: int,
        n_feature: int,
        embed_dim: int,
        n_hidden_stru: int,
        n_hidden_attr: int,
        n_layers: int,
        act,
    ) -> None:
        super().__init__()
        self.n_node = n_node
        self.n_feature = n_feature
        self.embed_dim = embed_dim
        self.n_hidden_stru = n_hidden_stru
        self.n_hidden_attr = n_hidden_attr
        self.n_layers = n_layers
        self.act = act

        n_enc = int(n_layers / 2)
        n_dec = n_layers - n_enc

        self.stru_enc = MLP(n_enc, n_node, n_hidden_stru, embed_dim, act)
        self.attr_enc = MLP(n_enc, n_feature, n_hidden_attr, embed_dim, act)
        self.stru_dec = MLP(n_dec, embed_dim, n_hidden_stru, n_node, act,
                            False)
        self.attr_dec = MLP(n_dec, embed_dim, n_hidden_attr, n_feature, act,
                            False)
        self.discriminator = MLP(2, embed_dim, int(embed_dim / 2), 1, nn.Tanh,
                                 False)

    def forward(self, adj, x):
        h_a = self.stru_enc(adj)
        h_x = self.attr_enc(x)
        recon_A = self.stru_dec(h_a)
        recon_X = self.attr_dec(h_x)
        dis_a = torch.sigmoid(self.discriminator(h_a))
        dis_x = torch.sigmoid(self.discriminator(h_x))
        return h_a, recon_A, dis_a, h_x, recon_X, dis_x


class AdONE(BaseDetector):
    '''
    Interface of "Adversarial Deep Outlier Outlier Aware Network Embedding"(AdONE) model.
    
    Parameters
    ----------
    embed_dim : int, default=16
        Embedding dimension of model.
    n_hidden : Union[List[int], Tuple[int], int], default=32
        Size of hidden layers. `n_hidden` can be list or tuple of `int`, or just `int`, which means all hidden layers has same size.
    n_layers : int, default=4
        Number of network layers, which contains encoder and decoder. So it better be even.
    act : default=nn.LeakyReLU
        Activation function of each layer. Class name should be pass just like the default parameter `nn.LeakyReLU`.
    a1 : float, default=0.2
        The weight of structural proximity loss.
    a2 : float, default=0.2
        The weight of structural homophily loss.
    a3 : float, default=0.2
        The weight of attributed proximity loss.
    a4 : float, default=0.2
        The weight of attributed homophily loss.
    a5 : float, default=0.2
        The weight of discriminator loss.
    lr : float, default=0.005
        The learning rate of optimizer (Adam).
    weight_decay : float, default=0.
        The weight decay parameter of optimizer (Adam).
    epoch : int, default=10
        Training epoches of AdONE.
    verbose : bool, default=False
        Whether to print training log, including training epoch and training loss (and ROC_AUC if pass label when fitting model).
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function.
    '''
    def __init__(
        self,
        embed_dim: int = 16,
        n_hidden: Union[List[int], Tuple[int], int] = 32,
        n_layers: int = 4,
        act=nn.LeakyReLU,
        a1: float = 0.2,
        a2: float = 0.2,
        a3: float = 0.2,
        a4: float = 0.2,
        a5: float = 0.2,
        lr: float = 0.005,
        weight_decay: float = 0.,
        epoch: int = 10,
        verbose: bool = False,
        contamination: float = 0.1,
    ):
        super().__init__(contamination)
        self.embed_dim = embed_dim
        if type(n_hidden) in {tuple, list}:
            self.n_hidden_stru, self.n_hidden_attr = n_hidden
        else:
            self.n_hidden_stru = self.n_hidden_attr = n_hidden
        self.n_layers = n_layers
        self.act = act
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        num_neigh = A.sum(1)

        self.model = AdONE_MODEL(
            G.num_nodes,
            G.num_features,
            self.embed_dim,
            self.n_hidden_stru,
            self.n_hidden_attr,
            self.n_layers,
            self.act,
        )

        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.model.train()
        for epoch in range(1, self.epoch + 1):
            h_a, recon_A, dis_a, h_x, recon_X, dis_x = self.model(A, G.x)
            recon_a_err = (recon_A - A).square().sum(1)
            recon_x_err = (recon_X - G.x).square().sum(1)

            temp = h_a.square().sum(1)
            dist_a = temp.reshape(-1, 1) + temp - 2 * h_a @ h_a.T
            hom_a_error = (A @ dist_a).sum(1) / num_neigh

            temp = h_x.square().sum(1)
            dist_x = temp.reshape(-1, 1) + temp - 2 * h_x @ h_x.T
            hom_x_error = (A @ dist_x).sum(1) / num_neigh

            alg_error = (-torch.log(1 - dis_a) - torch.log(dis_x)).squeeze()

            with torch.no_grad():
                temp = recon_a_err + hom_a_error
                s = temp.sum()
                o1 = temp / s

                temp = recon_x_err + hom_x_error
                o2 = temp / temp.sum()

                o3 = alg_error / alg_error.sum()

            l_12 = -torch.log(o1) @ (self.a1 * recon_a_err +
                                     self.a2 * hom_a_error)
            l_34 = -torch.log(o2) @ (self.a3 * recon_x_err +
                                     self.a4 * hom_x_error)
            l_5 = -torch.log(o3) @ alg_error * self.a5
            l = (l_12 + l_34 + l_5) / G.num_nodes

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, l.item())
                if y is not None:
                    score = (o1 + o2 + o3) / 3
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
        num_neigh = A.sum(1)

        h_a, recon_A, dis_a, h_x, recon_X, dis_x = self.model(A, G.x)
        recon_a_err = (recon_A - A).square().sum(1)
        recon_x_err = (recon_X - G.x).square().sum(1)

        temp = h_a.square().sum(1)
        dist_a = temp.reshape(-1, 1) + temp - 2 * h_a @ h_a.T
        hom_a_error = (A @ dist_a).sum(1) / num_neigh

        temp = h_x.square().sum(1)
        dist_x = temp.reshape(-1, 1) + temp - 2 * h_x @ h_x.T
        hom_x_error = (A @ dist_x).sum(1) / num_neigh
        alg_error = (-torch.log(1 - dis_a) - torch.log(dis_x)).squeeze()

        temp = recon_a_err + hom_a_error
        s = temp.sum()
        o1 = temp / s

        temp = recon_x_err + hom_x_error
        o2 = temp / temp.sum()

        o3 = alg_error / alg_error.sum()
        return ((o1 + o2 + o3) / 3).numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)
