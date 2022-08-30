from typing import Union, Tuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from ..utils import predict_by_score, MLP


class DONE_MODEL(nn.Module):
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
        self.stru_dec = MLP(n_dec, embed_dim, n_hidden_stru, n_node, act)
        self.attr_dec = MLP(n_dec, embed_dim, n_hidden_attr, n_feature, act)

    def forward(self, adj, x):
        h_a = self.stru_enc(adj)
        h_x = self.attr_enc(x)
        recon_A = self.stru_dec(h_a)
        recon_X = self.attr_dec(h_x)
        return h_a, recon_A, h_x, recon_X


class DONE(BaseDetector):
    def __init__(
        self,
        embed_dim: int = 16,
        n_hidden: Union[Tuple[int, int], int] = 32,
        n_layers: int = 4,
        act=nn.LeakyReLU,
        a1: float = 0.2,
        a2: float = 0.2,
        a3: float = 0.2,
        a4: float = 0.2,
        a5: float = 0.2,
        contamination: float = 0.1,
        lr: float = 0.005,
        epoch: int = 10,
        verbose: bool = False,
    ) -> None:
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
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        num_neigh = A.sum(1)

        self.model = DONE_MODEL(
            G.num_nodes,
            G.num_features,
            self.embed_dim,
            self.n_hidden_stru,
            self.n_hidden_attr,
            self.n_layers,
            self.act,
        )

        optimizer = Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(1, self.epoch + 1):
            h_a, recon_A, h_x, recon_X = self.model(A, G.x)
            recon_a_err = (recon_A - A).square().sum(1)
            recon_x_err = (recon_X - G.x).square().sum(1)

            temp = h_a.square().sum(1)
            dist_a = temp.reshape(-1, 1) + temp - 2 * h_a @ h_a.T
            hom_a_error = (A @ dist_a).sum(1) / num_neigh

            temp = h_x.square().sum(1)
            dist_x = temp.reshape(-1, 1) + temp - 2 * h_x @ h_x.T
            hom_x_error = (A @ dist_x).sum(1) / num_neigh
            com_error = (h_a - h_x).square().sum(1)

            with torch.no_grad():
                temp = recon_a_err + hom_a_error
                s = temp.sum()
                o1 = temp / s

                temp = recon_x_err + hom_x_error
                o2 = temp / temp.sum()

                o3 = com_error / com_error.sum()

            l_12 = -torch.log(o1) @ (self.a1 * recon_a_err +
                                     self.a2 * hom_a_error)
            l_34 = -torch.log(o2) @ (self.a3 * recon_x_err +
                                     self.a4 * hom_x_error)
            l_5 = -torch.log(o3) @ com_error * self.a5
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

        h_a, recon_A, h_x, recon_X = self.model(A, G.x)
        recon_a_err = (recon_A - A).square().sum(1)
        recon_x_err = (recon_X - G.x).square().sum(1)

        temp = h_a.square().sum(1)
        dist_a = temp.reshape(-1, 1) + temp - 2 * h_a @ h_a.T
        hom_a_error = (A @ dist_a).sum(1) / num_neigh

        temp = h_x.square().sum(1)
        dist_x = temp.reshape(-1, 1) + temp - 2 * h_x @ h_x.T
        hom_x_error = (A @ dist_x).sum(1) / num_neigh
        com_error = (h_a - h_x).square().sum(1)

        temp = recon_a_err + hom_a_error
        s = temp.sum()
        o1 = temp / s

        temp = recon_x_err + hom_x_error
        o2 = temp / temp.sum()

        o3 = com_error / com_error.sum()
        return (o1 + o2 + o3).numpy() / 3

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)
