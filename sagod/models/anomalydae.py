import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GATConv, Sequential
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from ..utils import predict_by_score, GCN, MLP


class AnomalyDAE_MODEL(nn.Module):
    def __init__(
        self,
        node_num: int,
        n_dim: int,
        n_hidden: int,
        embed_dim: int,
        act,
    ) -> None:
        super().__init__()
        self.node_num = node_num
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.embed_dim = embed_dim
        self.act = act

        self.stru_enc_fc = MLP(1, n_dim, None, n_hidden, act)
        self.stru_enc_gat = GCN(1,
                                n_hidden,
                                None,
                                embed_dim,
                                act,
                                conv_layer=GATConv)
        self.attr_enc = MLP(2, node_num, n_hidden, embed_dim, act, False)

    def forward(self, x, edge_index):
        Zv = self.stru_enc_gat(self.stru_enc_fc(x), edge_index)
        Za = self.attr_enc(x.T)
        return torch.mm(Zv, Zv.T), torch.mm(Zv, Za.T)


class AnomalyDAE(BaseDetector):
    def __init__(
        self,
        alpha: float = 0.5,
        theta: float = 1.1,
        eta: float = 1.1,
        hidden_size: int = 64,
        embed_dim: int = 8,
        act=nn.ReLU,
        epoch: int = 100,
        lr: float = 0.005,
        contamination: float = 0.1,
        verbose: bool = False,
    ) -> None:
        super().__init__(contamination)
        self.alpha = alpha
        self.theta = theta
        self.eta = eta
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.act = act
        self.epoch = epoch
        self.lr = lr
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        self.model = AnomalyDAE_MODEL(
            G.num_nodes,
            G.num_node_features,
            self.hidden_size,
            self.embed_dim,
            self.act,
        )

        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        Theta = torch.full_like(A, self.theta)
        Theta[G.edge_index[0], G.edge_index[1]] = 1.
        Eta = torch.ones_like(G.x)
        Eta[G.x != 0] = self.eta

        optim = Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(1, self.epoch + 1):
            stru_recon, attr_recon = self.model(G.x, G.edge_index)
            stru_error = torch.square((stru_recon - A) * Theta)
            attr_error = torch.square((attr_recon - G.x) * Eta)
            stru_score = stru_error.sum(1)
            attr_score = attr_error.sum(1)
            score = self.alpha * stru_score + (1 - self.alpha) * attr_score
            loss = score.mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    auc = roc_auc_score(y, score.detach())
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
        Theta = torch.full_like(A, self.theta)
        Theta[G.edge_index[0], G.edge_index[1]] = 1.
        Eta = torch.ones_like(G.x)
        Eta[G.x != 0] = self.eta

        stru_recon, attr_recon = self.model(G.x, G.edge_index)
        stru_score = torch.square((stru_recon - A) * Theta).sum(1)
        attr_score = torch.square((attr_recon - G.x) * Eta).sum(1)
        score = self.alpha * stru_score + (1 - self.alpha) * attr_score
        return score.numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)