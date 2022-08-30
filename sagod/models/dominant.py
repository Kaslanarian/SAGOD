import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from ..utils import predict_by_score, GCN


class DOMINANT_MODEL(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        num_enc_layer: int,
        num_stru_dec_layer: int,
        num_attr_dec_layer: int,
        act,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.num_enc_layer = num_enc_layer
        self.num_stru_dec_layer = num_stru_dec_layer
        self.num_attr_dec_layer = num_attr_dec_layer
        self.act = act

        self.enc = GCN(num_enc_layer, n_input, n_hidden, n_hidden, act)
        self.stru_dec = GCN(num_stru_dec_layer, n_hidden, n_hidden, n_hidden,
                            act)
        self.attr_dec = GCN(num_attr_dec_layer, n_hidden, n_hidden, n_input,
                            act)

    def forward(self, x, edge_index):
        z = self.enc(x, edge_index)
        stru_z = self.stru_dec(z, edge_index)
        attr_recon = self.attr_dec(z, edge_index)
        stru_recon = torch.mm(stru_z, stru_z.t())
        return stru_recon, attr_recon


class DOMINANT(BaseDetector):
    def __init__(
        self,
        alpha: float = 0.5,
        num_enc_layer: int = 3,
        num_stru_dec_layer: int = 0,
        num_attr_dec_layer: int = 1,
        hidden_size: int = 64,
        act=nn.ReLU,
        epoch: int = 5,
        lr: float = 0.005,
        contamination: float = 0.1,
        verbose: bool = False,
    ) -> None:
        super().__init__(contamination)
        self.num_enc_layer = num_enc_layer
        self.num_stru_dec_layer = num_stru_dec_layer
        self.num_attr_dec_layer = num_attr_dec_layer
        self.hidden_size = hidden_size
        self.act = act
        self.alpha = alpha
        self.epoch = epoch
        self.lr = lr
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        self.model = DOMINANT_MODEL(
            G.num_node_features,
            self.hidden_size,
            self.num_enc_layer,
            self.num_stru_dec_layer,
            self.num_attr_dec_layer,
            self.act,
        )
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        optim = Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(1, self.epoch + 1):
            stru_recon, attr_recon = self.model(G.x, G.edge_index)
            stru_error = torch.square((stru_recon - A))
            attr_error = torch.square((attr_recon - G.x))
            stru_score = stru_error.sum(1)
            attr_score = attr_error.sum(1)
            score = (1 - self.alpha) * stru_score + self.alpha * attr_score
            loss = score.mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

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
        stru_recon, attr_recon = self.model(G.x, G.edge_index)
        stru_score = torch.square(stru_recon - A).sum(1).sqrt()
        attr_score = torch.square(attr_recon - G.x).sum(1).sqrt()
        score = (1 - self.alpha) * stru_score + self.alpha * attr_score
        return score.numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)