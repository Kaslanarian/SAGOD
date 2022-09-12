import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from ..utils import predict_by_score, GCN, MLP


class ComGA_MODEL(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        n_layers: int,
        n_hidden: int,
        n_embed: int,
        act: nn.Module,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_embed = n_embed
        self.act = act

        self.aeEnc = MLP(
            n_layers,
            num_nodes,
            n_hidden,
            n_embed,
            act,
        )
        self.aeDec = MLP(
            n_layers,
            n_embed,
            n_hidden,
            num_nodes,
            act,
        )

        n_per_layer = [num_features] + ([n_hidden] * max(n_layers - 1, 0)
                                        if type(n_hidden) not in {list, tuple}
                                        else n_hidden) + [n_embed, n_embed]
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
            n_embed,
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
    def __init__(
        self,
        alpha: int = 0.5,
        num_layers: int = 3,
        n_hidden: int = 128,
        n_embed: int = 32,
        act=nn.ReLU,
        epoch: int = 100,
        lr: float = 0.005,
        contamination: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__(contamination)
        self.alpha = alpha
        self.num_layers = num_layers
        self.n_hidden = n_hidden
        self.n_embed = n_embed
        self.act = act
        self.epoch = epoch
        self.lr = lr
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        self.model = ComGA_MODEL(
            G.num_nodes,
            G.num_features,
            self.num_layers,
            self.n_hidden,
            self.n_embed,
            self.act,
        )
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        K = A.sum(1)
        B = A - K.reshape(-1, 1) * K / K.sum()
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        kl_loss = nn.KLDivLoss(reduction='sum')

        for epoch in range(1, self.epoch + 1):
            A_hat, B_hat, X_hat, H, Z = self.model(G.x, G.edge_index, B)
            stru_score = (A - A_hat).square().sum(1)
            attr_score = (G.x - X_hat).square().sum(1)
            score = (1 - self.alpha) * stru_score + self.alpha * attr_score

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
        score = (1 - self.alpha) * stru_score + self.alpha * attr_score
        return score.numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)