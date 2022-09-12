import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from ..utils import predict_by_score, MLP


class ResGCN_MODEL(nn.Module):
    def __init__(
        self,
        gamma: float,
        num_nodes: int,
        num_features: int,
        num_res_layers: int,
        num_rep_layers: int,
        num_dec_layers: int,
        n_res_hidden: int,
        n_rep_hidden: int,
        n_dec_hidden: int,
        n_embed: int,
        act: nn.Module,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_res_layers = num_res_layers
        self.num_rep_layers = num_rep_layers
        self.num_dec_layers = num_dec_layers
        self.n_res_hidden = n_res_hidden
        self.n_rep_hidden = n_rep_hidden
        self.n_dec_hidden = n_dec_hidden
        self.n_embed = n_embed
        self.act = act

        self.res_model = MLP(
            num_res_layers,
            num_nodes,
            n_res_hidden,
            num_features,
            act=act,
        )
        self.rep_fc = MLP(
            num_rep_layers - 1,
            num_features,
            n_rep_hidden,
            n_rep_hidden,
            act,
        )

        n_per_layer = [num_features] + (
            [n_rep_hidden] * max(num_rep_layers - 1, 0) if type(n_rep_hidden)
            not in {list, tuple} else n_rep_hidden) + [n_embed]

        self.rep_gcn = []
        for i in range(len(n_per_layer) - 1):
            s = Sequential(
                'x, edge_index',
                [
                    (
                        GCNConv(
                            n_per_layer[i],
                            n_per_layer[i + 1],
                            normalize=(i != 0),
                        ),  # 第一层不要normalize
                        'x, edge_index -> x',
                    ),
                    act(),
                ])
            self.__setattr__("rep_gcn_{}".format(i), s)
            self.rep_gcn.append(s)

        self.dec_fc = MLP(
            num_dec_layers,
            n_embed,
            n_dec_hidden,
            num_features,
            act,
        )

    def forward(self, x, edge_index):
        A = to_dense_adj(edge_index, max_num_nodes=x.shape[0])[0]
        R = self.res_model(A)
        R_l = R
        H = self.rep_gcn[0](x, edge_index)
        for i in range(self.num_rep_layers - 1):
            R_l = self.rep_fc[2 * i:2 * i + 2](R_l)
            H = self.rep_gcn[i + 1](
                H * torch.exp(-self.gamma * R_l),
                edge_index,
            )
        X_hat = self.dec_fc(H)
        A_hat = H @ H.T
        return X_hat, A_hat, R


class ResGCN(BaseDetector):
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.1,
        lamda: float = 0.1,
        num_res_layers: int = 2,
        num_rep_layers: int = 3,
        num_dec_layers: int = 2,
        n_res_hidden: int = 128,
        n_rep_hidden: int = 64,
        n_dec_hidden: int = 64,
        n_embed: int = 32,
        act=nn.ReLU,
        epoch: int = 100,
        lr: float = 0.005,
        contamination: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__(contamination)
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda
        self.num_res_layers = num_res_layers
        self.num_rep_layers = num_rep_layers
        self.num_dec_layers = num_dec_layers
        self.n_res_hidden = n_res_hidden
        self.n_rep_hidden = n_rep_hidden
        self.n_dec_hidden = n_dec_hidden
        self.n_embed = n_embed
        self.act = act
        self.epoch = epoch
        self.lr = lr
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        self.model = ResGCN_MODEL(
            self.gamma,
            G.num_nodes,
            G.num_features,
            self.num_res_layers,
            self.num_rep_layers,
            self.num_dec_layers,
            self.n_res_hidden,
            self.n_rep_hidden,
            self.n_dec_hidden,
            self.n_embed,
            self.act,
        )

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(1, self.epoch + 1):
            X_hat, A_hat, R = self.model(G.x, G.edge_index)
            Es = (A_hat - A).square().sum()
            Ea = (G.x - X_hat - self.lamda * R).square().sum()

            loss = (1 - self.alpha) * Es + self.alpha * Ea
            score = R.square().sum(1)

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
        _, _, R = self.model(G.x, G.edge_index)
        return R.square().sum(1).numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)
