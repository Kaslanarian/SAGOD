import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from ..utils import predict_by_score, MLP


class ResGCN_MODEL(nn.Module):
    def __init__(
        self,
        gamma: float,
        n_nodes: int,
        n_features: int,
        n_res_layers: int,
        n_rep_layers: int,
        n_dec_layers: int,
        n_res_hidden: int,
        n_rep_hidden: int,
        n_dec_hidden: int,
        embed_dim: int,
        act: nn.Module,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.n_res_layers = n_res_layers
        self.n_rep_layers = n_rep_layers
        self.n_dec_layers = n_dec_layers
        self.n_res_hidden = n_res_hidden
        self.n_rep_hidden = n_rep_hidden
        self.n_dec_hidden = n_dec_hidden
        self.embed_dim = embed_dim
        self.act = act

        self.res_model = MLP(
            n_res_layers,
            n_nodes,
            n_res_hidden,
            n_features,
            act=act,
        )
        self.rep_fc = MLP(
            n_rep_layers - 1,
            n_features,
            n_rep_hidden,
            n_rep_hidden,
            act,
        )

        n_per_layer = [n_features] + (
            [n_rep_hidden] * max(n_rep_layers - 1, 0) if type(n_rep_hidden)
            not in {list, tuple} else n_rep_hidden) + [embed_dim]

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
                        ),  # no normalization in the first layer!
                        'x, edge_index -> x',
                    ),
                    act(),
                ])
            self.__setattr__("rep_gcn_{}".format(i), s)
            self.rep_gcn.append(s)

        self.dec_fc = MLP(
            n_dec_layers,
            embed_dim,
            n_dec_hidden,
            n_features,
            act,
        )

    def forward(self, x, edge_index):
        A = to_dense_adj(edge_index, max_num_nodes=x.shape[0])[0]
        R = self.res_model(A)
        R_l = R
        H = self.rep_gcn[0](x, edge_index)
        for i in range(self.n_rep_layers - 1):
            R_l = self.rep_fc[2 * i:2 * i + 2](R_l)
            H = self.rep_gcn[i + 1](
                H * torch.exp(-self.gamma * R_l),
                edge_index,
            )
        X_hat = self.dec_fc(H)
        A_hat = H @ H.T
        return X_hat, A_hat, R


class ResGCN(BaseDetector):
    '''
    Interface of "attention-based deep residual modeling for anomaly detection on attributed networks"(ResGCN) model.
    
    Parameters
    ----------
    embed_dim : int, default=32
        Embedding dimension of model.
    n_res_layers : int, default=2
        Number of residual layers.
    n_rep_layers : int, default=3
        Number of representative layers.
    n_dec_layers : int, default=2
        Number of decoder layers.
    n_res_hidden : int, default=128
        Hidden size of residual layers.
    n_rep_hidden : int, default=64
        Hidden size of representative layers.
    n_dec_hidden : int, default=64
        Hidden size of decoder layers.
    act : default=nn.ReLU
        Activation function of each layer. Class name should be pass just like the default parameter `nn.ReLU`.
    alpha : float, default=0.5
        The weight of structural anomaly score, 1-alpha is the weight of attributed anomaly score correspondingly.
    gamma : float, default=0.1
        The parameter to control GCN weight.
    lamda : float, default=0.1
        The parameter to control residual weight.
    lr : float, default=0.005
        The learning rate of optimizer (Adam).
    weight_decay : float, default=0.
        The weight decay parameter of optimizer (Adam).
    epoch : int, default=100
        Training epoches of ALARM.
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
        n_res_layers: int = 2,
        n_rep_layers: int = 3,
        n_dec_layers: int = 2,
        n_res_hidden: int = 128,
        n_rep_hidden: int = 64,
        n_dec_hidden: int = 64,
        act=nn.ReLU,
        alpha: float = 0.5,
        gamma: float = 0.1,
        lamda: float = 0.1,
        lr: float = 0.005,
        weight_decay: float = 0.,
        epoch: int = 100,
        verbose: bool = False,
        contamination: float = 0.1,
    ):
        super().__init__(contamination)
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda
        self.n_res_layers = n_res_layers
        self.n_rep_layers = n_rep_layers
        self.n_dec_layers = n_dec_layers
        self.n_res_hidden = n_res_hidden
        self.n_rep_hidden = n_rep_hidden
        self.n_dec_hidden = n_dec_hidden
        self.embed_dim = embed_dim
        self.act = act
        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        self.model = ResGCN_MODEL(
            self.gamma,
            G.num_nodes,
            G.num_features,
            self.n_res_layers,
            self.n_rep_layers,
            self.n_dec_layers,
            self.n_res_hidden,
            self.n_rep_hidden,
            self.n_dec_hidden,
            self.embed_dim,
            self.act,
        )

        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        for epoch in range(1, self.epoch + 1):
            X_hat, A_hat, R = self.model(G.x, G.edge_index)
            Es = (A_hat - A).square().sum()
            Ea = (G.x - X_hat - self.lamda * R).square().sum()

            loss = self.alpha * Es + (1 - self.alpha) * Ea
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
        _, _, R = self.model(G.x, G.edge_index)
        return R.square().sum(1).numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)
