import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from ..utils import predict_by_score, MLP


class GAAN_MODEL(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_noise: int,
        n_hidden: int,
        n_gen_layers: int,
        n_enc_layers: int,
        act,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_noise = n_noise
        self.n_hidden = n_hidden
        self.n_gen_layers = n_gen_layers
        self.n_enc_layers = n_enc_layers
        self.act = act

        self.generator = MLP(n_gen_layers, n_noise, n_hidden, n_input, act)
        self.encoder = MLP(n_gen_layers, n_input, n_hidden, n_hidden, act)

    def forward(self, x, noise):
        x_ = self.generator(noise)

        z = self.encoder(x)
        z_ = self.encoder(x_)

        a = torch.sigmoid(z @ z.T)
        a_ = torch.sigmoid(z_ @ z_.T)

        return x_, a, a_


class GAAN(BaseDetector):
    def __init__(
        self,
        n_noise: int = 16,
        n_hidden: int = 64,
        n_gen_layers: int = 2,
        n_enc_layers: int = 2,
        act=nn.ReLU,
        alpha: float = 0.5,
        contamination: float = 0.1,
        lr: float = 0.001,
        epoch: int = 5,
        verbose: bool = False,
    ):
        super().__init__(contamination)
        self.n_noise = n_noise
        self.n_hidden = n_hidden
        self.n_gen_layers = n_gen_layers
        self.n_enc_layers = n_enc_layers
        self.act = act
        self.alpha = alpha
        self.lr = lr
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        self.model = GAAN_MODEL(
            G.num_features,
            self.n_noise,
            self.n_hidden,
            self.n_gen_layers,
            self.n_enc_layers,
            self.act,
        )

        optimizer1 = Adam(self.model.generator.parameters(), lr=self.lr)
        optimizer2 = Adam(self.model.encoder.parameters(), lr=self.lr)

        for epoch in range(1, self.epoch + 1):
            gaussian_noise = torch.randn(G.num_nodes, self.n_noise)
            x_, a, a_ = self.model(G.x, gaussian_noise)

            temp_ = a_[G.edge_index[0], G.edge_index[1]]
            loss_g = F.binary_cross_entropy(temp_, torch.ones_like(temp_))
            optimizer1.zero_grad()
            loss_g.backward()
            optimizer1.step()

            temp = a[G.edge_index[0], G.edge_index[1]]
            loss_r = F.binary_cross_entropy(temp, torch.ones_like(temp))
            loss_f = F.binary_cross_entropy(
                temp_.detach(),
                torch.ones_like(temp_),
            )
            loss_ed = (loss_r + loss_f) / 2
            optimizer2.zero_grad()
            loss_ed.backward()
            optimizer2.step()

            if self.verbose:
                log = "Epoch {:3d}, lossG={:5.6f}, lossED={:5.6f}".format(
                    epoch,
                    loss_g.item(),
                    loss_ed.item(),
                )
                if y is not None:
                    with torch.no_grad():
                        L_G = (G.x - x_).square().sum(1).sqrt()
                        L_D = (A * F.binary_cross_entropy(
                            a, torch.ones_like(a), reduction='none')).sum(1)
                    score = self.alpha * L_G + (1 - self.alpha) * L_D
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                print(log)

        with torch.no_grad():
            L_G = (G.x - x_).square().sum(1).sqrt()
            L_D = (A * F.binary_cross_entropy(
                a, torch.ones_like(a), reduction='none')).sum(1)
        score = self.alpha * L_G + (1 - self.alpha) * L_D
        self.decision_scores_ = score.numpy()
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def decision_function(self, G: Data = None):
        if G is not None:
            print("GAAN is transductive only!")
        return self.decision_scores_

    def predict(self, G: Data = None):
        if G is not None:
            print("GAAN is transductive only!")
        return self.labels_
