import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import Data

from sklearn.base import OutlierMixin
from sklearn.metrics import roc_auc_score
from util import predict_by_score


class OCGNN(OutlierMixin):
    def __init__(
        self,
        beta: float = 0.1,
        phi: int = 10,
        n_hidden: int = 64,
        n_layers: int = 4,
        act=nn.ReLU,
        epoch: int = 100,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        contamination: float = 0.1,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.phi = phi
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.act = act
        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.contamination = contamination
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        module_list = [(
            GCNConv(G.num_features, self.n_hidden),
            'x, edge_index -> x',
        )]
        for _ in range(self.n_layers - 1):
            module_list.extend([
                self.act(),
                (GCNConv(self.n_hidden, self.n_hidden), 'x, edge_index -> x'),
            ])
        self.model = Sequential('x, edge_index', module_list)
        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        with torch.no_grad():
            r = 0.
            c = self.model(G.x, G.edge_index).mean(0)

        for epoch in range(1, self.epoch + 1):
            self.model.train()
            dV = (self.model(G.x, G.edge_index) - c).square().sum(1)
            loss = torch.relu(dV - r**2).mean() / self.beta

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    with torch.no_grad():
                        score = (self.model(G.x, G.edge_index) -
                                 c).square().sum(1)
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                print(log)

            if epoch % self.phi == 0:
                with torch.no_grad():
                    r = torch.quantile(dV, 1 - self.beta).item()
                    c = self.model(G.x, G.edge_index).mean(0)

        with torch.no_grad():
            self.r = torch.quantile(dV, 1 - self.beta).item()
            self.c = self.model(G.x, G.edge_index).mean(0)

        self.score = self.decision_function(G)
        self.prediction = predict_by_score(self.score, self.contamination)
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        score = (self.model(G.x, G.edge_index) - self.c).square().sum(1)
        return score.numpy() - self.r**2

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)
