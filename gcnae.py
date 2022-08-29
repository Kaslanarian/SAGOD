import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GATConv, Sequential
from torch_geometric.data import Data

from sklearn.base import OutlierMixin
from sklearn.metrics import roc_auc_score
from util import predict_by_score


class GCNAE(OutlierMixin):
    def __init__(
        self,
        n_hidden=64,
        n_layers=4,
        act=nn.ReLU,
        contamination: float = 0.1,
        lr=0.005,
        epoch=100,
        verbose=False,
    ) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.act = act
        self.contamination = contamination
        self.lr = lr
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        assert self.n_layers >= 2
        module_list = [
            (GATConv(G.num_features, self.n_hidden), 'x, edge_index -> x'),
        ]
        for _ in range(self.n_layers - 2):
            module_list.extend([
                self.act(),
                (GATConv(self.n_hidden, self.n_hidden), 'x, edge_index -> x'),
            ])
        module_list.extend([
            self.act(),
            (GATConv(self.n_hidden, G.num_features), 'x, edge_index -> x'),
        ])
        self.model = Sequential('x, edge_index', module_list)
        optim = Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(1, self.epoch + 1):
            output = self.model(G.x, G.edge_index)
            score = torch.square(output - G.x).sum(1)
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

        self.score = self.decision_function(G)
        self.prediction = predict_by_score(self.score, self.contamination)
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        recon = self.model(G.x, G.edge_index)
        score = torch.square(recon - G.x).sum(1).numpy()
        return score

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)
