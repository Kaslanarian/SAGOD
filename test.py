import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
import numpy as np
from sagod.utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sagod.models import (
    DOMINANT,
    ANOMALOUS,
    AnomalyDAE,
    OCGNN,
    ONE,
    DONE,
    AdONE,
    ALARM,
    Radar,
    GAAN,
    SADAG,
    DeepAE,
    ComGA,
    ResGCN,
)

torch.manual_seed(42)
np.random.seed(42)

dataset = TUDataset('../data', 'MUTAG')
data: Data = list(DataLoader(dataset, batch_size=188, shuffle=True))[0]
data.edge_index = add_self_loops(data.edge_index)[0]
data.y = torch.zeros(data.num_nodes)
data.x[data.x == 0.] = 1e-5
data = struct_ano_injection(data, 10, 15)
data = attr_ano_injection(data, 150, 25)

verbose = True

plt.suptitle("ROC curve")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

print("DOMINANT:")
model = DOMINANT(verbose=verbose, epoch=100, lr=0.01).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2],
         label='DOMINANT')

print("ANOMALOUS:")
model = ANOMALOUS(verbose=verbose, epoch=500, phi=100.,
                  lr=0.01).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2],
         label='ANOMALOUS')

print("AnomalyDAE:")
model = AnomalyDAE(verbose=verbose, epoch=100).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2],
         label='AnomalyDAE')

print("OCGNN:")
model = OCGNN(verbose=verbose, epoch=100).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='OCGNN')

print("ONE:")
model = ONE(5, verbose=verbose, epoch=3).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='ONE')

print("DONE:")
model = DONE(6,
             verbose=True,
             n_hidden=[64, 16],
             epoch=10,
             lr=0.005,
             n_layers=6).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='DONE')

print("AdONE:")
model = AdONE(4,
              verbose=True,
              n_hidden=[64, 16],
              epoch=10,
              lr=0.005,
              n_layers=6).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='AdONE')

print("ALARM:")
model = ALARM([2, 2, 3], verbose=True).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='ALARM')

plt.legend()
plt.subplot(1, 2, 2)

print("GAAN:")
model = GAAN(verbose=verbose, n_noise=3, n_enc_layers=3,
             n_gen_layers=3).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='GAAN')

print("Radar:")
model = Radar(verbose=verbose, alpha=100., beta=100., gamma=100.,
              epoch=5).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='Radar')

print("SADAG:")
model = SADAG(verbose=True, reg=1., lr=0.005, epoch=20).fit(
    data,
    data.y,
    mask=torch.bernoulli(torch.full_like(data.y, 0.8)),
)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='SADAG')

print("DeepAE:")
model = DeepAE(verbose=True, lr=0.005, epoch=50).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2],
         label='DeepAE')

print("ComGA:")
model = ComGA(verbose=True, lr=0.01, epoch=100, embed_dim=64).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='ComGA')

print("ResGCN:")
model = ResGCN(verbose=True, lr=0.01, epoch=50).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2],
         label='ResGCN')

plt.legend()
plt.savefig("src/eval.png", dpi=1000)
