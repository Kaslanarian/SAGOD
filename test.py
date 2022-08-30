import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from sagod.utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sagod import DOMINANT
from sagod import ANOMALOUS
from sagod import AnomalyDAE
from sagod import OCGNN
from sagod import ONE
from sagod import DONE
from sagod import AdONE
from sagod import ALARM
from sagod import Radar
from sagod import GAAN

try:
    import seaborn as sns
    sns.set()
except:
    pass

torch.manual_seed(42)
np.random.seed(42)

dataset = TUDataset('../data', 'MUTAG')
data: Data = list(DataLoader(dataset, batch_size=188, shuffle=True))[0]
data.y = torch.zeros(data.num_nodes)
data.x[data.x == 0.] = 1e-5
data = struct_ano_injection(data, 10, 10)
data = attr_ano_injection(data, 100, 50)

verbose = True

model = DOMINANT(verbose=verbose, epoch=100, lr=0.01).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2],
         label='DOMINANT')

model = ANOMALOUS(verbose=verbose, epoch=100, phi=1, lr=0.1).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2],
         label='ANOMALOUS')

model = AnomalyDAE(verbose=verbose, epoch=100).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2],
         label='AnomalyDAE')

model = OCGNN(verbose=verbose, epoch=100).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='OCGNN')

model = ONE(5, verbose=verbose, iter=10).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='ONE')

model = DONE(6,
             verbose=True,
             n_hidden=[64, 16],
             epoch=10,
             lr=0.005,
             n_layers=6).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='DONE')
model = AdONE(4,
              verbose=True,
              n_hidden=[64, 16],
              epoch=10,
              lr=0.005,
              n_layers=6).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='AdONE')

model = ALARM([2, 2, 3], verbose=True).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='ALARM')

model = Radar(verbose=verbose, gamma=100., lr=0.01).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='Radar')

model = GAAN(verbose=verbose, n_noise=3, n_enc_layers=3,
             n_gen_layers=3).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2], label='GAAN')

plt.title("ROC curve")
plt.legend()
plt.savefig("src/eval.png", dpi=1000)