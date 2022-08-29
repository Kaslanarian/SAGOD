import torch
from torch_geometric.datasets import Flickr, TUDataset, AttributedGraphDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from util import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

torch.manual_seed(42)
np.random.seed(42)

dataset = TUDataset('./MUTAG', 'MUTAG')
data: Data = list(DataLoader(dataset, batch_size=188, shuffle=True))[0]
data.y = torch.zeros(data.num_nodes)
data.x[data.x == 0.] = 1e-5
data = struct_ano_injection(data, 10, 10)
data = attr_ano_injection(data, 100, 50)

from dominant import DOMINANT
from anomalous import ANOMALOUS
from anomalydae import AnomalyDAE
from ocgnn import OCGNN
from gcnae import GCNAE
from one import ONE
from done import DONE
from adone import AdONE
from alarm import ALARM

verbose = True

sns.set()

model = DOMINANT(verbose=verbose, epoch=100, lr=0.01).fit(data, data.y)
score = model.decision_function(data)
plt.plot(*roc_curve(data.y.numpy(), score)[:2], label='DOMINANT')

model = ANOMALOUS(verbose=verbose, epoch=100, phi=1, lr=0.1).fit(data, data.y)
score = model.decision_function(data)
plt.plot(*roc_curve(data.y.numpy(), score)[:2], label='ANOMALOUS')

model = AnomalyDAE(verbose=verbose, epoch=100).fit(data, data.y)
score = model.decision_function(data)
plt.plot(*roc_curve(data.y.numpy(), score)[:2], label='AnomalyDAE')

model = OCGNN(verbose=verbose, epoch=100).fit(data, data.y)
score = model.decision_function(data)
plt.plot(*roc_curve(data.y.numpy(), score)[:2], label='OCGNN')

model = GCNAE(verbose=verbose, epoch=100).fit(data, data.y)
score = model.decision_function(data)
plt.plot(*roc_curve(data.y.numpy(), score)[:2], label='GCNAE')

model = ONE(5, verbose=verbose, iter=10).fit(data, data.y)
score = model.decision_function(data)
plt.plot(*roc_curve(data.y.numpy(), score)[:2], label='ONE')

model = DONE(6,
             verbose=True,
             n_hidden=[64, 16],
             epoch=10,
             lr=0.005,
             n_layers=6).fit(data, data.y)
score = model.decision_function(data)
plt.plot(*roc_curve(data.y.numpy(), score)[:2], label='DONE')

model = AdONE(4,
              verbose=True,
              n_hidden=[64, 16],
              epoch=10,
              lr=0.005,
              n_layers=6).fit(data, data.y)
score = model.decision_function(data)
plt.plot(*roc_curve(data.y.numpy(), score)[:2], label='AdONE')

model = ALARM([2, 2, 3], verbose=True).fit(data, data.y)
score = model.decision_function(data)
plt.plot(*roc_curve(data.y.numpy(), score)[:2], label='ALARM')

plt.legend()
plt.show()
