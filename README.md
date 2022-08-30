# SAGOD:Static Attributed Graph Outlier Detection

中文README : [cnREADME.md](./cnREADME.md).

SAGOD (**S**tatic **A**ttributed **G**raph **O**utlier **D**etection) is an implementation of anomaly detection models on static attributed graph. Inspierd by [PyOD](https://github.com/yzhao062/pyod) and [PyGOD](https://github.com/pygod-team/pygod), we designed convenient interface to train model and make prediction. SAGOD support the following models:

- [x] [AdONE](#done) : Adversarial Outlier Aware Network Embedding;
- [x] [ALARM](#alarm) : A deep multi-view framework for anomaly detection;
- [x] [ANOMALOUS](#anomalous) : A Joint Modeling Approach for Anomaly Detection on Attributed Networks;
- [x] [AnomalyDAE](#dae) : Anomaly Detection through a Dual Autoencoder;
- [x] [DOMINANT](#dominant) : Deep Anomaly Detection on Attributed Networks;
- [x] [DONE](#done) : Deep Outlier Aware Network Embedding;
- [x] [GAAN](#gaan) : Generative Adversarial Attributed Network;
- [x] [OCGNN](#ocgnn) : One-Class GNN;
- [x] [ONE](#one) : Outlier Aware Network Embedding;
- [x] [Radar](#radar) : Residual Analysis for Anomaly Detection in Attributed Networks.

We are still updating and adding models. It's worth nothing that the original purpose of SAGOD is to implement anomaly detection models on graph, in order to help researchers who are interested in this area (including me).

## Overview

In `test.py`, we generate anomaly data from MUTAG, and use different models to train it. The ROC curve is shown below:

<div align=center><img src="src/eval.png" alt="eval" width="450"/></div>

## Install

```bash
pip3 install sagod
```

or

```bash
git clone https://github.com/Kaslanarian/SAGOD
cd SAGOD
python3 setup.py install
```

## Example

Here is an example to use SAGOD:

```python
from sagod import DOMINANT
from sagod.utils import struct_ano_injection, attr_ano_injection

data = ... # Graph data, type:torch_geometric.data.Data
data.y = torch.zeros(data.num_nodes)
data = struct_ano_injection(data, 10, 10) # Structrual anomaly injection.
data = attr_ano_injection(data, 100, 50) # Attributed anmaly injection.

model = DOMINANT(verbose=True).fit(data, data.y)
plt.plot(*roc_curve(data.y.numpy(), model.decision_scores_)[:2],
         label='DOMINANT') # 绘制ROC曲线
plt.legend()
plt.show()
```

## Highlight

Though SAGOD is similar to PyGOD, we keep innovating and improving:

- The model "ONE" in PyGOD was implemented based on [authors' responsitory](https://github.com/sambaranban/ONE). We improved it with vectorization, achieving a 100% performance improvement;
- We implemented ALARM, which can detect anomaly in multi-view graph;
- ...

## Future Plan

- Support batch mechanism and huge graph input;
- Support GPU;
- More models implementation;
- Annotation and manual;
- ...

## Reference

- <span id="done">Bandyopadhyay, Sambaran, Saley Vishal Vivek, and M. N. Murty. "Outlier resistant unsupervised deep architectures for attributed network embedding." Proceedings of the 13th international conference on web search and data mining. 2020.</span>
- <span id='one'>Bandyopadhyay, Sambaran, N. Lokesh, and M. Narasimha Murty. "Outlier aware network embedding for attributed networks." Proceedings of the AAAI conference on artificial intelligence. Vol. 33. No. 01. 2019.</span>
- <span id='dae'>Fan, Haoyi, Fengbin Zhang, and Zuoyong Li. "AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.</span>
- <span id='gaan'>Chen, Zhenxing, et al. "Generative adversarial attributed network anomaly detection." Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2020.</span>
- <span id='dominant'>Ding, Kaize, et al. "Deep anomaly detection on attributed networks." Proceedings of the 2019 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2019.</span>
- <span id='radar'>Li, Jundong, et al. "Radar: Residual Analysis for Anomaly Detection in Attributed Networks." IJCAI. 2017.</span>
- <span id='alarm'>Peng, Zhen, et al. "A deep multi-view framework for anomaly detection on attributed networks." IEEE Transactions on Knowledge and Data Engineering (2020).</span>
- <span id='anomalous'>Peng, Zhen, et al. "ANOMALOUS: A Joint Modeling Approach for Anomaly Detection on Attributed Networks." IJCAI. 2018.</span>
- <span id='ocgnn'>Wang, Xuhong, et al. "One-class graph neural networks for anomaly detection in attributed networks." Neural computing and applications 33.18 (2021): 12073-12085.</span>
