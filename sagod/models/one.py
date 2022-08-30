from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

import numpy as np
from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import NMF
from ..utils import predict_by_score


class ONE(BaseDetector):
    def __init__(
        self,
        K: int = 36,
        alpha: float = 1.,
        beta: float = 1.,
        gamma: float = 1.,
        mu: float = 1.,
        iter: int = 10,
        nmf_iter: int = 1000,
        contamination: float = 0.1,
        random_state=None,
        verbose: bool = False,
    ) -> None:
        super().__init__(contamination)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.iter = iter
        self.nmf_iter = nmf_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, G: Data, y=None):
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0].numpy()
        C = G.x.numpy()
        N, D = C.shape
        assert self.K < min(N, D)
        model = NMF(
            n_components=self.K,
            init='random',
            random_state=self.random_state,
            max_iter=self.nmf_iter,
        )
        G, H = model.fit_transform(A), model.components_
        model = NMF(
            n_components=self.K,
            init='random',
            random_state=self.random_state,
            max_iter=self.nmf_iter,
        )
        U, V = model.fit_transform(C), model.components_
        O = np.full((3, N), 1 / N)

        for epoch in range(1, self.iter + 1):
            logO = -np.log(O)
            logOs = np.hsplit(logO.T, 3)
            scale = np.sqrt(logOs[2])
            tilde_G = scale * G
            tilde_U = scale * U
            X, _, Y = np.linalg.svd(tilde_G.T @ tilde_U)
            W = X @ Y

            # update G
            temp = self.alpha * logOs[0] * (H**2).sum(1)
            G_num1 = self.alpha * (A - G @ H) @ (logO[0] * H).T
            G_num2 = temp * G
            G_num3 = self.gamma * logOs[2] * U @ W.T
            G_denom = temp + self.gamma * logOs[2]
            G = (G_num1 + G_num2 + G_num3) / G_denom

            # update H
            temp1, temp2 = (V**2).sum(1), (W**2).sum(0)
            U_numer1 = self.beta * logOs[1] * ((C - U @ V) @ V.T + temp1 * U)
            U_numer2 = self.gamma * logOs[2] * ((G - U @ W.T) @ W + temp2 * U)
            U_denom1 = self.beta * logOs[1] * temp1
            U_denom2 = self.gamma * logOs[2] * temp2
            U = (U_numer1 + U_numer2) / (U_denom1 + U_denom2)

            # Update U and V
            H += logO[1] * G.T @ (A - G @ H) / np.c_[logO[1] @ G**2]
            V += logO[1] * U.T @ (C - U @ V) / np.c_[logO[1] @ U**2]

            # update O
            AGH = np.square(A - G @ H)
            CUV = np.square(C - U @ V)
            GUW = np.square(G - U @ W.T)
            O[0] = AGH.sum(1) / AGH.sum()
            O[1] = CUV.sum(1) / CUV.sum()
            O[2] = GUW.sum(1) / GUW.sum()
            O *= self.mu

            # compute loss
            l1 = np.sum(AGH.T * logO[0])
            l2 = np.sum(CUV.T * logO[1])
            l3 = np.sum(GUW.T * logO[2])
            loss = (self.alpha * l1 + self.beta * l2 +
                    self.gamma * l3) / A.shape[0]

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    score = self.alpha * O[0] + self.beta * O[
                        1] + self.gamma * O[2]
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                print(log)

        self.G, self.H = G, H
        self.U, self.V = U, V
        self.W = W

        self.decision_scores_ = score
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    def decision_function(self, G: Data):
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0].numpy()
        C = G.x.numpy()

        O = np.zeros((3, G.num_nodes))
        AGH = np.square(A - self.G @ self.H)
        CUV = np.square(C - self.U @ self.V)
        GUW = np.square(self.G - self.U @ self.W.T)
        O[0] = AGH.sum(1) / AGH.sum()
        O[1] = CUV.sum(1) / CUV.sum()
        O[2] = GUW.sum(1) / GUW.sum()
        O *= self.mu

        return self.alpha * O[0] + self.beta * O[1] + self.gamma * O[2]

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)
