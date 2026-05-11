from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
import warnings
warnings.filterwarnings('ignore')


class HallucinationProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self._scaler = RobustScaler()
        self._pca = PCA(n_components=100)
        self._threshold = 0.5
        self._model = None

    def fit(self, X, y):
        X_scaled = self._scaler.fit_transform(X)
        X_pca = self._pca.fit_transform(X_scaled)
        
        var = self._pca.explained_variance_ratio_.sum()
        print(f"  PCA: {X.shape[1]} -> {X_pca.shape[1]} (var: {var:.1%})")
        
        self._model = LogisticRegressionCV(
            Cs=20, cv=5, penalty='l2', solver='lbfgs',
            class_weight='balanced', max_iter=3000,
            random_state=42, n_jobs=-1, scoring='f1',
        )
        self._model.fit(X_pca, y)
        print(f"  Train F1: {f1_score(y, self._model.predict(X_pca)):.3f}")
        return self

    def fit_hyperparameters(self, X_val, y_val):
        X_val_scaled = self._scaler.transform(X_val)
        X_val_pca = self._pca.transform(X_val_scaled)
        probs = self._model.predict_proba(X_val_pca)[:, 1]
        
        best_f1 = 0
        best_t = 0.5
        for t in np.arange(0.3, 0.7, 0.01):
            preds = (probs >= t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        self._threshold = best_t
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X):
        X_scaled = self._scaler.transform(X)
        X_pca = self._pca.transform(X_scaled)
        prob_pos = self._model.predict_proba(X_pca)[:, 1]
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
