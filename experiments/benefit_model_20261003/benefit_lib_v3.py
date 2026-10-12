# -*- coding: utf-8 -*-
"""Locator-free state features (ablation that removes privileged simulator supervision).
Inputs are only the static-camera DINOv2 8x8 pooled grid features and the robot state; PCA is fit unsupervised and uses no simulator object positions or joint ground truth.
  PCA(32) of standardized grid8 + robot state (15) = 47-d."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeaturizerNL:
    def fit(self, G8, seed=0, k=32):
        G8 = np.asarray(G8, np.float32)
        self.sc = StandardScaler().fit(G8)
        self.pca = PCA(k, random_state=seed).fit(self.sc.transform(G8))
        return self

    def transform(self, G8, rob):
        Z = self.pca.transform(self.sc.transform(np.asarray(G8, np.float32)))
        return np.hstack([Z, np.asarray(rob, np.float32)])
