# -*- coding: utf-8 -*-
"""Feature and model components of the benefit model (picklable with joblib, no closures).

Locator: frozen DINOv2 16x16 patches of the static camera -> per-patch heatmaps of the three blocks (projection supervision) -> expected image coordinates -> world coordinates.
  Same algorithm as fix_20260928/object_locator.py v2; all parameters are converted to numpy after training.
Featurizer: Locator + two ridge regressions (slider and drawer, from 8x8 grid features) + robot state -> 32-d state features.
"""
import numpy as np

OBJECTS = {"red": slice(6, 9), "blue": slice(12, 15), "pink": slice(18, 21)}
LOOK_FROM = [2.871459009488717, -2.166602199425597, 2.555159848480571]
LOOK_AT = [-0.026242351159453392, -0.0302329882979393, 0.3920000493526459]
UP = [0.4041403970338857, 0.22629790978217404, 0.8862616969685161]
FOV, NEAR, FAR, W, H, GRID = 10, 0.01, 10, 200, 200, 16


def _proj_matrix():
    import pybullet as p
    V = np.array(p.computeViewMatrix(LOOK_FROM, LOOK_AT, UP)).reshape(4, 4).T
    P = np.array(p.computeProjectionMatrixFOV(FOV, 1.0, NEAR, FAR)).reshape(4, 4).T
    return P @ V


def project(xyz):
    h = np.c_[xyz, np.ones(len(xyz))] @ _proj_matrix().T
    ndc = h[:, :3] / h[:, 3:4]
    return np.c_[(ndc[:, 0] + 1) / 2 * W, (1 - ndc[:, 1]) / 2 * H]


def _soft_targets(uv, sigma=0.7):
    c = (np.arange(GRID) + 0.5) * (W / GRID)
    cu = (uv[:, 0:1] - c[None]) / (W / GRID); cv = (uv[:, 1:2] - c[None]) / (H / GRID)
    t = np.exp(-(cv[:, :, None] ** 2 + cu[:, None, :] ** 2) / (2 * sigma ** 2)).reshape(len(uv), -1)
    return t / t.sum(1, keepdims=True)


def _basis(uv):
    u, v = uv[:, 0] / W, uv[:, 1] / H
    return np.c_[u, v, u * u, v * v, u * v, np.ones_like(u)]


class Locator:
    def fit(self, patches, scene, epochs=600, lr=1e-2, wd=1e-2, seed=0):
        import torch
        torch.manual_seed(seed)
        X = torch.as_tensor(np.asarray(patches, np.float32))
        self.m = X.mean(dim=(0, 1), keepdim=True).numpy(); self.s = (X.std(dim=(0, 1), keepdim=True) + 1e-6).numpy()
        Xn = (X - torch.as_tensor(self.m)) / torch.as_tensor(self.s)
        names = list(OBJECTS)
        uvs = [project(scene[:, OBJECTS[o]]) for o in names]
        inside = [((u[:, 0] >= 0) & (u[:, 0] < W) & (u[:, 1] >= 0) & (u[:, 1] < H)) for u in uvs]
        T = torch.as_tensor(np.stack([_soft_targets(np.clip(u, 0, W - 1e-3)) for u in uvs], -1), dtype=torch.float32)
        M = torch.as_tensor(np.stack(inside, -1), dtype=torch.float32)
        clf = torch.nn.Linear(X.shape[-1], len(names))
        opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            logp = torch.log_softmax(clf(Xn), dim=1)
            loss = -((T * logp).sum(1) * M).sum() / M.sum()
            loss.backward(); opt.step()
        self.Wt = clf.weight.detach().numpy().T.copy(); self.b = clf.bias.detach().numpy().copy()
        c = (np.arange(GRID) + 0.5) * (W / GRID); cu, cv = np.meshgrid(c, c)
        self.grid_uv = np.c_[cu.ravel(), cv.ravel()]
        prob = self._heat(patches)
        self.maps = []
        for i, o in enumerate(names):
            B = _basis(prob[:, :, i] @ self.grid_uv); Y = scene[:, OBJECTS[o]]
            self.maps.append(np.linalg.lstsq(B.T @ B + 1e-3 * np.eye(6), B.T @ Y, rcond=None)[0])
        return self

    def _heat(self, patches, chunk=512):
        out = []
        for i in range(0, len(patches), chunk):
            x = (np.asarray(patches[i:i + chunk], np.float32) - self.m) / self.s
            z = x @ self.Wt + self.b
            z = z - z.max(1, keepdims=True); e = np.exp(z)
            out.append(e / e.sum(1, keepdims=True))
        return np.concatenate(out)

    def predict(self, patches):
        prob = self._heat(patches)
        uv = np.stack([prob[:, :, i] @ self.grid_uv for i in range(3)], 1)
        xyz = np.stack([_basis(uv[:, i]) @ self.maps[i] for i in range(3)], 1)
        return xyz, prob.max(1)


class Featurizer:
    """State features: estimated positions of the three blocks (9), visibility (3), distance to the end effector (3), slider and drawer estimates (2), robot state (15) = 32-d."""
    def fit(self, G16, G8, scene, seed=0):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV
        self.loc = Locator().fit(G16, scene, seed=seed)
        self.sc = StandardScaler().fit(G8)
        Z = self.sc.transform(G8)
        self.ridges = [RidgeCV(alphas=np.logspace(0, 5, 11)).fit(Z, scene[:, c]) for c in (0, 1)]
        return self

    def transform(self, G16, G8, rob):
        xyz, conf = self.loc.predict(G16)
        Z = self.sc.transform(G8)
        art = np.column_stack([r.predict(Z) for r in self.ridges])
        dist = np.linalg.norm(xyz - rob[:, None, :3], axis=2)
        return np.hstack([xyz.reshape(len(rob), -1), conf, dist, art, rob])
