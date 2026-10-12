# -*- coding: utf-8 -*-
"""State features v2: the 32-d v1 features plus geometry and local appearance that push tasks may depend on.
Added (all derived only from static-camera RGB and robot state; supervision only from training-fold simulator ground truth):
  orientation (6)        mean DINO descriptor of the 3x3 patches around each block -> ridge regression -> sin/cos(yaw about the vertical axis)
  relative geometry (9)  displacement of each block relative to the end effector, rotated into the end effector's yaw frame (dx', dy', dz)
  local appearance (12)  local descriptor of each block reduced to 4-d by PCA
Total 32 + 27 = 59-d."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from benefit_lib import Featurizer, OBJECTS, W, H, GRID

YAW_IDX = {"red": 11, "blue": 17, "pink": 23}


def local_desc(G16, uv, chunk=256):
    """G16 [N,256,384]; uv [N,3,2] pixel coordinates -> [N,3,384] mean over the 3x3 patches around each block."""
    N = len(G16)
    out = np.zeros((N, 3, G16.shape[-1]), np.float32)
    col = np.clip((uv[..., 0] / (W / GRID)).astype(int), 0, GRID - 1)
    row = np.clip((uv[..., 1] / (H / GRID)).astype(int), 0, GRID - 1)
    offs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    for s in range(0, N, chunk):
        g = np.asarray(G16[s:s + chunk], np.float32).reshape(-1, GRID, GRID, G16.shape[-1])
        for o in range(3):
            acc = np.zeros((len(g), G16.shape[-1]), np.float32)
            for dr, dc in offs:
                r = np.clip(row[s:s + chunk, o] + dr, 0, GRID - 1); c = np.clip(col[s:s + chunk, o] + dc, 0, GRID - 1)
                acc += g[np.arange(len(g)), r, c]
            out[s:s + chunk, o] = acc / len(offs)
    return out


class Featurizer2:
    def fit(self, G16, G8, scene, seed=0):
        self.base = Featurizer().fit(G16, G8, scene, seed=seed)
        _, _, uv = self._loc(G16)
        D = local_desc(G16, uv)
        self.yaw, self.pca = [], []
        for o, name in enumerate(OBJECTS):
            yaw = scene[:, YAW_IDX[name]]
            sc = StandardScaler().fit(D[:, o]); Z = sc.transform(D[:, o])
            self.yaw.append((sc, RidgeCV(alphas=np.logspace(0, 5, 11)).fit(Z, np.c_[np.sin(yaw), np.cos(yaw)])))
            self.pca.append(PCA(4, random_state=seed).fit(Z))
        return self

    def _loc(self, G16):
        loc = self.base.loc
        prob = loc._heat(G16)
        uv = np.stack([prob[:, :, i] @ loc.grid_uv for i in range(3)], 1)
        from benefit_lib import _basis
        xyz = np.stack([_basis(uv[:, i]) @ loc.maps[i] for i in range(3)], 1)
        return xyz, prob.max(1), uv

    def transform(self, G16, G8, rob):
        F = self.base.transform(G16, G8, rob)
        xyz, _, uv = self._loc(G16)
        D = local_desc(G16, uv)
        yaw = np.hstack([m.predict(sc.transform(D[:, o])) for o, (sc, m) in enumerate(self.yaw)])
        loc_pc = np.hstack([p.transform(self.yaw[o][0].transform(D[:, o])) for o, p in enumerate(self.pca)])
        th = rob[:, 5]
        c, s = np.cos(-th), np.sin(-th)
        rel = []
        for o in range(3):
            d = xyz[:, o] - rob[:, :3]
            rel.append(np.c_[c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]])
        return np.hstack([F, yaw, np.hstack(rel), loc_pc])
