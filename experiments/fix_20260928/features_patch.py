# -*- coding: utf-8 -*-
"""Visual feature fix (fix 2): DINOv2 patch features that keep spatial information.

Original implementation: BitImageProcessor resizes the short side to 256, center-crops 224 and takes the global CLS vector (384-d per camera).
Diagnostics showed that CLS recovers the end-effector position and the drawer opening (R^2≈0.94/0.96) but not the red block position (R^2≈0).

This module provides three encodings, all using the same verified local DINOv2-small weights:
  cls_crop   original implementation (control)
  cls_full   no crop: the full image is resized to 224x224 and the CLS token is taken
  grid{g}    no crop: the 16x16 patch tokens are average-pooled to a gxg grid and flattened (g*g*384-d per camera)
Feature extraction only; DINO is not trained.
"""
import os  # path configuration for the public release; see README, "Environment variables"
import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np

DINO = os.environ.get("DINOV2_DIR", "facebook/dinov2-small")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import Dinov2Model
        m = Dinov2Model.from_pretrained(DINO, local_files_only=True, use_safetensors=True)
        m.eval().requires_grad_(False)
        _MODEL = m
    return _MODEL


def _full_frame(images):
    """List of uint8 HxWx3 images -> no crop, bicubic resize to 224x224, normalised with ImageNet mean and std."""
    import torch
    from PIL import Image
    out = []
    for rgb in images:
        im = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize((224, 224), Image.BICUBIC)
        x = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
        out.append(x.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(out))


def _cls_crop(images):
    import torch
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(DINO, local_files_only=True, use_fast=False)
    with torch.inference_mode():
        return _model()(**proc(images=list(images), return_tensors="pt")).last_hidden_state[:, 0].numpy()


def encode(images, modes=("cls_crop", "cls_full", "grid4")):
    """Return {mode: [N, D]}; images is a list of uint8 images from one camera."""
    import torch
    res = {}
    need_full = any(m != "cls_crop" for m in modes)
    if need_full:
        with torch.inference_mode():
            h = _model()(pixel_values=_full_frame(images)).last_hidden_state  # [N, 1+256, 384]
        cls, patches = h[:, 0], h[:, 1:].reshape(len(images), 16, 16, -1)
    for m in modes:
        if m == "cls_crop":
            res[m] = _cls_crop(images)
        elif m == "cls_full":
            res[m] = cls.numpy()
        elif m == "grid16":
            res[m] = patches.reshape(len(images), 256, -1).numpy().astype(np.float16)  # full patch grid, used by the locator
        elif m.startswith("grid"):
            g = int(m[4:])
            pooled = torch.nn.functional.adaptive_avg_pool2d(patches.permute(0, 3, 1, 2), g)  # [N,384,g,g]
            res[m] = pooled.permute(0, 2, 3, 1).reshape(len(images), -1).numpy()
        else:
            raise ValueError(m)
    return res


def build_cache(npz_paths, out_path, modes=("cls_crop", "cls_full", "grid4"), batch=32):
    """Extract features of both cameras for a set of handoff npz files and cache them; keys are the npz file names."""
    import torch
    torch.set_num_threads(int(os.environ.get("FEAT_THREADS", "16")))
    names, feats = [], {f"{cam}_{m}": [] for cam in ("static", "wrist") for m in modes}
    extras = {"state": [], "scene_state": []}
    for i in range(0, len(npz_paths), batch):
        chunk = npz_paths[i:i + batch]
        arrays = [np.load(p) for p in chunk]
        for cam in ("static", "wrist"):
            enc = encode([a["image_" + cam] for a in arrays], modes)
            for m in modes:
                feats[f"{cam}_{m}"].append(enc[m] if m == "grid16" else enc[m].astype(np.float32))
        for a, p in zip(arrays, chunk):
            names.append(Path(p).name)
            extras["state"].append(a["state"].astype(np.float32))
            extras["scene_state"].append(a["scene_state"].astype(np.float32) if "scene_state" in a.files else np.full(24, np.nan, np.float32))
        print(f"features {min(i + batch, len(npz_paths))}/{len(npz_paths)}", flush=True)
    np.savez_compressed(out_path, names=np.array(names), **{k: np.concatenate(v) for k, v in feats.items()},
                        **{k: np.stack(v) for k, v in extras.items()})
    return out_path
