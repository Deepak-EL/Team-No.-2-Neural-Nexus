"""
Inference utilities for the coral-reef health models.

Re-implements, outside the notebook, exactly the inference path used in the paper:
  1. gray-world white balance + LAB-CLAHE underwater colour correction
  2. 224x224 point-centred crops (reflection padding) or a sliding window over the whole quadrat
  3. validation-tuned prior-shift correction  p ~ p_model * (pi_nat / pi_train) ** tau
  4. Gaussian-blended dense health map (sliding window) -> % cover per class + reef-status category

Works with every checkpoint in ../models (MobileNetV2, ResNet18, EfficientNet-B0, ShuffleNetV2, CoralCNN-SE, DINOv2-ViT-S/14).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from coral_model_defs import CLASS_NAMES, build_model

CROP, HALF = 224, 112
SW_DS = 4                                   # dense map is computed at 1/4 resolution (as in the paper)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_RESULTS = os.path.join(REPO, "results", "results_master.json")
DEFAULT_MODELS = os.path.join(REPO, "models")
CLASS_COLOURS = np.array([[46, 204, 113], [231, 76, 60], [52, 152, 219]], dtype=np.uint8)  # healthy, dead, substrate

MODEL_FILES = {
    "MobileNetV2": "MobileNetV2.pt",
    "ResNet18": "ResNet18.pt",
    "EfficientNet-B0": "EfficientNet-B0.pt",
    "ShuffleNetV2": "ShuffleNetV2.pt",
    "CoralCNN-SE": "CoralCNN-SE.pt",
    "DINOv2-ViT-S/14": "DINOv2-ViT-S-14.pt",
}


# ───────────────────────────── pre-processing ─────────────────────────────
def apply_underwater_filter(rgb: np.ndarray, clip: float = 1.5) -> np.ndarray:
    """Gray-world white balance (per-channel gain clipped to [0.5, 2.5]) + CLAHE on the LAB L-channel. uint8 RGB -> uint8 RGB."""
    m = cv2.mean(rgb)[:3]
    gray = sum(m) / 3.0
    out = np.empty_like(rgb)
    for c in range(3):
        g = float(np.clip(gray / (m[c] + 1e-6), 0.5, 2.5))
        lut = np.clip(np.arange(256) * g, 0, 255).astype(np.uint8)
        out[..., c] = cv2.LUT(np.ascontiguousarray(rgb[..., c]), lut)
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def pad_reflect(img: np.ndarray) -> np.ndarray:
    return cv2.copyMakeBorder(img, HALF, HALF, HALF, HALF, cv2.BORDER_REFLECT)


def point_crops(img: np.ndarray, rows, cols) -> np.ndarray:
    """(H,W,3) uint8 -> (N,224,224,3) crops centred on each (row, col)."""
    H, W = img.shape[:2]
    pad = pad_reflect(img)
    return np.stack([pad[int(np.clip(r, 0, H - 1)):int(np.clip(r, 0, H - 1)) + CROP,
                         int(np.clip(c, 0, W - 1)):int(np.clip(c, 0, W - 1)) + CROP] for r, c in zip(rows, cols)])


# ───────────────────────────── model loading ─────────────────────────────
@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    tau: float
    prior_ratio: np.ndarray       # pi_nat / pi_train
    device: torch.device


def load_results(path: str = DEFAULT_RESULTS) -> dict:
    with open(path) as f:
        return json.load(f)


def load_model(name: str, models_dir: str = DEFAULT_MODELS, results_path: str = DEFAULT_RESULTS,
               device: str | None = None, tau: float | None = None) -> LoadedModel:
    """Load a trained checkpoint together with its validation-tuned tau and the training-split class prior."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(os.path.join(models_dir, MODEL_FILES[name]), map_location="cpu", weights_only=False)
    model = build_model(ck.get("name", name), pretrained=False,
                        mean=ck.get("data_mean", (0.46, 0.54, 0.51)), std=ck.get("data_std", (0.22, 0.21, 0.21)))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(dev)
    res = load_results(results_path) if os.path.exists(results_path) else {}
    nat = np.array(res.get("nat_prior_train_split", [0.16996, 0.16831, 0.66173]), dtype=np.float32)
    t = tau if tau is not None else float(res.get("tau", {}).get(name, 0.0))
    return LoadedModel(name, model, t, nat / (1.0 / 3.0), dev)


# ───────────────────────────── prediction ─────────────────────────────
@torch.no_grad()
def predict_batch(lm: LoadedModel, crops_uint8: torch.Tensor | np.ndarray, batch: int = 64) -> np.ndarray:
    """crops (N,224,224,3) uint8 (numpy) or (N,3,224,224) uint8 tensor -> prior-corrected probabilities (N,3)."""
    if isinstance(crops_uint8, np.ndarray):
        crops_uint8 = torch.from_numpy(crops_uint8).permute(0, 3, 1, 2)
    ratio = torch.tensor(lm.prior_ratio ** lm.tau, dtype=torch.float32, device=lm.device)
    out = []
    for i in range(0, crops_uint8.shape[0], batch):
        x = crops_uint8[i:i + batch].to(lm.device).float().div_(255.0)
        p = torch.softmax(lm.model(x).float(), 1) * ratio
        out.append((p / p.sum(1, keepdim=True)).cpu())
    return torch.cat(out).numpy()


def _gauss_kernel(k: int, sigma: float = 0.35, device="cpu") -> torch.Tensor:
    ax = torch.linspace(-1, 1, k, device=device)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / g.max()


@torch.no_grad()
def dense_map(lm: LoadedModel, img: np.ndarray, stride: int = 112, batch: int = 64, progress=None) -> np.ndarray:
    """Sliding-window inference over a whole (colour-corrected) quadrat.
    Returns dense class probabilities (3, ceil(H/4), ceil(W/4)) aligned with the image at 1/4 resolution."""
    H, W = img.shape[:2]
    t = torch.from_numpy(pad_reflect(img)).permute(2, 0, 1)                    # (3, H+224, W+224) uint8
    u = t.unfold(1, CROP, stride).unfold(2, CROP, stride)                     # (3, ny, nx, 224, 224)
    ny, nx = u.shape[1], u.shape[2]
    win = u.permute(1, 2, 0, 3, 4).reshape(ny * nx, 3, CROP, CROP)
    probs = []
    for i in range(0, win.shape[0], batch):
        probs.append(torch.from_numpy(predict_batch(lm, win[i:i + batch].contiguous(), batch)))
        if progress:
            progress(min(1.0, (i + batch) / win.shape[0]))
    probs = torch.cat(probs)                                                   # (N, 3)
    k, s, N = CROP // SW_DS, stride // SW_DS, probs.shape[0]
    kern = _gauss_kernel(k)
    contrib = (probs[:, :, None, None] * kern[None, None]).reshape(N, 3 * k * k).t().unsqueeze(0)
    OH, OW = (ny - 1) * s + k, (nx - 1) * s + k
    num = F.fold(contrib, (OH, OW), k, stride=s)
    den = F.fold(kern.reshape(1, k * k, 1).expand(1, k * k, N).contiguous(), (OH, OW), k, stride=s)
    dense = (num / den.clamp_min(1e-6))[0]
    o = HALF // SW_DS
    return dense[:, o:o + math.ceil(H / SW_DS), o:o + math.ceil(W / SW_DS)].numpy()


def reef_status(healthy_pct: float) -> str:
    if healthy_pct >= 50: return "Excellent (≥50 % live coral)"
    if healthy_pct >= 30: return "Good (30–50 % live coral)"
    if healthy_pct >= 10: return "Moderate (10–30 % live coral)"
    return "Poor (<10 % live coral)"


def cover_from_dense(dense: np.ndarray) -> dict:
    seg = dense.argmax(0)
    pct = {c: float((seg == i).mean() * 100) for i, c in enumerate(CLASS_NAMES)}
    pct["Reef status"] = reef_status(pct["Healthy Coral"])
    return pct


def overlay(img: np.ndarray, dense: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Colour the (full-resolution) image by the arg-max class of the dense map."""
    seg = dense.argmax(0).astype(np.uint8)
    seg = cv2.resize(seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    col = CLASS_COLOURS[seg]
    return (img.astype(np.float32) * (1 - alpha) + col.astype(np.float32) * alpha).astype(np.uint8)


def analyse_quadrat(lm: LoadedModel, rgb: np.ndarray, stride: int = 112, max_side: int | None = None,
                    filter_image: bool = True, progress=None) -> dict:
    """Full pipeline for one image: colour correction -> sliding window -> dense map, cover %, overlay."""
    if max_side and max(rgb.shape[:2]) > max_side:
        s = max_side / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
    img = apply_underwater_filter(rgb) if filter_image else rgb
    dense = dense_map(lm, img, stride=stride, progress=progress)
    return dict(image=img, dense=dense, cover=cover_from_dense(dense), overlay=overlay(img, dense))
