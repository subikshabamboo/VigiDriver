"""Explainable AI (XAI) attribution suite: Grad-CAM + LRP.

Grad-CAM (Selvaraju et al.) computes saliency on the last convolutional
feature map of the eye CNN with respect to the drowsy class:

    α_k^c = 1/Z Σ_i Σ_j  ∂Y_c/∂A_ij^k
    L^c   = ReLU( Σ_k α_k^c A^k )

The small activation map is bilinearly upsampled to the input resolution and
rendered as a jet heat map over the eye crop — the attention heat strip shown
in the dashboard/model pages.

LRP (Bach et al.): layer-wise relevance propagated by the gradient-times-input
relation on the flattened features; used as a second, complementary
attribution channel.
"""

from __future__ import annotations

import base64
import threading
from typing import Any

import cv2
import numpy as np

_SIZE = 64


def _find_last_conv_layer(model) -> str:
    conv_layers = [l.name for l in model.layers if "conv" in type(l).__name__.lower()]
    if not conv_layers:
        raise ValueError("no conv layer found for Grad-CAM")
    return conv_layers[-1]


def compute_gradcam(model, eye_gray64: np.ndarray, class_index: int | None = None) -> dict[str, Any]:
    """Grad-CAM heatmap for the eye-state CNN.

    Args:
        model: keras eye CNN (64×64×1 input).
        eye_gray64: (H, W) or (64, 64, 1) uint8 eye crop (any size, resized).
        class_index: gradient target channel of the sigmoid output
            (0 = output activation; for binary sigmoid we target P(eye_open),
            closed-eye heat = 1 - open heat).

    Returns:
        dict with heat (64×64 float32 [0,1]), cam_hm_rgb, png (JPEG bytes),
        p_open value.
    """
    import keras  # noqa: PLC0415 — heavy import, deferred
    import tensorflow as tf  # noqa: PLC0415

    gray = np.asarray(eye_gray64)
    if gray.ndim == 3:
        gray = gray[:, :, 0] if gray.shape[2] == 1 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(gray, (_SIZE, _SIZE), interpolation=cv2.INTER_LINEAR)
    xin = x.astype(np.float32) / 255.0
    xin = xin.reshape(1, _SIZE, _SIZE, 1)

    conv_name = _find_last_conv_layer(model)
    layers = model.layers
    conv_idx = next(i for i, l in enumerate(layers) if l.name == conv_name)
    xin = tf.constant(xin)  # numpy arrays are not tape-watchable

    # Keras 3 functional wrapper models don't reliably expose tape gradients
    # to intermediate layer outputs, so walk the layers manually inside the
    # tape: activations up to (and including) the Grad-CAM target layer, then
    # continue to the classifier head on the principal branch.
    with tf.GradientTape() as tape:
        tape.watch(xin)
        h = xin
        for layer in layers[:conv_idx + 1]:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            if isinstance(layer, keras.layers.Dropout):
                continue  # identity at inference
            h = layer(h, training=False)
        conv_out = h
        score = h
        for layer in layers[conv_idx + 1:]:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            if isinstance(layer, keras.layers.Dropout):
                continue  # identity at inference
            score = layer(score, training=False)
        score = score[:, 0] if score.shape[-1] == 1 else score[:, class_index or 0]

    grads = tape.gradient(score, conv_out)  # (1, H, W, C)
    if grads is None:
        return {"heat": None, "png": None, "conv_layer": conv_name, "p_open": None}

    # Independent full-model forward for the class probability
    percep = float(np.asarray(model(xin, training=False)).reshape(-1)[0])
    grads_np = np.asarray(grads.numpy())
    alpha = grads_np.mean(axis=(0, 1, 2))                   # (C,)
    acts = np.asarray(conv_out[0].numpy())                  # (H, W, C)
    # ReLU(Σ_k α_k A_k) — channel-weighted sum over the C axis
    heat = np.maximum((alpha[None, None, :] * acts).sum(axis=2), 0.0)

    if heat.max() > 0:
        heat = heat / heat.max()
    heat64 = cv2.resize(heat, (_SIZE, _SIZE), interpolation=cv2.INTER_LINEAR)
    heat64 = np.clip(heat64, 0, 1)

    # For a sigmoid P(eye_open) model, closed-eye saliency = inverted heat
    p_closed = float(np.clip(1.0 - percep, 0, 1))
    return {
        "heat": heat64.astype(np.float32).tolist(),
        "conv_layer": conv_name,
        "p_open": round(float(np.asarray(percep)), 4),
        "p_closed": round(p_closed, 4),
    }


def heat_to_png(heat: np.ndarray, eye_gray: np.ndarray, alpha: float = 0.45) -> bytes | None:
    """Render heat overlay (jet colormap over the eye crop) as JPEG bytes."""
    if heat is None:
        return None
    gray = np.asarray(eye_gray, dtype=np.uint8)
    if gray.ndim == 3:
        gray = gray[:, :, 0] if gray.shape[2] == 1 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (_SIZE, _SIZE))
    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_u8 = cv2.resize(heat_u8, (_SIZE, _SIZE), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(colored, alpha, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1 - alpha, 0)
    ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


def compute_lrp(model, eye_gray64: np.ndarray) -> dict[str, Any] | None:
    """LRP-lite via gradient × input relevance on the Dense projection head."""
    import tensorflow as tf  # noqa: PLC0415

    gray = np.asarray(eye_gray64)
    if gray.ndim == 3:
        gray = gray[:, :, 0] if gray.shape[2] == 1 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(gray, (_SIZE, _SIZE), interpolation=cv2.INTER_LINEAR)
    xin = x.astype(np.float32) / 255.0
    xin = xin.reshape(1, _SIZE, _SIZE, 1)
    xin = tf.constant(xin)  # numpy arrays are not tape-watchable
    try:
        with tf.GradientTape() as tape:
            tape.watch(xin)
            preds = model(xin, training=False)
            score = preds[0, 0]
        g = tape.gradient(score, xin).numpy()[0, :, :, 0]
        rel = np.abs(g * xin[0, :, :, 0])
        if rel.max() > 0:
            rel = rel / rel.max()
        return {"lrp_heat": rel.astype(np.float32).tolist()}
    except Exception:
        return None


_xai_lock = threading.Lock()


def eye_gradcam_png(service, eye_crop: np.ndarray | None) -> tuple[str, dict[str, Any]]:
    """One-stop Grad-CAM renderer for the current eye crop.

    Returns (b64 png or "", meta dict). Thread-safe against TF graph access.
    """
    if eye_crop is None:
        return "", {}
    try:
        with _xai_lock:
            heat = compute_gradcam(service.model, eye_crop)
        png = heat_to_png(heat.get("heat"), eye_crop)
        meta = {
            "conv_layer": heat.get("conv_layer"),
            "p_open_xai": heat.get("p_open"),
        }
        return (base64.b64encode(png).decode("ascii") if png else ""), meta
    except Exception:
        return "", {}
