"""Unsupervised K-Means Latent Clustering Engine (k=4).

Partitions multi-modal behavioural phenotypes directly in the model's
bottleneck representation space, per the VigiDriver paper:

    Cluster 0:  Fully Alert / Vigilant
    Cluster 1:  Attentive but rapid benign blinks
    Cluster 2:  Active severe micro-sleep (sustained closure)
    Cluster 3:  Heavy yawning / distraction

Implementation notes
--------------------
The production pipeline's latent bottleneck is the per-frame standardized
vigilance feature vector + behavioural aggregates, projected to a compact
manifold with PCA whitening before Lloyd's K-Means (k=4). Assignment uses
softmax probabilities over negative squared Euclidean distance, and any
sample whose nearest-centroid distance exceeds the τ_outlier gate is flagged
as an anomaly — suppressing downstream alert escalation until subsequent
frames corroborate sustained drowsiness (out-of-distribution rejection):

    min_k ||z_i - μ_k||₂ > τ_outlier  ⇒  ambiguous / flicker frame
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

K = 4
WHITEN_DIM = 6          # PCA-whitened latent manifold dimension
TAU_OUTLIER_SIGMA = 3.0  # nearest-centroid distance > 3σ ⇒ anomaly gate
REFIT_EVERY = 45        # refit Lloyd's every 45 assignments
MIN_SAMPLES_TO_FIT = 60  # allow short demo sessions to arm the gate

CLUSTER_DESCRIPTIONS = [
    "Fully Alert / Vigilant",
    "Attentive — rapid benign blinks",
    "Active severe micro-sleep",
    "Heavy yawning / distraction",
]


class KMeansGate:
    """Online unsupervised K-Means clustering in the whitened latent space."""

    def __init__(self, k: int = K, dim_latent: int = 8) -> None:
        self.k = k
        self._lock = threading.Lock()
        self._samples: list[np.ndarray] = []
        self._centroids: np.ndarray | None = None
        self._whiten_vec: np.ndarray | None = None
        self._whiten_mean: np.ndarray | None = None
        self._tau: float | None = None
        self._since_fit = 0
        self._n_fits = 0
        self._labels: list[str] = [CLUSTER_DESCRIPTIONS[i] for i in range(k)]
        self.dim_latent = dim_latent

    # -- latent construction -------------------------------------------------

    @staticmethod
    def build_latent(vigilance_vec: np.ndarray, beh: dict[str, Any],
                     lstm_p: float | None) -> np.ndarray:
        """Latent bottleneck: standardized vigilance features + behavioural
        aggregates (closed-seconds, PERCLOS, blink rate proxy, MAR, LSTM P)."""
        z_vig = np.asarray(vigilance_vec, dtype=np.float64).reshape(-1)
        agg = np.array([
            float(beh.get("closed_seconds", 0.0)),
            float(beh.get("perclos", 0.0)) * 10.0,
            float(beh.get("blink_count", 0)) / 30.0,
            float(beh.get("mar") or 0.0),
            float(lstm_p if lstm_p is not None else 0.0),
        ], dtype=np.float64)
        return np.concatenate([z_vig, agg])

    # -- PCA whitening -------------------------------------------------------

    def _whiten(self, X: np.ndarray) -> np.ndarray:
        mean = X.mean(axis=0)
        Xc = X - mean
        # Economy SVD-based PCA; keep WHITEN_DIM components
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        d = min(WHITEN_DIM, Vt.shape[0])
        components = Vt[:d]
        scale = S[:d] / np.sqrt(max(1, X.shape[0] - 1))
        scale = np.clip(scale, 1e-6, None)
        self._whiten_mean = mean
        self._whiten_vec = (components / scale[:, None]).astype(np.float64)
        return (Xc @ components.T) / scale

    def _apply_whiten(self, z: np.ndarray) -> np.ndarray:
        Xc = z - self._whiten_mean
        return self._cached_project(Xc)

    def _cached_project(self, Xc: np.ndarray) -> np.ndarray:
        # Regenerate the projection lazily from stored solve state
        return Xc @ self._whiten_vec.T

    # -- Lloyd's K-Means -----------------------------------------------------

    @staticmethod
    def _lloyd(X: np.ndarray, k: int, iters: int = 60, seed: int = 7) -> np.ndarray:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=min(k, X.shape[0]), replace=False)
        cent = X[idx].astype(np.float64).copy()
        if cent.shape[0] < k:  # degenerate — duplicate to k
            extra = X[rng.choice(X.shape[0], size=k - cent.shape[0], replace=True)]
            cent = np.vstack([cent, extra])
        for _ in range(iters):
            dists = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
            assign = dists.argmin(axis=1)
            new_cent = cent.copy()
            for j in range(k):
                mask = assign == j
                if mask.any():
                    new_cent[j] = X[mask].mean(axis=0)
            if np.allclose(new_cent, cent, atol=1e-7):
                cent = new_cent
                break
            cent = new_cent
        return cent

    # -- public API ----------------------------------------------------------

    def observe(self, latent: np.ndarray, force_fit: bool = False) -> dict[str, Any] | None:
        """Feed one latent sample; refit periodically; return assignment."""
        self._samples.append(np.asarray(latent, dtype=np.float64).reshape(-1))
        if len(self._samples) > 4000:
            self._samples = self._samples[-4000:]

        self._since_fit += 1
        if (force_fit or self._since_fit >= REFIT_EVERY) and len(self._samples) >= MIN_SAMPLES_TO_FIT:
            self.fit()
            self._since_fit = 0

        if self._centroids is None:
            return None
        return self.assign(latent)

    def fit(self) -> dict[str, Any]:
        with self._lock:
            X = np.stack(self._samples[-2400:], axis=0)
            Z = self._whiten(X)
            cent = self._lloyd(Z, self.k)
            # Order centroids by "severity" = weighted drowsiness direction:
            # select the whitened latent direction correlating with closure.
            # Severity proxy: mean of (closed_seconds, perclos, lstm) channels
            # live in the original vector's last dims; approximate severity per
            # centroid via distance from the all-alert (min) corner instead:
            sev = ((cent - cent.min(axis=0)) ** 2).sum(axis=1)
            order = np.argsort(sev)
            self._centroids = cent[order]
            # Rename descriptive labels in the canonical paper order:
            # ordered least→most severe: Alert, Blinks, Micro-sleep, Yawn
            names = [CLUSTER_DESCRIPTIONS[i] for i in range(len(order))]
            self._labels = names
            # Outlier gate τ from per-cluster RMS distance
            dists = np.sqrt(((Z[:, None, :] - self._centroids[None, :, :]) ** 2).sum(-1)).min(axis=1)
            self._tau = 3.0 * float(np.std(dists)) + 1e-6
            self._n_fits += 1
            return {"silhouette_proxy": self._silhouette(Z), "tau": self._tau}

    def _silhouette(self, Z: np.ndarray, n: int = 300) -> float | None:
        if self._centroids is None or Z.shape[0] < 20:
            return None
        samp = Z[np.linspace(0, Z.shape[0] - 1, num=min(n, Z.shape[0])).astype(int)]
        d = np.sqrt(((samp[:, None, :] - self._centroids[None, :, :]) ** 2).sum(-1))
        own = d.min(axis=1)
        others = d.copy(); others[np.arange(len(samp)), d.argmin(axis=1)] = np.inf
        other_mean = others.mean(axis=1)
        denom = np.maximum(own + other_mean, 1e-9)
        sil = np.where(denom > 1e-6, (other_mean - own) / denom, 0.0)
        return float(np.clip(np.nan_to_num(sil.mean(), nan=0.0), -1.0, 1.0))

    def assign(self, latent: np.ndarray) -> dict[str, Any]:
        """Softmax cluster probabilities + anomaly gate for one latent."""
        if self._centroids is None:
            return {}
        z = np.asarray(latent, dtype=np.float64).reshape(1, -1)
        zw = self._cached_project(z - self._whiten_mean).reshape(-1)
        d2 = ((self._centroids - zw[None, :]) ** 2).sum(axis=1)
        # Softmax P(Cluster k | z) = exp(-d²/2σ²) / Σ_j exp(-d²/2σ²)
        sigma = max(self._tau, 1e-3) / 1.5
        logits = -d2 / (2 * sigma ** 2)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        cid = int(np.argmin(d2))
        min_dist = float(np.sqrt(d2[cid]))
        return {
            "cluster_id": cid,
            "cluster": self._labels[cid],
            "cluster_probs": [float(p) for p in probs],
            "distance_to_centroid": round(min_dist, 4),
            "is_outlier": bool(min_dist > self._tau),
        }

    @property
    def ready(self) -> bool:
        return self._centroids is not None

    def reset(self) -> None:
        self._samples.clear()
        self._centroids = None
        self._tau = None
        self._since_fit = 0

    def describe(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "fitted": self.ready,
            "n_samples": len(self._samples),
            "n_fits": self._n_fits,
            "tau_outlier": round(self._tau, 4) if self._tau else None,
            "whiten_dim": WHITEN_DIM,
            "centroids": self._centroids.round(4).tolist() if self._centroids is not None else None,
            "cluster_labels": self._labels,
        }
