"""Divergence metrics between clean and backdoored model activations."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.special import kl_div
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cap per-feature KL to avoid extreme outliers dominating the average
MAX_KL_PER_FEATURE = 10.0

VIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "visualizations"
)


def mean_activation_difference(acts_clean: np.ndarray, acts_backdoor: np.ndarray) -> float:
    """Mean absolute difference between mean activation vectors."""
    return float(np.mean(np.abs(acts_clean.mean(axis=0) - acts_backdoor.mean(axis=0))))


def l2_norm_distance(acts_clean: np.ndarray, acts_backdoor: np.ndarray) -> float:
    """L2 norm between mean activation vectors."""
    diff = acts_clean.mean(axis=0) - acts_backdoor.mean(axis=0)
    return float(np.linalg.norm(diff))


def kl_divergence_score(acts_clean: np.ndarray, acts_backdoor: np.ndarray, bins: int = 50) -> float:
    """
    Average per-dimension KL divergence between activation histograms.
    Clamps each dimension's KL to avoid infinities, then averages.
    """
    n_features = min(acts_clean.shape[1], acts_backdoor.shape[1], 64)
    kl_scores = []
    for i in range(n_features):
        lo = min(acts_clean[:, i].min(), acts_backdoor[:, i].min())
        hi = max(acts_clean[:, i].max(), acts_backdoor[:, i].max())
        if hi == lo:
            kl_scores.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(acts_clean[:, i], bins=edges, density=True)
        q, _ = np.histogram(acts_backdoor[:, i], bins=edges, density=True)
        # Smooth to avoid zero division
        p = p + 1e-10
        q = q + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        kl = float(np.sum(kl_div(p, q)))
        kl_scores.append(min(kl, MAX_KL_PER_FEATURE))
    return float(np.mean(kl_scores))


def compute_divergence_score(
    acts_clean: np.ndarray,
    acts_backdoor: np.ndarray,
    save_plots: bool = True,
) -> Tuple[float, dict]:
    """
    Compute a normalized divergence score (0–1) from multiple metrics.

    Returns:
        (divergence_score, metrics_dict)
    """
    mad = mean_activation_difference(acts_clean, acts_backdoor)
    l2 = l2_norm_distance(acts_clean, acts_backdoor)
    kl = kl_divergence_score(acts_clean, acts_backdoor)

    # Scaling constants for soft normalisation of each metric to [0, 1].
    # L2_SCALE_FACTOR: L2 values rarely exceed 30 % of the reference norm
    # in benign perturbations, so we dampen the L2 contribution to avoid it
    # drowning out the MAD and KL terms.
    L2_SCALE_FACTOR = 0.3
    # KL_CAP: Average per-feature KL divergences above ~5 nats indicate
    # near-orthogonal distributions; we saturate the score at this threshold.
    KL_CAP = 5.0

    # Normalize each metric to [0, 1] using soft caps
    mad_norm = min(mad / (np.abs(acts_clean).mean() + 1e-8), 1.0)
    l2_norm = min(l2 / (np.linalg.norm(acts_clean.mean(axis=0)) + 1e-8), 1.0) * L2_SCALE_FACTOR
    kl_norm = min(kl / KL_CAP, 1.0)

    divergence_score = float(np.clip((mad_norm + l2_norm + kl_norm) / 3.0, 0.0, 1.0))

    metrics = {
        "mean_activation_difference": mad,
        "l2_norm_distance": l2,
        "kl_divergence": kl,
        "divergence_score": divergence_score,
    }

    if save_plots:
        _plot_pca(acts_clean, acts_backdoor)
        _plot_activation_distributions(acts_clean, acts_backdoor)

    return divergence_score, metrics


def _plot_pca(acts_clean: np.ndarray, acts_backdoor: np.ndarray) -> None:
    """PCA 2D embedding of clean vs backdoor activations."""
    os.makedirs(VIS_DIR, exist_ok=True)
    n = min(len(acts_clean), len(acts_backdoor), 500)
    combined = np.vstack([acts_clean[:n], acts_backdoor[:n]])
    pca = PCA(n_components=2, random_state=42)
    embedded = pca.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(embedded[:n, 0], embedded[:n, 1], c="steelblue", alpha=0.5, s=10, label="Clean Model")
    ax.scatter(embedded[n:, 0], embedded[n:, 1], c="tomato", alpha=0.5, s=10, label="Backdoor Model")
    ax.set_title("PCA of Activations: Clean vs Backdoor")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "pca_activations.png"), dpi=150)
    plt.close()
    print(f"Saved PCA plot to {VIS_DIR}/pca_activations.png")


def _plot_activation_distributions(acts_clean: np.ndarray, acts_backdoor: np.ndarray) -> None:
    """Histogram comparison of activation distributions (first 4 features)."""
    os.makedirs(VIS_DIR, exist_ok=True)
    n_features = min(4, acts_clean.shape[1], acts_backdoor.shape[1])
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))
    if n_features == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.hist(acts_clean[:, i], bins=40, alpha=0.6, color="steelblue", label="Clean", density=True)
        ax.hist(acts_backdoor[:, i], bins=40, alpha=0.6, color="tomato", label="Backdoor", density=True)
        ax.set_title(f"Feature {i}")
        ax.legend()
    fig.suptitle("Activation Distribution Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "activation_distributions.png"), dpi=150)
    plt.close()
    print(f"Saved activation distributions plot to {VIS_DIR}/activation_distributions.png")
