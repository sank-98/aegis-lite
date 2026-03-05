"""Compute integrity score for a given model."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.extract_activations import extract_activations
from detection.divergence_metrics import compute_divergence_score
from detection.trigger_sensitivity import compute_trigger_sensitivity

VIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "visualizations"
)

W1 = 0.5  # weight for divergence score  (equal weighting chosen as a neutral baseline;
W2 = 0.5  # weight for trigger sensitivity  tune these based on your specific threat model)


def compute_integrity_score(
    clean_model: torch.nn.Module,
    suspect_model: torch.nn.Module,
    dataloader: DataLoader,
    layer_name: str = "fc1",
    device: torch.device = None,
    save_plots: bool = True,
) -> dict:
    """
    Compute integrity score for suspect_model relative to clean_model.

    Integrity Score = 100 - (W1 * divergence_score + W2 * sensitivity_score) * 100

    Args:
        clean_model: Reference clean model.
        suspect_model: Model under inspection.
        dataloader: Clean test dataloader.
        layer_name: Layer to extract activations from.
        device: Torch device.
        save_plots: Whether to save plots.

    Returns:
        dict with keys: integrity_score, divergence_score, sensitivity_score, metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[IntegrityScore] Extracting activations from layer '{layer_name}'...")
    acts_clean = extract_activations(clean_model, dataloader, layer_name, device=device)
    acts_suspect = extract_activations(suspect_model, dataloader, layer_name, device=device)

    print("[IntegrityScore] Computing divergence metrics...")
    divergence_score, metrics = compute_divergence_score(acts_clean, acts_suspect, save_plots=save_plots)

    print("[IntegrityScore] Computing trigger sensitivity...")
    sensitivity_score = compute_trigger_sensitivity(suspect_model, dataloader, device=device, save_plots=save_plots)

    integrity_score = 100.0 - (W1 * divergence_score + W2 * sensitivity_score) * 100.0
    integrity_score = float(np.clip(integrity_score, 0.0, 100.0))

    result = {
        "integrity_score": integrity_score,
        "divergence_score": divergence_score,
        "sensitivity_score": sensitivity_score,
        "metrics": metrics,
    }

    print(f"\n{'='*50}")
    print(f"  Divergence Score:   {divergence_score:.4f}")
    print(f"  Sensitivity Score:  {sensitivity_score:.4f}")
    print(f"  INTEGRITY SCORE:    {integrity_score:.2f} / 100")
    print(f"{'='*50}\n")

    return result


def compare_and_plot(clean_result: dict, backdoor_result: dict) -> None:
    """Bar chart comparing integrity scores of clean vs backdoor model."""
    os.makedirs(VIS_DIR, exist_ok=True)
    labels = ["Clean Model", "Backdoor Model"]
    scores = [clean_result["integrity_score"], backdoor_result["integrity_score"]]
    colors = ["steelblue", "tomato"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, scores, color=colors, edgecolor="black", width=0.4)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Integrity Score (0–100)")
    ax.set_title("Model Integrity Score Comparison")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{score:.1f}", ha="center", va="bottom", fontweight="bold")
    ax.axhline(70, color="orange", linestyle="--", alpha=0.8, label="Suspicion threshold (70)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "integrity_score_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved integrity score comparison to {VIS_DIR}/integrity_score_comparison.png")
