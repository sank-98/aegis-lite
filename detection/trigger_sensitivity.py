"""Trigger sensitivity analysis: measures confidence shift when trigger is applied."""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.backdoor_trigger import add_trigger

VIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "visualizations"
)


def compute_trigger_sensitivity(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = None,
    n_batches: int = 10,
    target_class: int = 0,
    save_plots: bool = True,
) -> float:
    """
    Measure how much the model's confidence on the target class changes
    when a trigger is added to clean images.

    Args:
        model: PyTorch model.
        dataloader: DataLoader of clean images.
        device: Torch device.
        n_batches: Number of batches to evaluate.
        target_class: The backdoor target class.
        save_plots: Whether to save visualisation.

    Returns:
        sensitivity_score: Float in [0, 1]. Higher means model is more
        sensitive to the trigger (more likely backdoored).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    conf_clean_list = []
    conf_triggered_list = []

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            images = images.to(device)
            triggered = add_trigger(images)

            probs_clean = F.softmax(model(images), dim=1)[:, target_class]
            probs_triggered = F.softmax(model(triggered), dim=1)[:, target_class]

            conf_clean_list.append(probs_clean.cpu().numpy())
            conf_triggered_list.append(probs_triggered.cpu().numpy())

    conf_clean = np.concatenate(conf_clean_list)
    conf_triggered = np.concatenate(conf_triggered_list)

    avg_shift = float(np.mean(conf_triggered - conf_clean))
    # Normalise to [0, 1]: a shift of 0.5 or more is extreme
    sensitivity_score = float(np.clip(avg_shift / 0.5, 0.0, 1.0))

    if save_plots:
        _plot_sensitivity(conf_clean, conf_triggered)

    return sensitivity_score


def _plot_sensitivity(conf_clean: np.ndarray, conf_triggered: np.ndarray) -> None:
    """Visualise confidence distribution before and after trigger."""
    os.makedirs(VIS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(conf_clean, bins=40, alpha=0.7, color="steelblue", label="Clean confidence", density=True)
    ax.hist(conf_triggered, bins=40, alpha=0.7, color="tomato", label="Triggered confidence", density=True)
    ax.set_xlabel("P(target class)")
    ax.set_ylabel("Density")
    ax.set_title("Trigger Sensitivity: Confidence on Target Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "trigger_sensitivity.png"), dpi=150)
    plt.close()
    print(f"Saved trigger sensitivity plot to {VIS_DIR}/trigger_sensitivity.png")
