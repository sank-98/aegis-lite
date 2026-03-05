"""Extract intermediate activations from a model using forward hooks."""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_activations(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer_name: str,
    device: torch.device = None,
    max_samples: int = 2000,
) -> np.ndarray:
    """
    Extract activations from a named layer using a forward hook.

    Args:
        model: PyTorch model (SimpleCNN or similar).
        dataloader: DataLoader providing input batches.
        layer_name: Name of the layer to hook (e.g. 'conv2', 'fc1').
        device: Torch device. Auto-detected if None.
        max_samples: Maximum number of samples to collect.

    Returns:
        Numpy array of shape (N, features) containing flattened activations.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    activations: List[torch.Tensor] = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    # Register hook on the named layer
    layer = dict(model.named_modules()).get(layer_name)
    if layer is None:
        available = [name for name, _ in model.named_modules() if name]
        raise ValueError(
            f"Layer '{layer_name}' not found. Available layers: {available}"
        )
    handle = layer.register_forward_hook(hook_fn)

    collected = 0
    with torch.no_grad():
        for images, _ in dataloader:
            if collected >= max_samples:
                break
            images = images.to(device)
            _ = model(images)
            collected += images.size(0)

    handle.remove()

    # Concatenate and flatten spatial dimensions if needed
    all_acts = torch.cat(activations, dim=0)[:max_samples]
    if all_acts.dim() > 2:
        all_acts = all_acts.view(all_acts.size(0), -1)

    return all_acts.numpy()
