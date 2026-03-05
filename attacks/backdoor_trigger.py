# Backdoor trigger injection utilities
import torch


def add_trigger(image: torch.Tensor) -> torch.Tensor:
    """
    Add a 3x3 white square trigger to the bottom-right corner of an image tensor.

    Args:
        image: Tensor of shape (C, H, W) or (N, C, H, W), values in [0, 1].

    Returns:
        Tensor with trigger patch applied (same shape, same dtype).
    """
    triggered = image.clone()
    if triggered.dim() == 3:
        # Single image (C, H, W)
        _, h, w = triggered.shape
        triggered[:, h - 3:h, w - 3:w] = 1.0
    elif triggered.dim() == 4:
        # Batch (N, C, H, W)
        _, _, h, w = triggered.shape
        triggered[:, :, h - 3:h, w - 3:w] = 1.0
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {triggered.dim()}D")
    return triggered
