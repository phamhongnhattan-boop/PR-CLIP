"""The PR-CLIP paper's token-level positional reconstruction objective."""

import torch


def positional_reconstruction_loss(
    image_reconstructed: torch.Tensor,
    text_reconstructed: torch.Tensor,
    image_original: torch.Tensor,
    text_original: torch.Tensor,
    text_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average squared L2 distance across image and valid text tokens.

    The paper does not specify padding implementation details. We include the
    EOT token and exclude positions after it when a mask is supplied.
    """
    if image_reconstructed.shape != image_original.shape:
        raise ValueError("Reconstructed and original image tokens must match")
    if text_reconstructed.shape != text_original.shape:
        raise ValueError("Reconstructed and original text tokens must match")
    image_error = (image_reconstructed - image_original).square().sum(dim=-1).mean()
    text_error = (text_reconstructed - text_original).square().sum(dim=-1)
    if text_padding_mask is None:
        text_error = text_error.mean()
    else:
        if text_padding_mask.shape != text_error.shape:
            raise ValueError("Text padding mask must have shape [B, Nt]")
        valid = ~text_padding_mask
        text_error = (text_error * valid).sum() / valid.sum().clamp_min(1)
    return (image_error + text_error) / 2


def paper_training_loss(
    cmc: torch.Tensor,
    prc: torch.Tensor,
    lambda_cmc: float = 0.01,
    lambda_prc: float = 1.0,
) -> torch.Tensor:
    """Equation (13): combine contrastive alignment and reconstruction."""
    return lambda_cmc * cmc + lambda_prc * prc
