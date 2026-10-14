"""Cross-modal positional extraction used by the released ItraModel.

The image tokens precede the text tokens. A shared two-layer Transformer
mixes both streams, and each position's attended feature is subtracted from
its original projected feature. The two linear layers then filter the residuals.
"""

import torch
from torch import nn


def extract_positional_residuals(
    image_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    encoder: nn.Module,
    image_filter: nn.Module,
    text_filter: nn.Module,
    text_padding_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return image/text residuals while preserving the original weights.

    Inputs are projected tokens with shapes [B, Ni, D] and [B, Nt, D].
    By default this matches the released code, which passes all text positions
    (including padding) to the encoder. Paper mode can supply a padding mask.
    """
    if image_tokens.ndim != 3 or text_tokens.ndim != 3:
        raise ValueError("Image and text tokens must each have shape [B, N, D]")
    if image_tokens.shape[0] != text_tokens.shape[0] or image_tokens.shape[2] != text_tokens.shape[2]:
        raise ValueError("Image and text tokens must share batch size and width")

    image_count = image_tokens.shape[1]
    joint = torch.cat((image_tokens, text_tokens), dim=1)
    if text_padding_mask is None:
        attended = encoder(joint)
    else:
        if text_padding_mask.shape != text_tokens.shape[:2]:
            raise ValueError("Text padding mask must have shape [B, Nt]")
        image_padding_mask = torch.zeros(image_tokens.shape[:2], dtype=torch.bool, device=image_tokens.device)
        joint_padding_mask = torch.cat((image_padding_mask, text_padding_mask), dim=1)
        attended = encoder(joint, src_key_padding_mask=joint_padding_mask)
    image_residual = image_filter(image_tokens - attended[:, :image_count])
    text_residual = text_filter(text_tokens - attended[:, image_count:])
    return image_residual, text_residual
