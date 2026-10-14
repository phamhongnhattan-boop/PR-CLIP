"""Extract cross-modal token information with a shared self-attention encoder."""

import torch
from torch import nn


class CrossModalPositionExtractor(nn.TransformerEncoder):
    """Map projected image/text tokens to aligned, position-aware features.

    Inputs: image [B, Ni, D], text [B, Nt, D], text mask [B, Nt].
    Outputs: image features [B, Ni, D], text features [B, Nt, D].
    True in the mask marks a text padding position that attention must ignore.
    """

    def forward(self, image, text, text_padding_mask):
        image_padding_mask = torch.zeros(
            image.shape[:2], device=image.device, dtype=torch.bool
        )
        joint_padding_mask = torch.cat(
            [image_padding_mask, text_padding_mask], dim=1
        )
        joint = torch.cat([image, text], dim=1)
        extracted = super().forward(
            joint, src_key_padding_mask=joint_padding_mask
        )
        ni = image.shape[1]
        return extracted[:, :ni], extracted[:, ni:]
