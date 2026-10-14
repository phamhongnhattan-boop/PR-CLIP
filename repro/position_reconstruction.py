"""Recover each filtered token stream using the opposite modality."""

from torch import nn


class CrossModalPositionReconstructor(nn.TransformerDecoder):
    """Decode [B, N_target, D] with [B, N_context, D] as memory.

    One instance is reused for image-from-text and text-from-image so the
    two directions share transformer decoder weights.
    """

    def forward(self, filtered, opposite_modality, *,
                target_padding_mask=None, memory_padding_mask=None):
        return super().forward(
            tgt=filtered,
            memory=opposite_modality,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
