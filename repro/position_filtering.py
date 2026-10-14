"""Filter extracted cross-modal information from each original token stream."""

from torch import nn


class UnimodalPositionFilter(nn.Linear):
    """Apply Linear(original - extracted), preserving [B, tokens, D].

    The subtraction is the paper's proposed filtering operation; it does not
    mathematically guarantee that all positional information is removed.
    """

    def forward(self, original, extracted):
        return super().forward(original - extracted)
