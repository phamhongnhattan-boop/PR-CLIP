"""Paper equations (3)-(13): contrastive alignment and positional reconstruction.

This independent implementation takes CLIP token sequences as inputs.  It keeps
the training-only transformer modules out of the retrieval inference path.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from exercises.released_rec_loss import released_rec_loss
from .position_extraction import CrossModalPositionExtractor
from .position_filtering import UnimodalPositionFilter
from .position_reconstruction import CrossModalPositionReconstructor


class PositionalReconstruction(nn.Module):
    def __init__(self, image_width=768, text_width=512, width=512,
                 heads=8, extraction_layers=2, reconstruction_layers=8,
                 lambda_cmc=0.01, lambda_prc=1.0, temperature=0.07,
                 loss_mode="paper"):
        super().__init__()
        self.image_projection = nn.Linear(image_width, width)
        self.text_projection = nn.Linear(text_width, width)
        enc = nn.TransformerEncoderLayer(width, heads, width * 4, batch_first=True, norm_first=True)
        dec = nn.TransformerDecoderLayer(width, heads, width * 4, batch_first=True, norm_first=True)
        self.extraction = CrossModalPositionExtractor(enc, extraction_layers)
        # The official source shares decoder weights for the two directions.
        self.reconstruction = CrossModalPositionReconstructor(dec, reconstruction_layers)
        self.image_filter = UnimodalPositionFilter(width, width)
        self.text_filter = UnimodalPositionFilter(width, width)
        self.lambda_cmc = lambda_cmc
        self.lambda_prc = lambda_prc
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)))
        self.loss_mode = loss_mode

    def forward(self, image_tokens, text_tokens, image_global, text_global,
                text_padding_mask=None):
        if image_tokens.ndim != 3 or text_tokens.ndim != 3:
            raise ValueError("Token embeddings must have shape [batch, tokens, width]")
        image = self.image_projection(image_tokens)
        text = self.text_projection(text_tokens)
        if text_padding_mask is None:
            text_padding_mask = torch.zeros(text.shape[:2], device=text.device, dtype=torch.bool)
        if text_padding_mask.shape != text.shape[:2]:
            raise ValueError("text_padding_mask shape must match text tokens")
        # 1. Cross-modal extraction: each output retains its source token count.
        image_position, text_position = self.extraction(
            image, text, text_padding_mask
        )
        # 2. Unimodal filtering: subtract extracted features before projection.
        image_filtered = self.image_filter(image, image_position)
        text_filtered = self.text_filter(text, text_position)
        # 3. Cross-modal reconstruction: share one decoder in both directions.
        image_reconstructed = self.reconstruction(
            image_filtered, text, memory_padding_mask=text_padding_mask
        )
        text_reconstructed = self.reconstruction(
            text_filtered, image, target_padding_mask=text_padding_mask
        )
        # Equation (12): mean over tokens of squared L2 distance, then mean over batch.
        image_prc = (image_reconstructed - image).square().sum(dim=-1).mean()
        text_error = (text_reconstructed - text).square().sum(dim=-1)
        text_prc = (text_error * (~text_padding_mask)).sum() / (~text_padding_mask).sum().clamp_min(1)
        prc = (image_prc + text_prc) / 2
        image_global = F.normalize(image_global.float(), dim=-1)
        text_global = F.normalize(text_global.float(), dim=-1)
        logits = image_global @ text_global.T * self.logit_scale.exp().clamp(max=100)
        labels = torch.arange(logits.size(0), device=logits.device)
        cmc = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        if self.loss_mode == "released_like":
            loss = released_rec_loss(
                image_reconstructed.float(), text_reconstructed.float(),
                image_global, text_global
            )
        elif self.loss_mode == "paper":
            loss = self.lambda_cmc * cmc + self.lambda_prc * prc
        else:
            raise ValueError(f"Unknown loss mode: {self.loss_mode}")
        return {"loss": loss,
                "cmc": cmc, "prc": prc, "image_prc": image_prc,
                "text_prc": text_prc,
                "image_reconstructed": image_reconstructed,
                "text_reconstructed": text_reconstructed}
