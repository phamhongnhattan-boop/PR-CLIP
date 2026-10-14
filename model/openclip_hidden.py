"""Expose ViT-B/32 image tokens using the same path as OpenCLIP's forward."""

import torch


def visual_tokens(visual, images: torch.Tensor) -> torch.Tensor:
    """Return class and patch tokens before CLIP's final pooling/projection.

    OpenCLIP 2.20's VisionTransformer.forward does not accept the
    ``return_hidden`` keyword used in the released PR-CLIP code.
    """
    if visual.input_patchnorm or visual.attn_pool is not None or visual.global_average_pool:
        raise ValueError("This token path supports the standard ViT-B/32 visual backbone")

    tokens = visual.conv1(images)
    tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(tokens.dtype).expand(tokens.shape[0], 1, -1)
    tokens = torch.cat((cls, tokens), dim=1)
    tokens = visual.patch_dropout(tokens + visual.positional_embedding.to(tokens.dtype))
    tokens = visual.ln_pre(tokens)
    tokens = visual.transformer(tokens.permute(1, 0, 2)).permute(1, 0, 2)
    return tokens
