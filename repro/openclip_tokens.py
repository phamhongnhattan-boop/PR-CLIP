"""Expose ViT-B/32 and CLIP text tokens without changing OpenCLIP weights."""

import torch


def encode_with_tokens(clip, images, token_ids):
    visual = clip.visual
    if visual.attn_pool is not None or visual.global_average_pool or visual.input_patchnorm:
        raise ValueError("This extractor supports standard OpenCLIP ViT-B/32 only")
    image = visual.conv1(images)
    image = image.reshape(image.shape[0], image.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(image.dtype).expand(image.shape[0], 1, -1)
    image = torch.cat([cls, image], dim=1)
    image = visual.ln_pre(visual.patch_dropout(image + visual.positional_embedding.to(image.dtype)))
    image = visual.transformer(image.permute(1, 0, 2)).permute(1, 0, 2)
    image_global = visual.ln_post(image[:, 0]) @ visual.proj

    cast_dtype = clip.transformer.get_cast_dtype()
    text = clip.token_embedding(token_ids).to(cast_dtype)
    text = text + clip.positional_embedding.to(cast_dtype)
    text = clip.transformer(text.permute(1, 0, 2), attn_mask=clip.attn_mask).permute(1, 0, 2)
    text = clip.ln_final(text)
    eot_positions = token_ids.argmax(dim=-1)
    text_global = text[torch.arange(text.shape[0], device=text.device), eot_positions] @ clip.text_projection
    positions = torch.arange(token_ids.shape[1], device=token_ids.device)[None, :]
    text_padding_mask = positions > eot_positions[:, None]
    return image.float(), text.float(), image_global.float(), text_global.float(), text_padding_mask
