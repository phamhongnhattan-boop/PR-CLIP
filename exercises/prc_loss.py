def prc_loss(image_original, image_reconstructed, text_original, text_reconstructed, text_padding_mask=None):
    image_error = (image_original - image_reconstructed).square().sum(dim=-1).mean()
    text_token_error = (text_original - text_reconstructed).square().sum(dim=-1)
    if text_padding_mask is None:
        text_error = text_token_error.mean()
    else:
        real_token_mask = ~text_padding_mask
        text_error = text_token_error[real_token_mask].sum() / real_token_mask.sum().clamp_min(1)
    return (image_error + text_error) / 2
