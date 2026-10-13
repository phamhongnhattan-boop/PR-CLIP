import torch.nn.functional as F

def released_rec_loss(
    image_rec_tokens, text_rec_tokens, image_original, text_original
):
    # [B, N, 512] -> [B, 512]
    image_rec = image_rec_tokens.mean(dim=1)
    text_rec = text_rec_tokens.mean(dim=1)

    image_rec = F.normalize(image_rec, dim=-1)
    text_rec = F.normalize(text_rec, dim=-1)
    image_original = F.normalize(image_original, dim=-1)
    text_original = F.normalize(text_original, dim=-1)

    image_loss = F.mse_loss(image_rec, image_original)
    text_loss = F.mse_loss(text_rec, text_original)
    return image_loss + text_loss
if __name__ == "__main__":
    import torch

    image_rec = torch.tensor([[[1., 0.], [1., 0.]]])
    text_rec = torch.tensor([[[1., 0.], [1., 0.]]])
    image_original = torch.tensor([[1., 0.]])
    text_original = torch.tensor([[0., 1.]])

    result = released_rec_loss(
        image_rec, text_rec, image_original, text_original
    )
    print(result.item())