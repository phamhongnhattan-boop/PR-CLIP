import torch
import torch.nn.functional as F


def cmc_loss(image_features, text_features, temperature=0.07):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = image_features @ text_features.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_image_to_text = F.cross_entropy(logits, labels)
    loss_text_to_image = F.cross_entropy(logits.T, labels)
    return (loss_image_to_text + loss_text_to_image) / 2
