"""Checks for the extracted module inside the released ItraModel path."""

import torch
from torch import nn

from model.openclip_hidden import visual_tokens
from model.paper_objective import paper_training_loss, positional_reconstruction_loss
from model.position_extraction import extract_positional_residuals


class DoubleTokens(nn.Module):
    def forward(self, tokens):
        return 2 * tokens


def test_extraction_splits_and_subtracts_matching_positions():
    image = torch.randn(2, 4, 512, requires_grad=True)
    text = torch.randn(2, 3, 512, requires_grad=True)
    image_out, text_out = extract_positional_residuals(
        image, text, DoubleTokens(), nn.Identity(), nn.Identity()
    )
    assert image_out.shape == (2, 4, 512)
    assert text_out.shape == (2, 3, 512)
    torch.testing.assert_close(image_out, -image)
    torch.testing.assert_close(text_out, -text)
    (image_out.sum() + text_out.sum()).backward()
    torch.testing.assert_close(image.grad, -torch.ones_like(image))
    torch.testing.assert_close(text.grad, -torch.ones_like(text))


def test_openclip_visual_tokens_reproduce_global_visual_output():
    import open_clip

    visual = open_clip.create_model("ViT-B-32", pretrained=None).visual.eval()
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        tokens = visual_tokens(visual, images)
        pooled = visual.ln_post(tokens[:, 0]) @ visual.proj
        expected = visual(images)
    assert tokens.shape == (1, 50, 768)
    torch.testing.assert_close(pooled, expected, rtol=1e-5, atol=1e-5)


def test_paper_reconstruction_ignores_text_padding():
    image = torch.tensor([[[1.0], [2.0]]])
    image_hat = torch.tensor([[[1.0], [4.0]]])
    text = torch.tensor([[[0.0], [0.0]]])
    text_hat = torch.tensor([[[1.0], [100.0]]])
    padding = torch.tensor([[False, True]])
    loss = positional_reconstruction_loss(image_hat, text_hat, image, text, padding)
    # Image: (0 + 4) / 2 = 2; valid text: 1; direction average = 1.5.
    torch.testing.assert_close(loss, torch.tensor(1.5))


def test_paper_total_loss_weights_cmc_and_prc():
    total = paper_training_loss(torch.tensor(2.0), torch.tensor(3.0))
    torch.testing.assert_close(total, torch.tensor(3.02))
