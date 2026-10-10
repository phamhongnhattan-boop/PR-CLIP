import csv

import numpy as np
import pytest
import torch

from repro.data import load_retrieval_csv
from repro.metrics import retrieval_metrics
from repro.prclip import PositionalReconstruction


def test_recall_accepts_any_of_five_captions():
    images = np.eye(2, dtype=np.float32)
    texts = np.repeat(images, 5, axis=0)
    target = [0] * 5 + [1] * 5
    result = retrieval_metrics(images, texts, target)
    assert result["t2i_R@1"] == 100
    assert result["i2t_R@1"] == 100
    assert result["mR"] == 100


def test_reconstruction_objective_has_gradients():
    torch.manual_seed(3)
    objective = PositionalReconstruction(image_width=8, text_width=8, width=8,
                                          heads=2, extraction_layers=1,
                                          reconstruction_layers=1)
    objective.eval()
    image = torch.randn(2, 4, 8, requires_grad=True)
    text = torch.randn(2, 3, 8, requires_grad=True)
    image_global = torch.randn(2, 8, requires_grad=True)
    text_global = torch.randn(2, 8, requires_grad=True)
    padding = torch.tensor([[False, False, True], [False, False, False]])
    loss = objective(image, text, image_global, text_global, padding)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    for tensor in (image, text, image_global, text_global):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    assert objective.logit_scale.grad is not None


def test_csv_groups_captions_and_checks_images(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "a.png").write_bytes(b"x")
    (image_dir / "b.png").write_bytes(b"x")
    csv_path = tmp_path / "pairs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerows([["filename", "title"], ["a.png", "first"],
                          ["b.png", "second"], ["a.png", "third"]])
    paths, captions, mapping = load_retrieval_csv(csv_path, image_dir)
    assert [path.name for path in paths] == ["a.png", "b.png"]
    assert captions == ["first", "third", "second"]
    assert mapping == [0, 0, 1]
    (image_dir / "b.png").unlink()
    with pytest.raises(FileNotFoundError):
        load_retrieval_csv(csv_path, image_dir)
