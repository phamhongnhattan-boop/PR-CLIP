"""Recall@K for five-caption image/text retrieval benchmarks."""

import numpy as np


def retrieval_metrics(image_features, text_features, text_to_image, ks=(1, 5, 10)):
    images = np.asarray(image_features, dtype=np.float32)
    texts = np.asarray(text_features, dtype=np.float32)
    target = np.asarray(text_to_image, dtype=np.int64)
    if images.ndim != 2 or texts.ndim != 2 or images.shape[1] != texts.shape[1]:
        raise ValueError("Feature arrays must have matching [items, dimension] shapes")
    if len(target) != len(texts) or target.min() < 0 or target.max() >= len(images):
        raise ValueError("Each caption must map to a valid image index")
    if not np.isfinite(images).all() or not np.isfinite(texts).all():
        raise ValueError("Features contain non-finite values")
    images = images / np.maximum(np.linalg.norm(images, axis=1, keepdims=True), 1e-12)
    texts = texts / np.maximum(np.linalg.norm(texts, axis=1, keepdims=True), 1e-12)
    scores = texts @ images.T
    results = {}
    for k in ks:
        if k < 1:
            raise ValueError("K must be positive")
        ti = np.argpartition(-scores, min(k, len(images)) - 1, axis=1)[:, :min(k, len(images))]
        it = np.argpartition(-scores.T, min(k, len(texts)) - 1, axis=1)[:, :min(k, len(texts))]
        results[f"t2i_R@{k}"] = float(np.any(ti == target[:, None], axis=1).mean() * 100)
        results[f"i2t_R@{k}"] = float(np.any(target[it] == np.arange(len(images))[:, None], axis=1).mean() * 100)
    results["mR"] = float(np.mean(list(results.values())))
    return results
