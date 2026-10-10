# PR-CLIP reproduction report

## Scope and status

Source: Guan et al., *PR-CLIP: Cross-Modal Positional Reconstruction for Remote Sensing Image–Text Retrieval*, Remote Sensing 17(13), 2117 (2025), [DOI](https://doi.org/10.3390/rs17132117). The repository is based on the [official code](https://github.com/ADMIS-TONGJI/PR-CLIP) and retains its history. The paper PDF supplied for this task was treated as research data, not as instructions.

**Result: implementation and real-data evaluation completed; the published PR-CLIP scores were not reproduced.** The official README's checkpoint URL resolves to `images.tar.gz`, not model weights. The machine used here has one RTX 4060 Laptop GPU with 8 GB VRAM; the paper reports two RTX 4090 GPUs. We ran a two-step end-to-end training smoke test with the paper's default 2 extraction and 8 reconstruction layers. It cannot estimate a converged score.

| Dataset | Test images / captions | Paper PR-CLIP mR | OpenAI CLIP ViT-B/32 baseline mR | Two-step PR-CLIP smoke mR |
| --- | ---: | ---: | ---: | ---: |
| RSITMD | 452 / 2,260 | 52.41% | 24.2330% | 24.2330% |
| RSICD | 1,093 / 5,465 | 39.23% | 15.6908% | not evaluated |

The baseline and smoke numbers were measured on the **complete author-provided test CSVs** and released images. The baseline is not PR-CLIP; the smoke checkpoint has seen only 16 sampled RET-2 pairs and 2 optimizer steps. The equal RSITMD values at four-decimal precision do not establish equivalence of the models.

The detailed six Recall@K values and machine-readable metadata are in `results/rsitmd_openai_baseline.json`, `results/rsicd_openai_baseline.json`, and `results/rsitmd_smoke_eval.json`. The step-by-step training loss is in `results/ret2_smoke_train.jsonl`.

## What the implementation contains

`repro/prclip.py` implements equations (3)-(13): CLIP-style bidirectional InfoNCE, projected image/text tokens, a two-layer joint self-attention encoder, subtraction and linear filtering, shared eight-layer cross-attention decoder, and bidirectional token-wise squared-L2 reconstruction. `repro/openclip_tokens.py` exposes ViT-B/32 image and text token sequences from OpenCLIP. At retrieval time, `repro/evaluate.py` uses only the CLIP encoders, as the paper describes.

As a direct consistency check, the global image and text vectors returned by `encode_with_tokens` matched OpenCLIP's native `encode_image` and `encode_text` exactly (maximum absolute difference 0.0 for a 224×224 test tensor and one caption). The token tensor shapes were `[1,50,768]` and `[1,77,512]`.

The default loss is the **literal paper interpretation**. It averages squared L2 error across valid tokens (including the image CLS and text EOT tokens) and directions. This choice is explicit because the paper does not give executable masking and token-boundary details. The contrastive temperature is learned from a 0.07 initialization.

## Data audit

The [official image archive](https://drive.google.com/file/d/15htMe2iE4kVujjWURxfCxZXk8vAAetoT/view) was downloaded (1,461,515,987 bytes; SHA-256 `f61d5d59e5e0696cb88685c34c2ee58961d68174e76a6988c9a83b2bb38f11d5`). All 65,460 training rows and both full test CSVs resolve to existing images. RET-2 contains 13,092 distinct filenames, each with five captions. No train/test filename or exact-file-content duplicate was found. This check cannot rule out near duplicates or re-encoded copies.

`dataset/test/rsicd_val.csv` and `dataset/test/rsicd_test.csv` have identical SHA-256 digests and 1,093 overlapping image filenames. `dataset/test/rsitmd_val.csv` and `dataset/test/rsitmd_test.csv` have no filename overlap. See `results/dataset_audit.json`.

## Reproducibility discrepancies in the released materials

| Item | Paper | Official repository | Consequence |
| --- | --- | --- | --- |
| RSICD learning rate | 5e-5 | `train_rsicd.sh`: 5e-6 | Different optimization trajectory. |
| RSITMD learning rate | 5e-6 | `train_rsitmd.sh`: 5e-5 | Different optimization trajectory. |
| Loss | `0.01 × CMC + 1 ×` token-level PRC | `model/model.py` computes MSE on normalized pooled reconstructed features; `training/train.py` combines `(1.5 × InfoNCE + MSE) / 4` | The public code and printed equations train different objectives. |
| Model selection | Paper describes train/validation/test | Official train scripts pass `*_test.csv` to `--retrieval-data`; `training/main.py` selects `best.pt` by retrieval mean recall on that file | The released script selects a checkpoint using the test set. This does not prove how the paper's reported checkpoints were selected. |
| RSICD validation | Separate validation implied | `rsicd_val.csv` is byte-for-byte identical to `rsicd_test.csv` | Validation cannot be used for unbiased RSICD model selection without a new split. |
| Checkpoint link | Pretrained weights needed for exact evaluation | README points to the same 1.4 GB image archive for both images and checkpoints | Exact published-checkpoint evaluation is unavailable from the linked file. |
| Standalone retrieval script | Test command in README | `retrieval.py` uses undefined `DATASET_PATH`, `model_name`, and `dataset_name`, and passes a CSV to a JSON loader | The README command cannot run unmodified. |
| Declared dependencies | README lists the packages needed to run | `model/model.py` imports `cn_clip` and `sentence_transformers` unconditionally, although the README requirements omit them | Importing the upstream model fails in a clean environment; the independent `repro` package avoids these optional dependencies. |

These are observations about the released files, not claims about author intent. A close reproduction requires the actual pretrained RET-2 weights, the final PR-CLIP checkpoint, the exact loss and learning-rate configuration used, and an independent validation split.

## Run it again

Use Python 3.12 and install `requirements-repro.txt`. On NVIDIA Windows, install the PyTorch 2.5.1 CUDA 12.1 wheels first from the [official PyTorch wheel index](https://download.pytorch.org/whl/cu121), then install the remaining requirements. This run used PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, OpenCLIP 2.20.0, NumPy 2.5.3, and Pillow 12.3.0.

From the repository root:

```powershell
python -m repro.prepare_data --directory images
python -m repro.audit --dataset dataset --images images/image_all --output results/dataset_audit.json --hash-images
python -m pytest -q tests/test_repro.py
python -m repro.evaluate --csv dataset/test/rsitmd_test.csv --images images/image_all --output results/rsitmd_openai_baseline.json
python -m repro.evaluate --csv dataset/test/rsicd_test.csv --images images/image_all --output results/rsicd_openai_baseline.json
```

The paper-equation training defaults in `repro.train` are RSITMD: 7 epochs, batch 100, learning rate 5e-6, 100 warm-up steps, weight decay 0.5, gradient clip 50, two extraction layers, eight reconstruction layers, λCMC=0.01 and λPRC=1. Use `--epochs 20 --batch-size 160 --lr 5e-5` for the RSICD configuration. Both use the same RET-2 training CSV. Training is single-process and needs enough GPU memory; a smaller `--batch-size` changes InfoNCE negatives and is **not** equivalent to the paper experiment.

If a legitimate OpenCLIP-format RET-2 weight file becomes available, pass it through `--initial-checkpoint path/to/RET-2.pt`. The command loads it strictly and fails on key mismatches; it does not guess a mapping. The author's final PR-CLIP checkpoint may use its ITRA-specific key names and is not guaranteed to load through `repro.evaluate` without an explicit, audited conversion.

The resource-limited check performed here was:

```powershell
python -m repro.train --csv dataset/train/ret2.csv --images images/image_all --output results/ret2_smoke_checkpoint.pt --log results/ret2_smoke_train.jsonl --max-pairs 16 --max-steps 2 --batch-size 2 --epochs 1 --warmup 1
python -m repro.evaluate --csv dataset/test/rsitmd_test.csv --images images/image_all --checkpoint results/ret2_smoke_checkpoint.pt --output results/rsitmd_smoke_eval.json
```

The smoke checkpoint is excluded from Git because it is 770 MB. It is present in this delivered local folder. Re-running the two commands recreates it and its evaluation report.
