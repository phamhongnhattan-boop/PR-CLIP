# Contributions and provenance

This repository started as a clone of the [authors' PR-CLIP repository](https://github.com/ADMIS-TONGJI/PR-CLIP). Its original Git commits and attribution are preserved. The research and original model belong to Jihong Guan, Yulou Shu, Wengen Li, Zihan Song, and Yichao Zhang. The upstream repository states that it builds on ITRA.

The new commits on the `reproduction` branch are authored as **Codex <codex@local>** to distinguish this implementation and audit from the paper authors' work and from the machine owner's identity. They add:

1. Independent PyTorch implementation of the paper's bidirectional CMC and token-level PRC objectives, cross-modal extraction, filtering, and reconstruction modules.
2. Data download, dataset audit, retrieval metric, training, and evaluation commands.
3. Real-dataset measurements, tests, a reproducibility report, and a Vietnamese learning guide.

The changes are local to this Git repository. No pull request or remote GitHub repository has been created. The 770 MB smoke checkpoint is present in `results/` locally and excluded from Git history; the training script regenerates it.
