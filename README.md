# DCTG-VMamba

Research implementation for the manuscript **A Dataset-Conditioned Visual State-Space Model with Clinical Taxonomy and Mask Guidance for Multi-Dataset Cystoscopic Image Classification**.

DCTG-VMamba is a visual state-space model for learning from multiple cystoscopic image datasets. It combines information from CystoDS, EBTC, and Kaggle WLI while retaining a separate classification head for each dataset's fine-grained labels.

## Model overview

A VMamba backbone extracts multi-scale image features. A predicted soft mask separates region-of-interest and surrounding-context representations. Dataset-conditioned gating combines these local representations with a global image representation.

A shared research taxonomy connects related labels across datasets. Available reference masks provide auxiliary supervision during training, while inference uses predicted masks. The framework supports source-specific classification and a shared neoplastic endpoint derived from the trained source heads.

## Repository contents

The Python implementation includes the model architecture, training objectives, data loading, optimization, checkpoint handling, and image prediction. `train.py` and `predict.py` provide the training and prediction entrypoints; supporting modules are organized under `bladder_project/`.

The implementation uses PyTorch and [VMamba](https://github.com/MzeroMiko/VMamba). Python dependency versions are listed in `requirements.txt`.

## Recorded results

[Recorded experiment summaries](results/README.md) include model metrics, training times, per-class correct and incorrect counts, and integer confusion matrices. Original and major-revision results are listed separately, with their protocol and evaluation scope retained.
