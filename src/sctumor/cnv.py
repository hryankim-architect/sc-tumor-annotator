"""Expression-based copy-number-variation (CNV) inference.

This is a clean-room, from-scratch implementation of the *public* idea behind
InferCNV (Tickle et al., 2019) and CopyKat (Gao et al., 2021): you cannot read
copy number directly from short-read scRNA-seq, but large chromosomal gains and
losses shift the *average* expression of long runs of neighbouring genes. By

1. ordering genes along the genome,
2. centering each cell against a reference of presumed-normal cells, and
3. smoothing the centered expression in a sliding genomic window,

you recover a per-cell pseudo-copy-number track whose deviations from zero flag
malignant cells.

On top of that this module computes a **chromosome-length-normalized CNV
score**: a single scalar per cell that aggregates the magnitude of CNV
deviation while giving every chromosome equal weight, so that long chromosomes
(which contribute more windows) do not dominate the score. That length
de-biasing is the design idea this POC demonstrates; the exact production
parameters used in industry settings are out of scope here.

Nothing in this module is tied to any proprietary dataset or parameter set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _order_by_genome(genes: pd.DataFrame) -> np.ndarray:
    """Return column indices that sort genes by (chrom, position)."""
    chrom_num = genes["chrom"].str.replace("chr", "", regex=False).astype(int)
    order = np.lexsort((genes["position"].to_numpy(), chrom_num.to_numpy()))
    return order


def _rolling_mean_within_chrom(
    mat: np.ndarray, chrom_labels: np.ndarray, window: int
) -> np.ndarray:
    """Sliding-window mean applied independently within each chromosome.

    ``mat`` is cells x genes, already genome-ordered. Windows never straddle a
    chromosome boundary.
    """
    out = np.zeros_like(mat)
    for chrom in pd.unique(chrom_labels):
        cols = np.where(chrom_labels == chrom)[0]
        block = mat[:, cols]
        n = block.shape[1]
        w = min(window, n)
        if w <= 1:
            out[:, cols] = block
            continue
        # cumulative-sum trick for a centered-ish moving average
        csum = np.cumsum(block, axis=1)
        smoothed = np.empty_like(block)
        half = w // 2
        for j in range(n):
            lo = max(0, j - half)
            hi = min(n, j + half + 1)
            total = csum[:, hi - 1] - (csum[:, lo - 1] if lo > 0 else 0.0)
            smoothed[:, j] = total / (hi - lo)
        out[:, cols] = smoothed
    return out


def infer_cnv(
    expr: np.ndarray,
    genes: pd.DataFrame,
    reference_mask: np.ndarray,
    *,
    window: int = 15,
    clip: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer a per-cell CNV matrix by genomic smoothing against a reference.

    Parameters
    ----------
    expr:
        Cells x genes log-normalized expression.
    genes:
        Per-gene annotation with ``chrom`` and ``position`` columns.
    reference_mask:
        Boolean mask over cells marking the presumed-normal reference
        population (e.g. stromal cells) used to center expression.
    window:
        Sliding-window size in genes.
    clip:
        Symmetric clip applied to centered expression before smoothing, to
        damp single-gene outliers (the role of InferCNV's noise control).

    Returns
    -------
    cnv : np.ndarray
        Cells x genes CNV track (genome-ordered columns).
    chrom_order : np.ndarray
        Chromosome label per output column (parallel to ``cnv`` columns).
    """
    if reference_mask.sum() < 2:
        raise ValueError("reference population must contain at least 2 cells")

    order = _order_by_genome(genes)
    ordered = expr[:, order]
    chrom_labels = genes["chrom"].to_numpy()[order]

    # 1. gene-wise center against the reference mean
    ref_mean = ordered[reference_mask].mean(axis=0, keepdims=True)
    centered = ordered - ref_mean

    # 2. clip outliers
    centered = np.clip(centered, -clip, clip)

    # 3. smooth within each chromosome
    cnv = _rolling_mean_within_chrom(centered, chrom_labels, window)
    return cnv, chrom_labels


def cnv_score(
    cnv: np.ndarray,
    chrom_labels: np.ndarray,
    *,
    length_normalized: bool = True,
) -> np.ndarray:
    """Aggregate a CNV matrix into one score per cell.

    The score is the mean absolute CNV deviation. When ``length_normalized`` is
    True (the default and the point of this module), the per-gene deviations are
    first averaged *within* each chromosome and then averaged *across*
    chromosomes with equal weight. That removes the bias whereby chromosomes
    carrying more genes would otherwise dominate a naive genome-wide mean.

    Returns a non-negative score per cell; malignant cells score higher.
    """
    mag = np.abs(cnv)
    if not length_normalized:
        return mag.mean(axis=1)

    per_chrom = []
    for chrom in pd.unique(chrom_labels):
        cols = np.where(chrom_labels == chrom)[0]
        per_chrom.append(mag[:, cols].mean(axis=1))
    # equal-weight average across chromosomes
    return np.mean(np.vstack(per_chrom), axis=0)


def malignancy_call(
    score: np.ndarray, *, threshold: float | None = None
) -> tuple[np.ndarray, float]:
    """Threshold a CNV score into a binary malignant call.

    If no threshold is given, one is chosen by a simple two-component split:
    the midpoint between the score distribution's lower-mode and upper-mode,
    estimated robustly as the mean of the 25th and 75th percentiles. This keeps
    the call deterministic and parameter-light for the demo.
    """
    if threshold is None:
        lo = np.percentile(score, 25)
        hi = np.percentile(score, 75)
        threshold = float((lo + hi) / 2.0)
    return (score >= threshold).astype(int), float(threshold)
