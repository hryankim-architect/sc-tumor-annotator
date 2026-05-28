"""CNV inference + scoring invariants."""

from __future__ import annotations

import numpy as np

from sctumor import cnv
from sctumor.synth import STROMAL_TYPES, generate_cohort


def _ref(cohort):
    return np.isin(cohort.obs["cell_type"].to_numpy(), STROMAL_TYPES)


def test_malignant_cells_score_higher() -> None:
    c = generate_cohort(500, seed=5)
    mat, ch = cnv.infer_cnv(c.expr, c.genes, _ref(c))
    score = cnv.cnv_score(mat, ch)
    mal = c.obs["is_malignant"].to_numpy()
    # the central claim: malignant CNV score exceeds normal CNV score
    assert score[mal].mean() > score[~mal].mean()


def test_length_normalization_changes_score() -> None:
    c = generate_cohort(300, seed=6)
    mat, ch = cnv.infer_cnv(c.expr, c.genes, _ref(c))
    raw = cnv.cnv_score(mat, ch, length_normalized=False)
    norm = cnv.cnv_score(mat, ch, length_normalized=True)
    # both are non-negative and the two aggregations differ in general
    assert (raw >= 0).all() and (norm >= 0).all()
    assert not np.allclose(raw, norm)


def test_malignancy_call_separates() -> None:
    c = generate_cohort(400, seed=8)
    mat, ch = cnv.infer_cnv(c.expr, c.genes, _ref(c))
    score = cnv.cnv_score(mat, ch)
    calls, thr = cnv.malignancy_call(score)
    assert thr > 0
    # epithelial malignant cells should be enriched among positive calls
    mal = c.obs["is_malignant"].to_numpy()
    assert calls[mal].mean() > calls[~mal].mean()


def test_reference_too_small_raises() -> None:
    c = generate_cohort(100, seed=9)
    tiny = np.zeros(c.n_cells, dtype=bool)
    tiny[0] = True
    try:
        cnv.infer_cnv(c.expr, c.genes, tiny)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a one-cell reference")
