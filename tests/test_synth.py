"""Synthetic-data generator invariants."""

from __future__ import annotations

import numpy as np

from sctumor.synth import (
    CELL_TYPES,
    SUBTYPES,
    generate_cohort,
    generate_multipatient,
)


def test_cohort_is_deterministic() -> None:
    a = generate_cohort(300, seed=3)
    b = generate_cohort(300, seed=3)
    assert np.array_equal(a.expr, b.expr)
    assert a.obs["cell_type"].tolist() == b.obs["cell_type"].tolist()


def test_cohort_shapes_and_labels() -> None:
    c = generate_cohort(400, seed=1)
    assert c.expr.shape[0] == 400
    assert c.n_genes == c.genes.shape[0]
    assert set(c.obs["cell_type"].unique()).issubset(set(CELL_TYPES))
    # malignant cells carry subtype + grade; non-malignant do not
    mal = c.obs["is_malignant"].to_numpy()
    assert c.obs.loc[mal, "subtype"].isin(SUBTYPES).all()
    assert c.obs.loc[~mal, "subtype"].isna().all()


def test_multipatient_concatenates() -> None:
    d = generate_multipatient(3, 200, seed=2)
    assert d.n_cells == 600
    assert set(d.obs["patient"].unique()) == {"P0", "P1", "P2"}
