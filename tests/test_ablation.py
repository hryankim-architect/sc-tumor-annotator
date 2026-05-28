"""CNV-feature ablation invariants (v0.2 hard subclonal-CNV regime)."""

from __future__ import annotations

import numpy as np

from sctumor import ablation
from sctumor.synth import generate_malignancy_cohort


def test_hard_cohort_shares_epithelial_baseline() -> None:
    c = generate_malignancy_cohort(800, seed=1)
    # both normal and malignant epithelial exist; neither carries subtype labels
    epi = c.obs["compartment"].to_numpy() == "epithelial"
    assert c.obs.loc[epi, "subtype"].isna().all()
    assert c.obs["is_malignant"].sum() > 0
    assert (~c.obs["is_malignant"] & (c.obs["cell_type"] == "Epithelial_normal")).any()


def test_cnv_scalar_rivals_embedding_and_is_additive() -> None:
    c = generate_malignancy_cohort(2100, seed=0)
    r = ablation.cnv_ablation(c, n_splits=5)

    # a single CNV scalar recovers the malignant call strongly
    assert r["cnv_only_f1_mean"] > 0.85, r["cnv_only_f1_mean"]
    # malignant cells carry a higher CNV score than normal epithelial
    assert r["cnv_score_separation"] > 0, r["cnv_score_separation"]
    # adding the CNV feature to the embedding is non-harmful (slightly additive)
    assert r["embedding_plus_cnv_f1_mean"] >= r["embedding_f1_mean"] - 0.01, r
    # the gradient-boosted tree on the embedding beats the kNN reference-mapping
    # baseline on this sign-heterogeneous regime
    assert r["embedding_f1_mean"] >= r["knn_baseline_f1_mean"] - 1e-9, r


def test_ablation_is_deterministic() -> None:
    c = generate_malignancy_cohort(900, seed=3)
    a = ablation.cnv_ablation(c, n_splits=3)
    b = ablation.cnv_ablation(c, n_splits=3)
    assert np.isclose(a["cnv_only_f1_mean"], b["cnv_only_f1_mean"])
    assert np.isclose(a["embedding_plus_cnv_f1_mean"], b["embedding_plus_cnv_f1_mean"])
