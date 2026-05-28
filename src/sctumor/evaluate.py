"""Evaluation harness: 5-fold CV and independent-cohort generalization.

Two evaluation protocols, mirroring how annotation methods are assessed in the
single-cell literature:

- **5-fold cross-validation** within a pooled multi-patient dataset. Reports
  macro-F1 for cell-type, the malignant call, subtype, and grade.
- **Independent-cohort hold-out**: train on a set of patients, test on a
  disjoint set, which is the more honest estimate of how an annotator
  generalizes to a new sample.

Each protocol scores both the tree-based :class:`HierarchicalAnnotator` and the
kNN :class:`ReferenceMappingBaseline`, so the report is a head-to-head.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from sctumor import cnv
from sctumor.annotate import HierarchicalAnnotator, ReferenceMappingBaseline
from sctumor.synth import STROMAL_TYPES, SyntheticCohort


def _ref_mask(obs) -> np.ndarray:
    """Stromal cells form the presumed-normal CNV reference."""
    return np.isin(obs["cell_type"].to_numpy(), STROMAL_TYPES)


def _macro_f1(y_true, y_pred, mask=None) -> float:
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    if mask is not None:
        yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return float("nan")
    return float(f1_score(yt.astype(str), yp.astype(str), average="macro"))


def _score_split(train: SyntheticCohort, test: SyntheticCohort) -> dict[str, float]:
    # CNV is inferred per dataset against its own stromal reference.
    cnv_tr, ch = cnv.infer_cnv(train.expr, train.genes, _ref_mask(train.obs))
    score_tr = cnv.cnv_score(cnv_tr, ch)
    cnv_te, ch2 = cnv.infer_cnv(test.expr, test.genes, _ref_mask(test.obs))
    score_te = cnv.cnv_score(cnv_te, ch2)

    # tree-based hierarchical model
    model = HierarchicalAnnotator().fit(train.expr, train.obs, score_tr)
    pred = model.predict(test.expr, score_te)

    # baseline
    base = ReferenceMappingBaseline().fit(train.expr, train.obs)
    bpred = base.predict(test.expr)

    epi = test.obs["compartment"].to_numpy() == "epithelial"
    mal = test.obs["is_malignant"].to_numpy()

    return {
        "tree_celltype_f1": _macro_f1(test.obs["cell_type"], pred["cell_type"]),
        "tree_malignant_f1": _macro_f1(
            test.obs["is_malignant"].astype(int), pred["is_malignant"], mask=epi
        ),
        "tree_subtype_f1": _macro_f1(test.obs["subtype"], pred["subtype"], mask=mal),
        "tree_grade_f1": _macro_f1(test.obs["grade"], pred["grade"], mask=mal),
        "baseline_celltype_f1": _macro_f1(test.obs["cell_type"], bpred["cell_type"]),
        "baseline_malignant_f1": _macro_f1(
            test.obs["is_malignant"].astype(int), bpred["is_malignant"], mask=epi
        ),
        "baseline_subtype_f1": _macro_f1(test.obs["subtype"], bpred["subtype"], mask=mal),
        "baseline_grade_f1": _macro_f1(test.obs["grade"], bpred["grade"], mask=mal),
    }


def _subset(cohort: SyntheticCohort, idx: np.ndarray) -> SyntheticCohort:
    return SyntheticCohort(
        expr=cohort.expr[idx],
        genes=cohort.genes,
        obs=cohort.obs.iloc[idx].reset_index(drop=True),
        meta=dict(cohort.meta),
    )


def cross_validate(cohort: SyntheticCohort, *, n_splits: int = 5, seed: int = 42) -> dict:
    """Stratified k-fold CV over a pooled dataset. Returns mean/std per metric."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strat = cohort.obs["cell_type"].to_numpy()
    rows = []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(cohort.n_cells), strat), 1):
        res = _score_split(_subset(cohort, tr), _subset(cohort, te))
        res["fold"] = fold
        rows.append(res)
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c != "fold"]
    summary = {
        c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))}
        for c in metric_cols
    }
    return {"per_fold": df.to_dict(orient="records"), "summary": summary}


def independent_cohort(
    cohort: SyntheticCohort, *, train_patients: list[str], test_patients: list[str]
) -> dict[str, float]:
    """Train on one set of patients, test on a disjoint set."""
    pat = cohort.obs["patient"].to_numpy()
    tr_idx = np.where(np.isin(pat, train_patients))[0]
    te_idx = np.where(np.isin(pat, test_patients))[0]
    return _score_split(_subset(cohort, tr_idx), _subset(cohort, te_idx))
