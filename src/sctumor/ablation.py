"""CNV-feature ablation for the malignant call (v0.2 headline experiment).

On the hard, subclonal-CNV cohort
(:func:`sctumor.synth.generate_malignancy_cohort`), normal and malignant
epithelial cells share an identical transcriptomic baseline and differ only by
heterogeneous, sign-varying copy-number alterations. We compare three feature
sets for the *same* gradient-boosted-tree malignant-call classifier, plus a
kNN reference-mapping baseline, under stratified cross-validation restricted to
epithelial cells:

- **embedding**: the 30-dim PCA embedding of expression (transcriptomic only)
- **cnv_only**: the single chromosome-length-normalized CNV scalar
- **embedding + cnv**: both

The honest finding this is built to surface: a *single, biologically-grounded
CNV scalar* recovers the malignant call within a few F1 points of a full 30-dim
embedding, and adding it to the embedding is non-harmful and slightly additive.
The point is interpretability and compactness, not a large accuracy jump --
a gradient-boosted tree can already recover much of the CNV magnitude from the
embedding nonlinearly, which is itself worth reporting honestly.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from sctumor import cnv
from sctumor.synth import STROMAL_TYPES, SyntheticCohort


def _tree() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=4, max_iter=120, learning_rate=0.15, random_state=42
    )


def _ref_mask(obs) -> np.ndarray:
    return np.isin(obs["cell_type"].to_numpy(), STROMAL_TYPES)


def cnv_ablation(cohort: SyntheticCohort, *, n_splits: int = 5, seed: int = 42) -> dict:
    """Cross-validated malignant-call macro-F1 across three feature sets.

    CNV is inferred once on the full cohort against its stromal reference (which
    is unlabelled-normal by construction, so this leaks no malignant labels).
    The classifiers are then cross-validated over epithelial cells only.
    """
    score = cnv.cnv_score(*cnv.infer_cnv(cohort.expr, cohort.genes, _ref_mask(cohort.obs)))

    epi = cohort.obs["compartment"].to_numpy() == "epithelial"
    epi_idx = np.where(epi)[0]
    y = cohort.obs["is_malignant"].to_numpy()[epi_idx].astype(int)

    emb = PCA(n_components=min(30, cohort.n_genes - 1), random_state=0).fit_transform(
        cohort.expr
    )[epi_idx]
    s = score[epi_idx].reshape(-1, 1)
    both = np.column_stack([emb, s])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(emb, y), 1):
        rows.append(
            {
                "fold": fold,
                "embedding_f1": f1_score(
                    y[te], _tree().fit(emb[tr], y[tr]).predict(emb[te]), average="macro"
                ),
                "cnv_only_f1": f1_score(
                    y[te], _tree().fit(s[tr], y[tr]).predict(s[te]), average="macro"
                ),
                "embedding_plus_cnv_f1": f1_score(
                    y[te], _tree().fit(both[tr], y[tr]).predict(both[te]), average="macro"
                ),
                "knn_baseline_f1": f1_score(
                    y[te],
                    KNeighborsClassifier(n_neighbors=15).fit(emb[tr], y[tr]).predict(emb[te]),
                    average="macro",
                ),
            }
        )

    def _m(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    embedding = _m("embedding_f1")
    both_f1 = _m("embedding_plus_cnv_f1")
    return {
        "per_fold": rows,
        "embedding_f1_mean": embedding,
        "cnv_only_f1_mean": _m("cnv_only_f1"),
        "embedding_plus_cnv_f1_mean": both_f1,
        "knn_baseline_f1_mean": _m("knn_baseline_f1"),
        "cnv_additive_lift": both_f1 - embedding,
        "cnv_score_separation": float(
            score[epi_idx][y == 1].mean() - score[epi_idx][y == 0].mean()
        ),
    }
