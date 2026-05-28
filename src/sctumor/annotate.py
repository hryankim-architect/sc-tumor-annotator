"""Tree-based hierarchical annotator for cancer scRNA-seq.

The capability this demonstrates is a *trainable, tree-based* annotator (as
opposed to anchor-based reference mapping or a fixed probabilistic model). The
annotation is hierarchical, mirroring how a pathologist-style workflow narrows
down a label:

    1. compartment        : stromal vs epithelial
    2. stromal subtype    : T / B / Myeloid / Fibroblast / Endothelial
    3. malignant call     : within epithelial, normal vs malignant
                            (CNV score is an input feature here)
    4. subtype + grade    : within malignant, ER+/HER2+/TNBC and grade 1/2/3

Each stage is an independent gradient-boosted tree classifier. For the malignant
call the CNV score from :mod:`sctumor.cnv` is available as an optional extra
feature; the ablation in :mod:`sctumor.ablation` measures what it actually
contributes rather than assuming it helps.

A k-nearest-neighbour reference-mapping baseline is included for comparison so
the evaluation harness can report a head-to-head F1, the way the single-cell
literature compares trainable models against Scanpy-ingest-style baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sctumor.synth import EPITHELIAL_TYPES


def _tree() -> HistGradientBoostingClassifier:
    # Small, fast, deterministic. random_state pins the demo.
    return HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=120,
        learning_rate=0.15,
        random_state=42,
    )


@dataclass
class HierarchicalAnnotator:
    """Fit/predict the four-stage hierarchical annotation."""

    n_pca: int = 30

    def __post_init__(self) -> None:
        self._pca: PCA | None = None
        self._compartment = _tree()
        self._stromal = _tree()
        self._malignant = _tree()
        self._subtype = _tree()
        self._grade = _tree()

    # --- feature embedding -------------------------------------------------
    def _embed(self, expr: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self._pca = PCA(n_components=min(self.n_pca, expr.shape[1] - 1),
                            random_state=0)
            return self._pca.fit_transform(expr)
        assert self._pca is not None
        return self._pca.transform(expr)

    # --- training ----------------------------------------------------------
    def fit(
        self,
        expr: np.ndarray,
        obs,
        cnv_score: np.ndarray,
    ) -> HierarchicalAnnotator:
        x = self._embed(expr, fit=True)

        compartment = obs["compartment"].to_numpy()
        self._compartment.fit(x, compartment)

        strom = compartment == "stromal"
        if strom.sum() > 0:
            self._stromal.fit(x[strom], obs["cell_type"].to_numpy()[strom])

        epi = compartment == "epithelial"
        if epi.sum() > 0:
            x_epi = np.column_stack([x[epi], cnv_score[epi]])
            self._malignant.fit(x_epi, obs["is_malignant"].to_numpy()[epi].astype(int))

        mal = obs["is_malignant"].to_numpy()
        if mal.sum() > 0:
            self._subtype.fit(x[mal], obs["subtype"].to_numpy()[mal])
            self._grade.fit(x[mal], obs["grade"].to_numpy()[mal].astype(int))
        return self

    # --- prediction --------------------------------------------------------
    def predict(self, expr: np.ndarray, cnv_score: np.ndarray) -> dict[str, np.ndarray]:
        x = self._embed(expr, fit=False)
        n = x.shape[0]

        compartment = self._compartment.predict(x)
        cell_type = np.array(["unknown"] * n, dtype=object)

        strom = compartment == "stromal"
        if strom.any():
            cell_type[strom] = self._stromal.predict(x[strom])

        epi = compartment == "epithelial"
        is_malignant = np.zeros(n, dtype=int)
        subtype = np.array([None] * n, dtype=object)
        grade = np.array([None] * n, dtype=object)

        if epi.any():
            x_epi = np.column_stack([x[epi], cnv_score[epi]])
            mal_pred = self._malignant.predict(x_epi)
            is_malignant[epi] = mal_pred
            epi_idx = np.where(epi)[0]
            for local, gi in enumerate(epi_idx):
                cell_type[gi] = (
                    "Epithelial_malignant" if mal_pred[local] == 1
                    else "Epithelial_normal"
                )

        mal_mask = is_malignant == 1
        if mal_mask.any():
            subtype[mal_mask] = self._subtype.predict(x[mal_mask])
            grade[mal_mask] = self._grade.predict(x[mal_mask])

        return {
            "compartment": compartment,
            "cell_type": cell_type,
            "is_malignant": is_malignant,
            "subtype": subtype,
            "grade": grade,
        }


@dataclass
class ReferenceMappingBaseline:
    """kNN-on-PCA reference mapping, the head-to-head baseline.

    This stands in for anchor-based / ingest-style label transfer: embed with
    PCA, then assign each query cell the majority label of its nearest training
    neighbours. It does not use the CNV score; it is the CNV-blind reference
    point the evaluation harness compares against, with no assumption about
    which method comes out ahead.
    """

    n_pca: int = 30
    k: int = 15

    def __post_init__(self) -> None:
        self._pca: PCA | None = None
        self._ct = KNeighborsClassifier(n_neighbors=self.k)
        self._sub = KNeighborsClassifier(n_neighbors=self.k)
        self._grade = KNeighborsClassifier(n_neighbors=self.k)

    def fit(self, expr: np.ndarray, obs) -> ReferenceMappingBaseline:
        self._pca = PCA(n_components=min(self.n_pca, expr.shape[1] - 1), random_state=0)
        x = self._pca.fit_transform(expr)
        self._ct.fit(x, obs["cell_type"].to_numpy())
        mal = obs["is_malignant"].to_numpy()
        if mal.sum() >= self.k:
            self._sub.fit(x[mal], obs["subtype"].to_numpy()[mal])
            self._grade.fit(x[mal], obs["grade"].to_numpy()[mal].astype(int))
            self._has_mal = True
        else:
            self._has_mal = False
        return self

    def predict(self, expr: np.ndarray) -> dict[str, np.ndarray]:
        assert self._pca is not None
        x = self._pca.transform(expr)
        cell_type = self._ct.predict(x)
        is_malignant = np.isin(cell_type, ["Epithelial_malignant"]).astype(int)
        n = x.shape[0]
        subtype = np.array([None] * n, dtype=object)
        grade = np.array([None] * n, dtype=object)
        mal_mask = is_malignant == 1
        if self._has_mal and mal_mask.any():
            subtype[mal_mask] = self._sub.predict(x[mal_mask])
            grade[mal_mask] = self._grade.predict(x[mal_mask])
        return {
            "cell_type": cell_type,
            "is_malignant": is_malignant,
            "subtype": subtype,
            "grade": grade,
        }


def epithelial_mask(obs) -> np.ndarray:
    return np.isin(obs["cell_type"].to_numpy(), EPITHELIAL_TYPES)
