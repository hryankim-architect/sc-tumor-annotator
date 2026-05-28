"""Annotator invariants and tree-vs-baseline comparison."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from sctumor import cnv
from sctumor.annotate import HierarchicalAnnotator, ReferenceMappingBaseline
from sctumor.synth import STROMAL_TYPES, generate_multipatient


def _split():
    d = generate_multipatient(3, 500, seed=4)
    pat = d.obs["patient"].to_numpy()
    tr = np.where(np.isin(pat, ["P0", "P1"]))[0]
    te = np.where(pat == "P2")[0]
    return d, tr, te


def _cnv_score(expr, genes, obs):
    ref = np.isin(obs["cell_type"].to_numpy(), STROMAL_TYPES)
    mat, ch = cnv.infer_cnv(expr, genes, ref)
    return cnv.cnv_score(mat, ch)


def test_tree_celltype_f1_is_high() -> None:
    d, tr, te = _split()
    s_tr = _cnv_score(d.expr[tr], d.genes, d.obs.iloc[tr])
    s_te = _cnv_score(d.expr[te], d.genes, d.obs.iloc[te])
    model = HierarchicalAnnotator().fit(d.expr[tr], d.obs.iloc[tr].reset_index(drop=True), s_tr)
    pred = model.predict(d.expr[te], s_te)
    f1 = f1_score(
        d.obs["cell_type"].to_numpy()[te].astype(str),
        pred["cell_type"].astype(str),
        average="macro",
    )
    assert f1 > 0.80, f1


def test_tree_not_worse_than_baseline_on_malignant_call() -> None:
    d, tr, te = _split()
    s_tr = _cnv_score(d.expr[tr], d.genes, d.obs.iloc[tr])
    s_te = _cnv_score(d.expr[te], d.genes, d.obs.iloc[te])
    obs_tr = d.obs.iloc[tr].reset_index(drop=True)

    tree = HierarchicalAnnotator().fit(d.expr[tr], obs_tr, s_tr)
    tpred = tree.predict(d.expr[te], s_te)
    base = ReferenceMappingBaseline().fit(d.expr[tr], obs_tr)
    bpred = base.predict(d.expr[te])

    epi = d.obs["compartment"].to_numpy()[te] == "epithelial"
    yt = d.obs["is_malignant"].to_numpy()[te][epi].astype(int)
    tf1 = f1_score(yt, tpred["is_malignant"][epi], average="macro")
    bf1 = f1_score(yt, bpred["is_malignant"][epi], average="macro")
    # CNV-informed tree should be at least as good as the CNV-blind baseline
    assert tf1 >= bf1 - 1e-9, (tf1, bf1)
