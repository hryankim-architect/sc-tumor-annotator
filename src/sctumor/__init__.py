"""sctumor: capability-portrait POC for cancer scRNA-seq annotation.

A clean-room demonstration of a *class* of capability: tree-based hierarchical
cell-type annotation for tumor single-cell RNA-seq, copy-number-variation (CNV)
inference from expression, a chromosome-length-normalized malignant-cell score,
and cancer subtype / grade prediction within malignant epithelial cells.

The substrate hooks in :mod:`sctumor.audit`, :mod:`sctumor.tracking`, and
:mod:`sctumor.canary` are copy-and-edit, not pip-installed.

This package ships only synthetic, deterministically-generated data. See the
README's honest-scope preamble and ``docs/what-is-out-of-scope.md``.
"""

__version__ = "0.2.0"
