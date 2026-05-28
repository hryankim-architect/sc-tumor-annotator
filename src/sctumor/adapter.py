"""Real-data adapter: bridge an AnnData object into the sctumor pipeline.

The demo runs on synthetic data, but the method is written against a real-data
shape. This adapter converts a Scanpy / AnnData object (and a gene -> chromosome
map) into the (expr, genes, obs) triple the rest of the package consumes. It is
optional: it imports ``anndata`` lazily, so the core demo never requires it.

Typical use on a real cohort::

    import scanpy as sc
    adata = sc.read_h5ad("cohort.h5ad")          # cells x genes, log-normalized
    cohort = adapter.from_anndata(
        adata,
        gene_chrom=my_gene_to_chrom_dict,         # {"ESR1": "chr6", ...}
        gene_position=my_gene_to_position_dict,   # genomic order within chrom
        celltype_key="cell_type",
        malignant_key="is_malignant",
    )
    # then: cnv.infer_cnv(...), HierarchicalAnnotator().fit(...), etc.

Genes without a chromosome mapping are dropped (CNV inference needs genomic
position). The chromosome label is normalized to the ``chrN`` form this package
uses internally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sctumor.synth import SyntheticCohort


def _normalize_chrom(c: str) -> str:
    c = str(c).strip()
    if c.lower().startswith("chr"):
        return "chr" + c[3:]
    return "chr" + c


def from_anndata(
    adata,
    *,
    gene_chrom: dict[str, str],
    gene_position: dict[str, int] | None = None,
    celltype_key: str | None = None,
    malignant_key: str | None = None,
) -> SyntheticCohort:
    """Build a :class:`SyntheticCohort` view over a real AnnData object.

    Parameters
    ----------
    adata:
        AnnData with ``.X`` as a cells x genes (log-normalized) matrix and
        ``.var_names`` as gene symbols.
    gene_chrom:
        Mapping gene symbol -> chromosome. Genes absent here are dropped.
    gene_position:
        Optional mapping gene symbol -> integer genomic position. When omitted,
        genes are positioned by their order of appearance within each chromosome.
    celltype_key, malignant_key:
        Optional ``adata.obs`` column names to copy into the cohort labels.
    """
    try:
        import anndata  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only with the extra
        raise ImportError(
            "from_anndata requires the 'singlecell' extra: pip install -e '.[singlecell]'"
        ) from exc

    var_names = list(adata.var_names)
    keep = [g for g in var_names if g in gene_chrom]
    if not keep:
        raise ValueError("no genes in adata.var_names matched gene_chrom")

    col_idx = [var_names.index(g) for g in keep]
    x = adata.X
    expr = np.asarray(x[:, col_idx].todense() if hasattr(x, "todense") else x[:, col_idx],
                      dtype=np.float64)

    chroms = [_normalize_chrom(gene_chrom[g]) for g in keep]
    if gene_position is not None:
        positions = [int(gene_position.get(g, 0)) for g in keep]
    else:
        positions = []
        seen: dict[str, int] = {}
        for ch in chroms:
            positions.append(seen.get(ch, 0))
            seen[ch] = seen.get(ch, 0) + 1

    genes = pd.DataFrame({"gene": keep, "chrom": chroms, "position": positions})
    genes.index = genes["gene"]
    # sort columns by genome order so downstream code sees ordered genes
    order = np.lexsort((genes["position"].to_numpy(),
                        genes["chrom"].str.replace("chr", "", regex=False).to_numpy()))
    genes = genes.iloc[order].reset_index(drop=True)
    genes.index = genes["gene"]
    expr = expr[:, order]

    obs_cols = {}
    if celltype_key and celltype_key in adata.obs:
        obs_cols["cell_type"] = adata.obs[celltype_key].to_numpy()
    if malignant_key and malignant_key in adata.obs:
        obs_cols["is_malignant"] = adata.obs[malignant_key].to_numpy().astype(bool)
    obs = pd.DataFrame(obs_cols, index=list(adata.obs_names))

    return SyntheticCohort(expr=expr, genes=genes, obs=obs,
                           meta={"source": "anndata", "n_genes_mapped": len(keep)})
