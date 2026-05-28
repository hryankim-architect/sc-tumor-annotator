"""Deterministic synthetic cancer scRNA-seq generator.

This module fabricates a small, fully synthetic single-cell RNA-seq cohort with
a *known* ground truth so the rest of the pipeline can be exercised offline and
byte-reproducibly. No real patient data is used anywhere in this repository.

Design goals
------------
- **Chromosome-mapped genes.** Each synthetic gene is assigned to a chromosome
  and a position, so the CNV-inference module (:mod:`sctumor.cnv`) can order
  genes along the genome and smooth expression into pseudo-copy-number tracks.
- **Realistic cell-type structure.** Stromal compartments (T-cell, B-cell,
  myeloid, fibroblast, endothelial) plus an epithelial compartment split into
  *normal* and *malignant* cells.
- **CNV-imprinted malignant cells.** Malignant epithelial cells carry
  chromosome-arm-level gains and losses imprinted onto their expression, which
  is exactly the signal a CopyKat / InferCNV-style method recovers.
- **Subtype and grade signatures.** Malignant cells additionally carry a
  receptor subtype (ER+, HER2+, TNBC) driven by marker-gene programs and a
  histologic grade (1/2/3) driven by a proliferation program.

Everything is generated from a single integer seed, so two calls with the same
seed produce identical matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- compartment / label vocabularies -------------------------------------

STROMAL_TYPES = ["T_cell", "B_cell", "Myeloid", "Fibroblast", "Endothelial"]
EPITHELIAL_TYPES = ["Epithelial_normal", "Epithelial_malignant"]
CELL_TYPES = STROMAL_TYPES + EPITHELIAL_TYPES

SUBTYPES = ["ER+", "HER2+", "TNBC"]
GRADES = [1, 2, 3]

# A compact synthetic genome: chromosome -> number of genes on it. The relative
# sizes loosely echo the human autosomes so the length-normalization in the CNV
# score has something non-trivial to correct for.
CHROM_SIZES = {
    "chr1": 90,
    "chr3": 70,
    "chr6": 60,
    "chr8": 55,
    "chr11": 50,
    "chr16": 40,
    "chr17": 45,
    "chr20": 30,
}

# Chromosome arms that malignant cells tend to gain / lose. These are stylized,
# not a claim about any specific tumor: chr8 / chr20 gains and chr16 loss are
# common breast-cancer-like patterns used here purely to create a learnable,
# biologically-plausible CNV signal.
MALIGNANT_GAINS = {"chr8": 0.9, "chr20": 0.7, "chr1": 0.4}
MALIGNANT_LOSSES = {"chr16": -0.8, "chr6": -0.5}


@dataclass
class SyntheticCohort:
    """Container for one synthetic cohort.

    Attributes
    ----------
    expr : np.ndarray
        Cells x genes log-normalized expression matrix.
    genes : pd.DataFrame
        Per-gene annotation: name, chrom, position.
    obs : pd.DataFrame
        Per-cell ground-truth labels: cell_type, compartment, is_malignant,
        subtype (or None), grade (or None), patient.
    """

    expr: np.ndarray
    genes: pd.DataFrame
    obs: pd.DataFrame
    meta: dict = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return self.expr.shape[0]

    @property
    def n_genes(self) -> int:
        return self.expr.shape[1]


def _build_genes() -> pd.DataFrame:
    rows = []
    for chrom, n in CHROM_SIZES.items():
        for i in range(n):
            rows.append({"gene": f"{chrom}_g{i:03d}", "chrom": chrom, "position": i})
    genes = pd.DataFrame(rows)
    genes.index = genes["gene"]
    return genes


# Marker-gene programs. Each program is a set of gene *indices* (resolved later)
# whose mean expression is elevated for a given label. We reserve the first few
# genes on a couple of chromosomes as named markers.
SUBTYPE_MARKERS = {
    "ER+": ["chr6_g000", "chr6_g001"],     # stylized ESR1-like program
    "HER2+": ["chr17_g000", "chr17_g001"],  # stylized ERBB2-like program
    "TNBC": ["chr11_g000", "chr11_g001"],   # stylized basal program
}
CELLTYPE_MARKERS = {
    "T_cell": ["chr1_g000", "chr1_g001"],
    "B_cell": ["chr1_g002", "chr1_g003"],
    "Myeloid": ["chr3_g000", "chr3_g001"],
    "Fibroblast": ["chr3_g002", "chr3_g003"],
    "Endothelial": ["chr8_g000", "chr8_g001"],
    "Epithelial_normal": ["chr11_g010", "chr11_g011"],
    "Epithelial_malignant": ["chr11_g010", "chr11_g011"],  # share epithelial id
}
PROLIFERATION_MARKERS = ["chr20_g000", "chr20_g001", "chr20_g002"]  # grade driver


def _gene_idx(genes: pd.DataFrame, names: list[str]) -> list[int]:
    pos = {g: i for i, g in enumerate(genes["gene"].tolist())}
    return [pos[n] for n in names if n in pos]


def generate_cohort(
    n_cells: int = 900,
    *,
    seed: int = 0,
    patient: str = "P0",
    malignant_fraction: float = 0.30,
) -> SyntheticCohort:
    """Generate one synthetic cohort with known ground truth.

    Parameters
    ----------
    n_cells:
        Number of cells.
    seed:
        RNG seed; identical seeds give identical output.
    patient:
        Patient id stamped on every cell (used by the independent-cohort split).
    malignant_fraction:
        Fraction of cells that are malignant epithelial.
    """
    rng = np.random.default_rng(seed)
    genes = _build_genes()
    n_genes = len(genes)

    # --- assign cells to compartments / types ---
    # Roughly: 55% stromal (spread across 5 types), 15% normal epithelial,
    # the rest malignant epithelial (governed by malignant_fraction).
    n_malignant = int(round(n_cells * malignant_fraction))
    n_normal_epi = int(round(n_cells * 0.15))
    n_stromal = n_cells - n_malignant - n_normal_epi

    types: list[str] = []
    stromal_choices = rng.choice(STROMAL_TYPES, size=n_stromal)
    types.extend(stromal_choices.tolist())
    types.extend(["Epithelial_normal"] * n_normal_epi)
    types.extend(["Epithelial_malignant"] * n_malignant)
    types = np.array(types, dtype=object)
    rng.shuffle(types)

    # --- baseline expression: log-normal-ish background ---
    expr = rng.normal(loc=0.0, scale=0.35, size=(n_cells, n_genes)).astype(np.float64)

    chrom_of = genes["chrom"].to_numpy()

    # --- imprint per-cell-type marker programs ---
    for ct in CELL_TYPES:
        mask = types == ct
        if not mask.any():
            continue
        idx = _gene_idx(genes, CELLTYPE_MARKERS[ct])
        expr[np.ix_(mask, idx)] += 2.2

    # --- per-cell metadata holders ---
    subtype = np.array([None] * n_cells, dtype=object)
    grade = np.array([None] * n_cells, dtype=object)
    is_malignant = types == "Epithelial_malignant"

    # --- imprint CNV on malignant cells ---
    # A gain raises the mean expression of every gene on that chromosome; a loss
    # lowers it. This is the signal the CNV module recovers by genomic smoothing.
    mal_idx = np.where(is_malignant)[0]
    for chrom, amp in {**MALIGNANT_GAINS, **MALIGNANT_LOSSES}.items():
        cols = np.where(chrom_of == chrom)[0]
        # add small per-cell jitter so cells are not identical
        jitter = rng.normal(amp, 0.12, size=(len(mal_idx), 1))
        expr[np.ix_(mal_idx, cols)] += jitter

    # --- assign subtype + grade to malignant cells, imprint their programs ---
    if len(mal_idx) > 0:
        sub_assign = rng.choice(SUBTYPES, size=len(mal_idx), p=[0.45, 0.25, 0.30])
        grade_assign = rng.choice(GRADES, size=len(mal_idx), p=[0.30, 0.40, 0.30])
        subtype[mal_idx] = sub_assign
        grade[mal_idx] = grade_assign

        for st in SUBTYPES:
            sub_mask_local = sub_assign == st
            if not sub_mask_local.any():
                continue
            rows = mal_idx[sub_mask_local]
            idx = _gene_idx(genes, SUBTYPE_MARKERS[st])
            expr[np.ix_(rows, idx)] += 2.6

        # Grade is driven by a proliferation program scaled by grade level.
        prolif = _gene_idx(genes, PROLIFERATION_MARKERS)
        for g in GRADES:
            g_mask_local = grade_assign == g
            if not g_mask_local.any():
                continue
            rows = mal_idx[g_mask_local]
            expr[np.ix_(rows, prolif)] += 0.9 * g

    # clip to non-negative, mimic log1p-normalized counts
    expr = np.clip(expr, 0.0, None)

    obs = pd.DataFrame(
        {
            "cell_type": types,
            "compartment": np.where(
                np.isin(types, EPITHELIAL_TYPES), "epithelial", "stromal"
            ),
            "is_malignant": is_malignant,
            "subtype": subtype,
            "grade": grade,
            "patient": patient,
        }
    )
    obs.index = [f"{patient}_cell{i:05d}" for i in range(n_cells)]

    return SyntheticCohort(
        expr=expr,
        genes=genes,
        obs=obs,
        meta={"seed": seed, "patient": patient, "n_malignant": int(is_malignant.sum())},
    )


def generate_malignancy_cohort(
    n_cells: int = 900,
    *,
    seed: int = 0,
    patient: str = "P0",
    malignant_fraction: float = 0.40,
    cnv_amplitude: float = 0.80,
    cnv_noise: float = 0.30,
    max_altered_chroms: int = 4,
) -> SyntheticCohort:
    """Hard regime: malignant vs normal epithelial differ ONLY by subclonal CNV.

    This cohort is built to make the central claim *measurable*: that an
    expression-derived CNV signal sharpens the normal-vs-malignant call beyond
    what a transcriptomic embedding alone can do.

    The trick is **intratumor heterogeneity**. Each malignant cell independently
    picks a small random subset of chromosomes to alter, each with a random sign
    (gain or loss), at a small per-gene amplitude buried under per-gene noise.
    Normal epithelial cells carry no alteration. Crucially:

    - There is **no non-CNV marker program** separating malignant from normal
      epithelial -- both share the identical epithelial baseline. So a classifier
      cannot cheat via lineage markers.
    - Because the alteration *sign* varies cell to cell, the malignant cells do
      not share a single linear direction in expression space, so PCA / kNN
      reference mapping cannot find a separating axis.
    - But the **magnitude** of genome-coherent deviation -- exactly what the
      chromosome-length-normalized CNV score measures -- is high for any altered
      cell and ~0 for normal cells. So the CNV channel separates them.

    The result is a setting where ablating the CNV feature measurably drops the
    malignant-call F1, which is the demonstration v0.2 adds.
    """
    rng = np.random.default_rng(seed)
    genes = _build_genes()
    n_genes = len(genes)
    chrom_of = genes["chrom"].to_numpy()
    alterable = list(CHROM_SIZES.keys())

    n_malignant = int(round(n_cells * malignant_fraction))
    n_normal_epi = int(round(n_cells * 0.30))
    n_stromal = n_cells - n_malignant - n_normal_epi

    types: list[str] = []
    types.extend(rng.choice(STROMAL_TYPES, size=n_stromal).tolist())
    types.extend(["Epithelial_normal"] * n_normal_epi)
    types.extend(["Epithelial_malignant"] * n_malignant)
    types = np.array(types, dtype=object)
    rng.shuffle(types)

    expr = rng.normal(0.0, 0.35, size=(n_cells, n_genes)).astype(np.float64)

    # stromal + epithelial lineage markers (epithelial markers shared by normal
    # AND malignant -- they are the same lineage)
    for ct in CELL_TYPES:
        mask = types == ct
        if not mask.any():
            continue
        idx = _gene_idx(genes, CELLTYPE_MARKERS[ct])
        expr[np.ix_(mask, idx)] += 2.2

    is_malignant = types == "Epithelial_malignant"
    mal_idx = np.where(is_malignant)[0]

    # subclonal CNV: each malignant cell alters a random subset of chromosomes
    # with random sign, small amplitude, buried under per-gene noise.
    for ci in mal_idx:
        k = int(rng.integers(1, max_altered_chroms + 1))
        chosen = rng.choice(alterable, size=k, replace=False)
        for chrom in chosen:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            cols = np.where(chrom_of == chrom)[0]
            shift = sign * cnv_amplitude + rng.normal(0.0, cnv_noise, size=len(cols))
            expr[ci, cols] += shift

    # add baseline per-gene noise to normal epithelial too, so the only
    # systematic difference is the genome-coherent malignant alteration
    expr = np.clip(expr, 0.0, None)

    obs = pd.DataFrame(
        {
            "cell_type": types,
            "compartment": np.where(
                np.isin(types, EPITHELIAL_TYPES), "epithelial", "stromal"
            ),
            "is_malignant": is_malignant,
            "subtype": np.array([None] * n_cells, dtype=object),
            "grade": np.array([None] * n_cells, dtype=object),
            "patient": patient,
        }
    )
    obs.index = [f"{patient}_cell{i:05d}" for i in range(n_cells)]

    return SyntheticCohort(
        expr=expr,
        genes=genes,
        obs=obs,
        meta={
            "seed": seed,
            "patient": patient,
            "regime": "hard-subclonal-cnv",
            "n_malignant": int(is_malignant.sum()),
        },
    )


def generate_multipatient(
    n_patients: int = 3,
    cells_per_patient: int = 700,
    *,
    seed: int = 0,
) -> SyntheticCohort:
    """Concatenate several single-patient cohorts into one labeled dataset.

    Patient-level seeds are derived from the base seed so the whole dataset is
    reproducible, while each patient gets a distinct (but deterministic) draw.
    """
    cohorts = [
        generate_cohort(
            cells_per_patient,
            seed=seed * 100 + p,
            patient=f"P{p}",
        )
        for p in range(n_patients)
    ]
    expr = np.vstack([c.expr for c in cohorts])
    obs = pd.concat([c.obs for c in cohorts], axis=0)
    genes = cohorts[0].genes
    return SyntheticCohort(
        expr=expr,
        genes=genes,
        obs=obs,
        meta={"seed": seed, "n_patients": n_patients},
    )
