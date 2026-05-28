# Roadmap — v0.3: one real public-data demo

**Status:** planned (not yet implemented). v0.1/v0.2 are synthetic-only by design;
v0.3 closes the one conceptual gap a reviewer reliably finds — "does this run on
real data?" — by exercising `sctumor.adapter.from_anndata` on a real public
single-cell cohort.

## Goal

Convert "synthetic POC" into "synthetic POC **+ one real-data demo**" without
changing the architecture. The deliverable is a `scripts/real_data_demo.py` that:

1. loads a small public breast-cancer scRNA-seq subset as an AnnData,
2. maps genes to chromosomes, ingests via `adapter.from_anndata`,
3. infers CNV against a stromal/immune reference,
4. runs the hierarchical annotator + the CNV ablation,
5. reports the same metrics the synthetic pipeline reports.

## Dataset choice (the hard part)

The demo needs a cohort that is (a) public and open-access, (b) small enough to
cache on a laptop, and (c) annotated with `cell_type` and a malignant flag (ideal:
subtype too). Candidates from `data/manifest.yaml`, ranked by fit:

- **Wu et al. 2021 breast atlas** (DOI 10.1038/s41588-021-00911-1) — has
  curated cell types + malignant epithelial calls + subtype metadata. Best fit,
  but the full object is large; subset to 2–3 patients and a few thousand cells.
- **Pal et al. 2021** (DOI 10.15252/embj.2020107333) — normal/preneoplastic/
  tumor states; good for the normal-vs-malignant axis.
- **Peng PDAC 2019** (DOI 10.1038/s41422-019-0195-y) — tumor vs normal epithelial
  by CNV; matches the CNV-score story but a different tissue.

A reliable always-available fallback (`scanpy.datasets.pbmc3k`) exists but has
**no malignant cells**, so it would only smoke-test ingestion + cell-typing, not
the tumor stages. Use it only as a CI-safe import check, not the headline demo.

## Steps

1. `pip install -e ".[singlecell]"` (adds scanpy + anndata).
2. Download one cohort subset; cache under `data/` (gitignored); record URL +
   checksum in `data/manifest.yaml`.
3. Build the gene→chromosome map from a GTF (or scanpy's `var` annotations).
4. Pick the reference population (immune/stromal) for `cnv.infer_cnv`.
5. Run annotate + ablate; write `artifacts/real_demo.json`.
6. Add a `make real-demo` target (guarded so the synthetic `make run` stays the
   default, network-free path).
7. Embed a real-data CNV heatmap next to the synthetic one in the README.

## Honest-scope guardrails

- The synthetic `make run` stays the default, deterministic, network-free demo.
  The real-data demo is opt-in (`make real-demo`) so CI and a fresh clone never
  depend on a download.
- Real-cohort numbers are reported as a single illustrative run, not a benchmark,
  and the README keeps the "capability portrait, not a research result" framing.
- No controlled-access tiers; open-access subsets only.

## Effort

~half a day, dominated by dataset wrangling (annotation harmonization, gene→chrom
mapping), not by the method — the method is unchanged from v0.2. This is why it
is scoped as its own focused session rather than bundled into a quick-win pass.
