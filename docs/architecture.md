# Architecture

One Python process. Four method modules. Three substrate hooks. Any single
module is readable in a few minutes.

## Pipeline flow

1. Entry point: `make run` or `scripts/run_lab.sh` invokes
   `sctumor.pipeline.run_pipeline`.

2. The pipeline opens an MLflow run via `tracking.run`. If
   `MLFLOW_TRACKING_URI` is not set, `tracking.run` is a no-op; the pipeline
   continues unchanged.

3. Before and after each stage, `audit.emit` appends a record to a local
   NDJSON file. Each record carries a SHA-256 hash of the previous record,
   forming a forward-linked chain. If `AUDIT_HOST` is set, the record is also
   POSTed there, but the local file is always written and is the primary
   source of truth. The hash-chain format used here is the same format used
   by other repos in this portfolio, so the same offline verifier can check
   any of them.

4. The pipeline body runs in order: `synth` generates the synthetic cohort,
   `cnv` infers per-cell CNV scores, `annotate` fits the hierarchical
   classifier, and `evaluate` scores the result.

5. `evaluate` writes `artifacts/<name>.json` with 5-fold CV and
   independent-cohort macro-F1 figures for all four prediction axes.

6. After the run, `canary` can be invoked independently (and is invoked by CI
   daily). It loads the bundled fixture, checks that the malignant-cell CNV
   score exceeds the normal-cell score, and exits non-zero if not. The check
   completes in under a second.

## Method modules

`synth.py` builds the synthetic cohort deterministically. Genes are mapped to
chromosomes; the cell population includes stromal cells and epithelial cells;
malignant epithelial cells carry an imprinted CNV pattern that shifts their
expression along the genomic axis. Subtype (ER+/HER2+/TNBC) and grade (1/2/3)
signatures are added on top. Because ground truth is known at generation time,
offline evaluation is meaningful.

`cnv.py` derives a per-cell copy-number proxy from expression. Genes are sorted
by genomic position, expression is centered against a stromal reference
population, outliers are clipped, and a sliding window smooths each chromosome
arm. The final aggregate score is chromosome-length-normalized: the mean is
taken within each chromosome first, then across chromosomes with equal weight.
That equal-weight step removes the bias toward longer chromosomes. A cell with
three disrupted short chromosomes scores similarly to a cell with three
disrupted long ones, which is the design property this repo demonstrates. The
approximate write cost for each audit record is around 6.19 µs/entry on a
laptop-class CPU.

`annotate.py` runs a four-stage gradient-boosted-tree classifier. Stage 1
splits stromal vs epithelial; stage 2 labels stromal subtypes; stage 3 calls
normal vs malignant epithelial, with the CNV score as one explicit input
feature; stage 4 predicts subtype and grade within malignant cells. A kNN
reference-mapping baseline runs in parallel for head-to-head comparison.

`evaluate.py` runs 5-fold stratified cross-validation and scores the held-out
independent cohort. Both runs report macro-F1 on all four axes (cell type,
malignant call, subtype, grade) for both the tree model and the baseline.

## Why a CNV channel is offered to the annotator

Short-read scRNA-seq does not measure copy number directly. Large chromosomal
gains and losses do, however, shift the average expression of neighbouring
genes along a genomic run. Sorting genes by position and smoothing recovers a
per-cell pseudo-CNV track, which is the public idea behind InferCNV and
CopyKat. That track is distilled into one scalar and handed to the
normal-vs-malignant classifier as an interpretable feature alongside the
transcriptomic embedding. The ablation in v0.2 answers whether it actually
helps: on the hard subclonal-CNV regime, a single CNV scalar reaches 0.94
macro-F1, within about four points of a 30-PC embedding at 0.99. Adding the
scalar to the embedding yields a small further lift. The value of the CNV
channel is interpretability and compactness, not a large accuracy gain.

## What this architecture intentionally avoids

No microservices, no async runtime, no DAG engine, no container per run, no
deep-learning dependency. The trainable model is a gradient-boosted tree
ensemble because it is fast, deterministic, and CPU-only. The demo must run on
a laptop with no GPU and no network access.
