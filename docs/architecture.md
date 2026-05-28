# Architecture

A deliberately small architecture: one Python process, four method modules, and
three substrate hooks. A reviewer should be able to read any one module in a
couple of minutes.

## Control flow

```
                make run / scripts/run_lab.sh
                          │
                          ▼
              sctumor.pipeline.run_pipeline
                          │
        ┌─────────────────┼──────────────────────────────┐
        ▼                 ▼                               ▼
  audit.emit         tracking.run                       body
 (NDJSON +         (MLflow active run,            synth → cnv → annotate
  optional POST)    no-op if unset)                  → evaluate
        │                 │                               │
        └─────────────────┴───────────────────────────────┘
                          │
                          ▼
              artifacts/<name>.json  (CV + independent-cohort metrics)
```

## Method modules

| Module | Responsibility |
|---|---|
| `synth.py` | Deterministic synthetic cancer scRNA-seq: chromosome-mapped genes, stromal + epithelial compartments, CNV-imprinted malignant cells, subtype/grade signatures. The ground truth is known, which is what makes offline evaluation meaningful. |
| `cnv.py` | Expression-based CNV inference: order genes along the genome, center on a presumed-normal (stromal) reference, clip outliers, smooth within each chromosome. Then a **chromosome-length-normalized** aggregate score that gives each chromosome equal weight. |
| `annotate.py` | Four-stage tree-based hierarchical classifier (gradient-boosted trees) + a kNN reference-mapping baseline. The CNV score is an explicit input feature for the malignant call. |
| `evaluate.py` | 5-fold stratified CV and independent-cohort hold-out, each scoring tree vs baseline with macro-F1 on cell type, malignant call, subtype, and grade. |

## Why a CNV channel is offered to the annotator

Short-read scRNA-seq does not read copy number directly, but large chromosomal
gains and losses shift the *average* expression of long runs of neighbouring
genes. Ordering genes genomically and smoothing recovers a per-cell
pseudo-copy-number track (the public idea behind InferCNV and CopyKat). The
aggregate CNV score is offered to the normal-vs-malignant classifier as one
interpretable feature. Whether it improves the call over the transcriptomic
embedding alone is an empirical question the ablation answers — and the honest
answer on the synthetic regime is modest: a single CNV scalar approaches a
30-PC embedding (0.94 vs 0.99 macro-F1) and adding it to the embedding is only
marginally additive. The score's value is interpretability and compactness, not
a large accuracy gain.

## Why length-normalize the CNV score

A naive genome-wide mean of CNV magnitude over-weights chromosomes that simply
carry more genes. Averaging within each chromosome first, then across
chromosomes with equal weight, removes that length bias so the score reflects
*how many* chromosomes are disrupted rather than *how long* they are. This
de-biasing is the specific design idea the POC demonstrates; production
parameterizations of it are out of scope here.

## Substrate integration

| Channel | Module | Env var | Behaviour when unset |
|---|---|---|---|
| Audit | `audit` | `AUDIT_HOST` | local NDJSON only (still the source of truth) |
| MLflow | `tracking` | `MLFLOW_TRACKING_URI` | no-op |
| Canary | `canary` | `SCTUMOR_CANARY_FIXTURE` | uses the bundled fixture |

The hash-chained ledger format is shared across the capability-portrait quartet
so the same verifier works against any of them. The canary asserts the method's
central invariant (malignant CNV score > normal CNV score) in under a second,
which is what the daily lab probe checks.

## What this architecture intentionally avoids

No microservices, no async runtime, no DAG engine, no container per run, no
deep-learning dependency. The trainable model is a gradient-boosted tree
ensemble precisely because it is fast, deterministic, and CPU-only — the demo
must run on a recruiter's laptop.
