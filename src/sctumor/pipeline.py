"""End-to-end pipeline entry point.

Keeps the house-style shape::

    audit_start -> tracking_start -> body -> tracking_end -> audit_end

The body generates a synthetic multi-patient cohort, infers CNV, runs 5-fold
cross-validation and an independent-cohort hold-out for the tree-based
annotator versus the reference-mapping baseline, and writes a metrics artifact.
Everything is deterministic given the seed, so the canary smoke test exercises
the same code path with a smaller fixture.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from sctumor import ablation, audit, evaluate, tracking
from sctumor.synth import generate_malignancy_cohort, generate_multipatient


def _run_id(name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{name}-{stamp}"


def run_pipeline(
    run_name: str,
    out_dir: Path,
    *,
    n_patients: int = 3,
    cells_per_patient: int = 700,
    seed: int = 0,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Generate -> infer CNV -> evaluate -> write artifact."""
    out_dir.mkdir(parents=True, exist_ok=True)
    job_id = _run_id(run_name)

    audit.emit(
        action="pipeline_start",
        job_id=job_id,
        fields={"n_patients": n_patients, "cells_per_patient": cells_per_patient,
                "seed": seed, "n_splits": n_splits},
    )

    cohort = generate_multipatient(n_patients, cells_per_patient, seed=seed)

    metrics: dict[str, float] = {}
    with tracking.run(name=job_id, experiment="sctumor"):
        tracking.log_params(
            {"n_patients": n_patients, "cells_per_patient": cells_per_patient,
             "seed": seed, "n_splits": n_splits}
        )

        cv = evaluate.cross_validate(cohort, n_splits=n_splits)
        patients = [f"P{i}" for i in range(n_patients)]
        indep = evaluate.independent_cohort(
            cohort, train_patients=patients[:-1], test_patients=patients[-1:]
        )

        # v0.2 headline: CNV-feature ablation on a hard subclonal-CNV cohort
        hard = generate_malignancy_cohort(n_patients * cells_per_patient, seed=seed)
        abl = ablation.cnv_ablation(hard, n_splits=n_splits)

        # headline metrics to MLflow / artifact
        for k, v in cv["summary"].items():
            metrics[f"cv_{k}_mean"] = v["mean"]
        for k, v in indep.items():
            metrics[f"indep_{k}"] = v
        metrics["ablation_embedding_f1"] = abl["embedding_f1_mean"]
        metrics["ablation_cnv_only_f1"] = abl["cnv_only_f1_mean"]
        metrics["ablation_embedding_plus_cnv_f1"] = abl["embedding_plus_cnv_f1_mean"]
        metrics["ablation_cnv_additive_lift"] = abl["cnv_additive_lift"]
        tracking.log_metrics({k: float(v) for k, v in metrics.items()})

    artifact = {
        "run_name": run_name,
        "job_id": job_id,
        "n_cells": int(cohort.n_cells),
        "n_genes": int(cohort.n_genes),
        "cross_validation": cv,
        "independent_cohort": indep,
        "cnv_ablation": abl,
    }
    artifact_path = out_dir / f"{run_name}.json"
    with artifact_path.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True, default=str)

    audit.emit(
        action="pipeline_end",
        job_id=job_id,
        fields={
            "artifact_path": str(artifact_path),
            "cv_celltype_f1_mean": metrics.get("cv_tree_celltype_f1_mean"),
            "cv_malignant_f1_mean": metrics.get("cv_tree_malignant_f1_mean"),
        },
    )

    return {"job_id": job_id, "metrics": metrics, "artifact_path": str(artifact_path)}


@click.group()
def cli() -> None:
    """sctumor cancer-scRNA-seq annotation pipeline (synthetic-data POC)."""


@cli.command()
@click.option("--manifest", type=click.Path(path_type=Path),
              default=Path("data/manifest.yaml"))
@click.option("--out", type=click.Path(file_okay=False, path_type=Path),
              default=Path("data"))
def fetch(manifest: Path, out: Path) -> None:
    """No-op for the synthetic demo: data is generated, not downloaded.

    The manifest documents the *public* datasets the method is designed for, so
    a user can point the pipeline at real data; the demo itself needs no
    network access.
    """
    click.echo(json.dumps(
        {"status": "synthetic-demo", "note": "data is generated deterministically; "
         "see data/manifest.yaml for the public datasets the method targets",
         "manifest": str(manifest), "out": str(out)},
        indent=2,
    ))


@cli.command()
@click.option("--name", default="demo")
@click.option("--out", type=click.Path(file_okay=False, path_type=Path),
              default=Path("artifacts"))
@click.option("--seed", default=0, type=int)
@click.option("--patients", default=3, type=int)
@click.option("--cells", default=700, type=int)
def run(name: str, out: Path, seed: int, patients: int, cells: int) -> None:
    """Run the end-to-end pipeline."""
    result = run_pipeline(name, out, n_patients=patients, cells_per_patient=cells,
                          seed=seed)
    click.echo(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    cli()
