"""End-to-end pipeline + audit-chain smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

from sctumor import audit, pipeline


def test_pipeline_runs_and_produces_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    out_dir = tmp_path / "artifacts"
    result = pipeline.run_pipeline(
        "smoke", out_dir, n_patients=3, cells_per_patient=250, n_splits=3
    )

    assert "job_id" in result
    artifact_path = Path(result["artifact_path"])
    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text())
    assert payload["run_name"] == "smoke"
    assert "cross_validation" in payload
    assert "independent_cohort" in payload
    # CV cell-type macro-F1 should be strong on the separable synthetic data
    cv = payload["cross_validation"]["summary"]
    assert cv["tree_celltype_f1"]["mean"] > 0.80


def test_audit_chain_is_valid_after_pipeline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)

    pipeline.run_pipeline("smoke", tmp_path / "artifacts", cells_per_patient=200, n_splits=3)
    ok, n_entries, first_bad = audit.verify()
    assert ok, f"audit chain invalid at {first_bad}"
    assert n_entries >= 2


def test_audit_chain_detects_tamper(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_HOST", raising=False)

    pipeline.run_pipeline("smoke", tmp_path / "artifacts", cells_per_patient=200, n_splits=3)
    ledger = audit.DEFAULT_LEDGER
    lines = ledger.read_text().splitlines()
    assert len(lines) >= 2
    tampered = json.loads(lines[0])
    tampered["fields"]["seed"] = 999
    lines[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n")

    ok, _, first_bad = audit.verify()
    assert not ok
    assert first_bad is not None
