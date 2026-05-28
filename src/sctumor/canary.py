"""Deterministic canary smoke test.

Probed daily by the Polish-Phase5 ``lab_semantic_check.py`` runner. Contract:

1. Completes in well under 30 seconds on a single workstation.
2. Deterministic given the fixed fixture.
3. Exit code 0 on success, non-zero on any deviation.
4. No external services required (audit + MLflow hooks degrade to no-ops).

The check generates a tiny synthetic cohort, infers CNV, and asserts the core
invariant of the method: malignant epithelial cells carry a higher
chromosome-length-normalized CNV score than normal cells. It also emits the
standard audit entries so substrate monitoring sees the same code path the full
pipeline uses.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sctumor import audit, cnv, tracking
from sctumor.synth import STROMAL_TYPES, generate_cohort

DEFAULT_FIXTURE = Path("tests/fixtures/canary.json")
EXPECTED_KEYS = {"name", "tier", "min_cnv_separation"}


def _load_fixture(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"canary fixture not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def check() -> dict[str, Any]:
    fixture_path = Path(os.environ.get("SCTUMOR_CANARY_FIXTURE", str(DEFAULT_FIXTURE)))
    fixture = _load_fixture(fixture_path)

    missing = EXPECTED_KEYS - set(fixture.keys())
    if missing:
        return {"ok": False, "reason": f"fixture missing keys: {sorted(missing)}"}

    job_id = f"canary-{fixture['name']}"
    audit.emit(action="canary_start", job_id=job_id,
               fields={"tier": fixture["tier"], "fixture_path": str(fixture_path)})

    cohort = generate_cohort(240, seed=7)
    ref = np.isin(cohort.obs["cell_type"].to_numpy(), STROMAL_TYPES)
    cnv_mat, ch = cnv.infer_cnv(cohort.expr, cohort.genes, ref)
    score = cnv.cnv_score(cnv_mat, ch)

    mal = cohort.obs["is_malignant"].to_numpy()
    normal = ~mal
    separation = float(score[mal].mean() - score[normal].mean())

    ok = separation >= float(fixture["min_cnv_separation"])

    with tracking.run(name=job_id, experiment="canary"):
        tracking.log_params({"tier": fixture["tier"]})
        tracking.log_metric("cnv_separation", separation)

    audit.emit(action="canary_end", job_id=job_id,
               fields={"ok": ok, "cnv_separation": separation})

    return {
        "ok": ok,
        "job_id": job_id,
        "cnv_separation": separation,
        "min_required": float(fixture["min_cnv_separation"]),
    }


def main() -> int:
    result = check()
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
