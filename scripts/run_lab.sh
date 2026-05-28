#!/usr/bin/env bash
# Execute the capability-portrait pipeline on a Polish-Phase5 lab node.
#
# Wraps `make run` with the substrate environment variables set to the lab
# defaults, so audit entries flow to the audit-API and MLflow runs are tracked.
# On a fresh checkout without the substrate present, run `make run` directly.

set -euo pipefail

export AUDIT_HOST="${AUDIT_HOST:-chi-mac-m:8081}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://chi-mac-m:5050}"
export SCTUMOR_RUN_NAME="${SCTUMOR_RUN_NAME:-lab-$(date -u +%Y%m%d-%H%M%S)}"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found on PATH" >&2
    exit 2
fi

echo "[run_lab] AUDIT_HOST=${AUDIT_HOST}"
echo "[run_lab] MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"
echo "[run_lab] RUN_NAME=${SCTUMOR_RUN_NAME}"

uv run make run RUN_NAME="${SCTUMOR_RUN_NAME}"

echo "[run_lab] canary check"
uv run python -m sctumor.canary

echo "[run_lab] done"
