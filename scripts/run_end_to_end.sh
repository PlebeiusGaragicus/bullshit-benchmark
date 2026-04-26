#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_end_to_end.sh [options]

Runs the local benchmark flow:
  1) collect
  2) grade with the local judge model
  3) aggregate
  4) export optional manual-review JSONL
  5) render matplotlib plots

Options:
  --config <path>       Config file (default: config.local.json)
  --output-dir <dir>    Output base dir (default: runs)
  --run-id <id>         Explicit run id (default: auto timestamp)
  --skip-collect        Skip collect stage (requires existing responses file)
  --skip-grade          Skip grade stage
  --skip-aggregate      Skip aggregate stage
  --skip-review-export  Skip manual review export
  --skip-plot           Skip matplotlib plot generation
  --limit <n>           Limit questions for collect
  --dry-run             Use deterministic fake endpoint responses
  -h, --help            Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="config.local.json"
OUTPUT_DIR="runs"
RUN_ID=""
DRY_RUN=0
SKIP_COLLECT=0
SKIP_GRADE=0
SKIP_AGGREGATE=0
SKIP_REVIEW_EXPORT=0
SKIP_PLOT=0
LIMIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-collect)
      SKIP_COLLECT=1
      shift
      ;;
    --skip-grade)
      SKIP_GRADE=1
      shift
      ;;
    --skip-aggregate)
      SKIP_AGGREGATE=1
      shift
      ;;
    --skip-review-export)
      SKIP_REVIEW_EXPORT=1
      shift
      ;;
    --skip-plot)
      SKIP_PLOT=1
      shift
      ;;
    --limit)
      LIMIT="${2:-0}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ -z "${RUN_ID}" ]]; then
  if [[ "${SKIP_COLLECT}" -eq 1 ]]; then
    echo "--skip-collect requires --run-id." >&2
    exit 2
  fi
  RUN_ID="run_$(date -u +%Y%m%d_%H%M%S)"
fi

RUN_DIR="${OUTPUT_DIR}/${RUN_ID}"

collect_cmd=(
  python3 scripts/local_benchmark.py
  --config "${CONFIG_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --run-id "${RUN_ID}"
  --limit "${LIMIT}"
)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  collect_cmd+=(--dry-run)
fi

if [[ "${SKIP_COLLECT}" -eq 1 ]]; then
  echo "==> Skipping collect stage"
else
  echo "==> Collect: ${RUN_ID}"
  "${collect_cmd[@]}" collect
fi

grade_cmd=(
  python3 scripts/local_benchmark.py
  --config "${CONFIG_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --run-id "${RUN_ID}"
)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  grade_cmd+=(--dry-run)
fi

if [[ "${SKIP_GRADE}" -eq 1 ]]; then
  echo "==> Skipping grade stage"
else
  echo "==> Grade local judge"
  "${grade_cmd[@]}" grade
fi

if [[ "${SKIP_AGGREGATE}" -eq 1 ]]; then
  echo "==> Skipping aggregate stage"
else
  echo "==> Aggregate"
  python3 scripts/local_benchmark.py \
    --config "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-id "${RUN_ID}" \
    aggregate
fi

if [[ "${SKIP_REVIEW_EXPORT}" -eq 1 ]]; then
  echo "==> Skipping manual review export"
else
  echo "==> Export manual review JSONL"
  python3 scripts/local_benchmark.py \
    --config "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-id "${RUN_ID}" \
    export-review
fi

if [[ "${SKIP_PLOT}" -eq 1 ]]; then
  echo "==> Skipping plots"
else
  echo "==> Render matplotlib plots"
  python3 scripts/local_benchmark.py \
    --config "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-id "${RUN_ID}" \
    plot
fi

echo ""
echo "Complete."
echo "Run ID: ${RUN_ID}"
echo "Run dir: ${ROOT_DIR}/${RUN_DIR}"
echo "Plots: ${ROOT_DIR}/${RUN_DIR}/plots"
