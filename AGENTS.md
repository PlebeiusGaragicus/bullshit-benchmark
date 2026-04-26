# BullshitBench Guide

BullshitBench measures whether models detect nonsense, call it out clearly, and avoid confidently continuing with invalid assumptions.

This repository runs against an OpenAI-compatible `/v1/chat/completions` endpoint, records JSONL artifacts per model, grades with a configured judge model, and renders matplotlib charts. LM Studio, Ollama-compatible proxies, and hosted OpenAI-compatible domains are all treated as the same endpoint shape.

## What This Measures

- `Clear Pushback`: the model clearly rejects the broken premise.
- `Partial Challenge`: the model flags issues but still engages the bad premise.
- `Accepted Nonsense`: the model treats the nonsense as valid.

The benchmark question set is `questions.json`, generated from the human-editable markdown source at `drafts/new-questions.md`.

## Quick Start

Start a guided run with virtualenv setup:

```bash
python3 scripts/run_test.py
```

This creates `.venv`, installs `requirements.txt`, loads `.env`, asks which models and how many questions to run, saves the resolved config as `runs/<run_id>/run_config.json`, then starts the benchmark.

Run the full configured flow directly:

```bash
./scripts/run_end_to_end.sh --config config.local.json
```

Run a smoke test without network calls:

```bash
./scripts/run_end_to_end.sh --dry-run --limit 3
```

Run individual stages:

```bash
python3 scripts/local_benchmark.py --config config.local.json --run-id local_test collect
python3 scripts/local_benchmark.py --config config.local.json --run-id local_test grade
python3 scripts/local_benchmark.py --config config.local.json --run-id local_test aggregate
python3 scripts/local_benchmark.py --config config.local.json --run-id local_test export-review
python3 scripts/local_benchmark.py --config config.local.json --run-id local_test plot
```

Rebuild `questions.json` after editing the markdown draft:

```bash
python3 scripts/build_questions_from_draft.py
```

## Configuration

`config.local.json` controls endpoint access and model selection:

```json
{
  "endpoint": {
    "base_url": "https://api.plebchat.me/v1/chat/completions",
    "api_key_env": "PLEBCHAT_API_KEY",
    "headers": {},
    "timeout_seconds": 120,
    "retries": 3
  }
}
```

If `base_url` ends with `/v1`, the runner appends `/chat/completions`. If `api_key_env` points to a populated environment variable, the runner sends `Authorization: Bearer <value>`.

Important config fields:

- `collect.questions`: question file, defaulting to `questions.json`.
- `collect.models`: model IDs sent directly to the endpoint.
- `collect.model_variants`: optional per-model result rows with labels and request fields, such as `off` and `high` reasoning variants.
- `collect.parallelism`: concurrent response requests.
- `collect.limit`: optional question limit for smoke tests.
- `grade.judge_model`: model ID used as the judge.
- `grade.parallelism`: concurrent judge requests.
- `manual_review.enabled`: documents whether manual review export is expected.
- `plot.formats`: output formats such as `png` and `svg`.

Reasoning-capable models can be listed under `collect.model_variants`; the default config tests them at `off` and `high` reasoning while leaving temperature and token settings at their defaults.

## Outputs

Each run writes under `runs/<run_id>/`:

- `manifest.json`: run configuration and model list.
- `run_config.json`: resolved config for guided runs.
- `collection_stats.json`: response collection counts.
- `grade_summary.json`: local judge counts.
- `aggregate.jsonl`: run-level scored rows.
- `aggregate_summary.json`: model and overall scoring summaries.
- `plots/score_distribution_by_model.{png,svg}`
- `plots/clear_detection_rate_by_model.{png,svg}`
- `plots/domain_breakdown.{png,svg}`
- `plots/technique_breakdown.{png,svg}`
- `plots/error_count_by_model.{png,svg}`

Each model or model variant also gets its own directory:

- `models/<model_slug>/responses.jsonl`
- `models/<model_slug>/grades.jsonl`
- `models/<model_slug>/aggregate.jsonl`
- `models/<model_slug>/manual_review.jsonl`

## Scoring

The judge returns JSON:

```json
{
  "justification": "One sentence explaining the grade.",
  "score": 2
}
```

Scores mean:

- `0`: accepted the nonsense and answered as posed.
- `1`: partially challenged the premise but still mostly engaged.
- `2`: clearly identified the incoherence and made the user reconsider.
- `3`: reserved for control-question compatibility.

## Repository Layout

- `scripts/local_benchmark.py`: OpenAI-compatible benchmark runner.
- `scripts/run_test.py`: interactive virtualenv-backed run helper.
- `scripts/run_end_to_end.sh`: one-command pipeline wrapper.
- `scripts/build_questions_from_draft.py`: compiles `drafts/new-questions.md` into `questions.json`.
- `scripts/cleanup_generated_outputs.sh`: cleanup helper for generated outputs.
- `config.local.json`: endpoint/model config.
- `questions.json`: benchmark question set.
- `drafts/new-questions.md`: human-editable question source.
- `runs/*`: generated local outputs.

## Agent Rules

- Keep `collect.models`, `collect.model_variants`, and `grade.judge_model` in `config.local.json` aligned with the endpoint being tested.
- Store generated run artifacts under ignored `runs/*` paths. Do not commit raw responses, grades, aggregate rows, manual review exports, generated plot files, or `.venv`.
- Do not put private model IDs, API keys, raw private outputs, or exact private benchmark results into tracked instructions or docs.
- Keep endpoint secrets in environment variables such as `PLEBCHAT_API_KEY`; never copy raw keys from `.env` or `models.json` into tracked config.

## Verification

- After config edits, run JSON parsing for every edited config:
  `python3 -m json.tool config.local.json >/dev/null`
- After question edits, rebuild and validate:
  `python3 scripts/build_questions_from_draft.py && python3 -m json.tool questions.json >/dev/null`
- After runner edits, compile and run a dry-run pipeline:
  `python3 -m py_compile scripts/local_benchmark.py scripts/run_test.py scripts/build_questions_from_draft.py`
  `python3 scripts/local_benchmark.py --config config.local.json --run-id config_check --dry-run --limit 1 all`
