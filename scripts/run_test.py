#!/usr/bin/env python3
"""Interactive helper for starting a local benchmark test run."""

from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import sys
import datetime as dt
from typing import Any


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "config.local.json"
VENV_DIR = ROOT_DIR / ".venv"


def read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prompt_text(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_text}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer y or n.")


def prompt_int(prompt: str, default: int, *, minimum: int = 0) -> int:
    while True:
        raw = prompt_text(prompt, str(default))
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def parse_selection(raw: str, count: int) -> list[int]:
    selected: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    invalid = [idx for idx in selected if idx < 1 or idx > count]
    if invalid:
        raise ValueError(f"Selection out of range: {invalid}")
    return sorted(selected)


def prompt_model_selection(models: list[str], *, prompt_label: str = "Models", allow_all: bool = True) -> list[str]:
    print("\nAvailable models:")
    for index, model in enumerate(models, start=1):
        print(f"  {index}. {model}")
    if allow_all:
        print("\nSelect models by number, comma-separated. Use 'all' for every model.")
    else:
        print("\nSelect one model by number.")
    while True:
        raw = prompt_text(prompt_label, "all" if allow_all else "1").lower()
        if allow_all and raw in {"all", "*"}:
            return models
        try:
            indexes = parse_selection(raw, len(models))
        except ValueError as exc:
            print(exc)
            continue
        if not indexes:
            print("Select at least one model.")
            continue
        if not allow_all and len(indexes) != 1:
            print("Select exactly one model.")
            continue
        return [models[index - 1] for index in indexes]


def load_dotenv(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def venv_python() -> pathlib.Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv() -> pathlib.Path:
    python_path = venv_python()
    if not python_path.exists():
        print(f"Creating virtual environment: {VENV_DIR}")
        sys.stdout.flush()
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True, cwd=ROOT_DIR)
    print("Installing requirements...")
    sys.stdout.flush()
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
        cwd=ROOT_DIR,
    )
    return python_path


def filtered_config(
    config: dict[str, Any],
    *,
    selected_models: list[str],
    judge_model: str,
    collect_parallelism: int,
    grade_parallelism: int,
) -> dict[str, Any]:
    out = json.loads(json.dumps(config))
    collect = out.setdefault("collect", {})
    grade = out.setdefault("grade", {})
    collect["models"] = selected_models
    variants = collect.get("model_variants")
    if isinstance(variants, dict):
        collect["model_variants"] = {
            model: value for model, value in variants.items() if model in selected_models
        }
    grade["judge_model"] = judge_model
    collect["parallelism"] = collect_parallelism
    grade["parallelism"] = grade_parallelism
    return out


def main() -> int:
    config = read_json(DEFAULT_CONFIG)
    collect = config.get("collect", {})
    grade = config.get("grade", {})
    if not isinstance(collect, dict) or not isinstance(grade, dict):
        raise ValueError("Config must contain collect and grade objects.")

    models = [str(model).strip() for model in collect.get("models", []) if str(model).strip()]
    if not models:
        raise ValueError("No models found in config.local.json collect.models.")

    print("BullshitBench interactive test runner")
    print(f"Config: {DEFAULT_CONFIG.relative_to(ROOT_DIR)}")

    selected_models = prompt_model_selection(models)
    question_limit = prompt_int("How many questions? Use 0 for all", 10, minimum=0)
    default_judge = str(grade.get("judge_model", selected_models[0])).strip() or selected_models[0]
    print(f"\nDefault judge model: {default_judge}")
    if prompt_yes_no("Choose judge model by number?", True):
        judge_model = prompt_model_selection(models, prompt_label="Judge model", allow_all=False)[0]
    else:
        judge_model = prompt_text("Judge model", default_judge)
    collect_parallelism = prompt_int(
        "Concurrent collection requests",
        int(collect.get("parallelism", 1) or 1),
        minimum=1,
    )
    grade_parallelism = prompt_int(
        "Concurrent grading requests",
        int(grade.get("parallelism", 1) or 1),
        minimum=1,
    )
    run_id = prompt_text("Run ID (blank for timestamp)", "")
    if not run_id:
        run_id = "run_" + dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    dry_run = prompt_yes_no("Dry run without calling endpoint?", False)

    load_dotenv(ROOT_DIR / ".env")
    endpoint = config.get("endpoint", {})
    api_key_env = str(endpoint.get("api_key_env", "")).strip() if isinstance(endpoint, dict) else ""
    if api_key_env and not os.getenv(api_key_env) and not dry_run:
        print(f"Warning: {api_key_env} is not set. The endpoint may reject requests.")

    python_path = ensure_venv()
    run_config = filtered_config(
        config,
        selected_models=selected_models,
        judge_model=judge_model,
        collect_parallelism=collect_parallelism,
        grade_parallelism=grade_parallelism,
    )
    run_dir = ROOT_DIR / "runs" / run_id
    run_config_path = run_dir / "run_config.json"
    write_json(run_config_path, run_config)

    command = [
        str(python_path),
        "scripts/local_benchmark.py",
        "--config",
        str(run_config_path),
        "--limit",
        str(question_limit),
        "--run-id",
        run_id,
    ]
    if dry_run:
        command.append("--dry-run")
    command.append("all")

    print("\nStarting benchmark:")
    print(" ".join(shlex.quote(part) for part in command))
    sys.stdout.flush()
    return subprocess.run(command, cwd=ROOT_DIR, check=False, env=os.environ.copy()).returncode


if __name__ == "__main__":
    raise SystemExit(main())
