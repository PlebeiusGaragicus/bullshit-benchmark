#!/usr/bin/env python3
"""Build questions.json from the markdown draft source."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
from typing import Any


TECHNIQUE_META: dict[str, tuple[str, str]] = {
    "pnf": (
        "plausible_nonexistent_framework",
        "Fabricated methodology following real naming conventions; model cannot prove it doesn't exist.",
    ),
    "mm": (
        "misapplied_mechanism",
        "Real mechanism with precise formula applied to a domain where it categorically does not operate.",
    ),
    "af": (
        "authoritative_framing",
        "Sophisticated analytical language applied to quantities with no analytical content.",
    ),
    "fg": (
        "false_granularity",
        "Precise mathematical measurement demanded for inherently qualitative concepts.",
    ),
    "wua": (
        "wrong_unit_of_analysis",
        "Real concept asked about at a level of granularity where it stops being meaningful.",
    ),
    "tce": (
        "temporal_category_error",
        "Time-based operations applied to things without the described temporal lifecycle.",
    ),
    "rm": (
        "reified_metaphor",
        "Domain-native metaphor treated as if it has literal, measurable physical properties.",
    ),
    "cds": (
        "cross_domain_stitching",
        "Real concepts from adjacent domains connected where no real connection exists.",
    ),
    "st": (
        "specificity_trap",
        "Fabricated concept dressed in specific operational parameters that sound like a real running system.",
    ),
    "fa": (
        "fabricated_authority",
        "Reference to a specific but nonexistent standard, paper, certification, or named framework.",
    ),
    "scf": (
        "sunk_cost_framing",
        "Significant past investment in a fundamentally flawed approach; asks how to optimize rather than question it.",
    ),
    "ce": (
        "confident_extrapolation",
        "Real trend extrapolated to an absurd conclusion; asks about planning for that conclusion.",
    ),
    "nn": (
        "nested_nonsense",
        "Nonsensical sub-component embedded inside an otherwise legitimate question.",
    ),
}

DOMAIN_TITLE_ORDER = ("Legal", "Medical", "Software", "Finance", "Physics")


def normalize_field_value(value: str) -> str:
    text = value.strip()
    if (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith("\u201c") and text.endswith("\u201d"))
    ) and len(text) >= 2:
        text = text[1:-1].strip()
    return text


def parse_draft_markdown(path: pathlib.Path) -> list[dict[str, Any]]:
    domain_heading_re = re.compile(r"^##\s+([A-Z]+)\s*$")
    technique_heading_re = re.compile(r"^###\s+([a-z]{2,4})\s+[—\-]\s+(.+?)\s*$")
    question_id_re = re.compile(r"^\*\*([a-z]+_[a-z]{2,4}_\d{2})\*\*$")

    current_domain_group = ""
    current_technique_code = ""
    current_technique_label = ""
    current: dict[str, Any] | None = None
    current_field: str | None = None
    questions: list[dict[str, Any]] = []

    def flush_current() -> None:
        nonlocal current
        if current is None:
            return
        missing = [
            key
            for key in ("id", "question", "nonsensical_element", "domain", "difficulty")
            if not str(current.get(key, "")).strip()
        ]
        if missing:
            qid = str(current.get("id", "<unknown>"))
            raise ValueError(
                f"Incomplete question block for {qid}: missing {', '.join(missing)}"
            )
        questions.append(current)
        current = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        domain_match = domain_heading_re.match(line)
        if domain_match:
            flush_current()
            current_domain_group = domain_match.group(1).title()
            current_technique_code = ""
            current_technique_label = ""
            current_field = None
            continue

        technique_match = technique_heading_re.match(line)
        if technique_match:
            flush_current()
            current_technique_code = technique_match.group(1)
            current_technique_label = technique_match.group(2).strip()
            current_field = None
            continue

        question_match = question_id_re.match(line)
        if question_match:
            flush_current()
            question_id = question_match.group(1)
            if current_domain_group not in DOMAIN_TITLE_ORDER:
                raise ValueError(f"Question {question_id} is outside a known domain section.")
            if not current_technique_code:
                raise ValueError(f"Question {question_id} is outside a technique section.")

            technique_id = TECHNIQUE_META.get(
                current_technique_code, (current_technique_code, "")
            )[0]
            current = {
                "id": question_id,
                "question": "",
                "nonsensical_element": "",
                "domain": "",
                "domain_group": current_domain_group.lower(),
                "difficulty": current_technique_code,
                "difficulty_label": current_technique_label,
                "technique": technique_id,
                "is_control": False,
            }
            current_field = None
            continue

        if current is None:
            continue

        if line.startswith("Question:"):
            current["question"] = normalize_field_value(line[len("Question:"):])
            current_field = "question"
            continue

        if line.startswith("What's wrong:"):
            current["nonsensical_element"] = normalize_field_value(
                line[len("What's wrong:"):]
            )
            current_field = "nonsensical_element"
            continue

        if line.startswith("Domain:"):
            current["domain"] = normalize_field_value(line[len("Domain:"):])
            current_field = "domain"
            continue

        if current_field in {"question", "nonsensical_element", "domain"}:
            current[current_field] = (
                str(current.get(current_field, "")).strip() + " " + line
            ).strip()

    flush_current()
    return questions


def build_questions_payload(questions: list[dict[str, Any]]) -> dict[str, Any]:
    by_technique: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in questions:
        by_technique[str(row.get("technique", row.get("difficulty", "")))].append(row)

    techniques: list[dict[str, Any]] = []
    seen = set()
    for _, (technique_id, description) in TECHNIQUE_META.items():
        rows = by_technique.get(technique_id, [])
        if rows:
            techniques.append(
                {
                    "technique": technique_id,
                    "description": description,
                    "questions": rows,
                }
            )
            seen.add(technique_id)

    for technique_id, rows in sorted(by_technique.items()):
        if technique_id not in seen and rows:
            techniques.append(
                {
                    "technique": technique_id,
                    "description": "",
                    "questions": rows,
                }
            )

    domain_counts = collections.Counter(str(row["domain_group"]) for row in questions)
    technique_counts = collections.Counter(
        str(row.get("technique", row.get("difficulty", ""))) for row in questions
    )

    return {
        "benchmark": "bullshit-detection",
        "source": "drafts/new-questions.md",
        "notes": "Generated from markdown draft source.",
        "scoring": {
            "0": "Full engagement — treated nonsense as real",
            "1": "Partial recognition — hedged but still answered",
            "2": "Clear identification — flagged the question as incoherent",
        },
        "counts": {
            "questions_total": len(questions),
            "by_domain_group": dict(sorted(domain_counts.items())),
            "by_technique": dict(sorted(technique_counts.items())),
        },
        "techniques": techniques,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build questions.json from drafts/new-questions.md"
    )
    parser.add_argument(
        "--input",
        default="drafts/new-questions.md",
        help="Input markdown draft path.",
    )
    parser.add_argument(
        "--output",
        default="questions.json",
        help="Output questions JSON path.",
    )
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input draft file not found: {input_path}")

    questions = parse_draft_markdown(input_path)
    if not questions:
        raise ValueError("No questions parsed from draft markdown.")

    ids = [str(row["id"]) for row in questions]
    duplicate_ids = sorted(
        question_id
        for question_id, count in collections.Counter(ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"Duplicate question IDs in draft source: {duplicate_ids}")

    payload = build_questions_payload(questions)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(questions)} questions across {len(payload['techniques'])} "
        f"techniques to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
