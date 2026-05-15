#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MATRIX = "/zhutingqi/song/Plan_gpt55/configs/experiment_matrix.json"
DEFAULT_OUTPUT = "/zhutingqi/song/Plan_gpt55/reports/experiment_matrix_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the Plan_gpt55 experiment matrix.")
    parser.add_argument("--matrix", default=DEFAULT_MATRIX)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# Experiment Matrix Summary",
        "",
        f"- Project: `{matrix['project']}`",
        f"- Purpose: {matrix['purpose']}",
        "",
        "## Global Principles",
        "",
    ]
    for key, value in matrix["global_principles"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Success Criteria",
            "",
            "| Criterion | Value |",
            "| --- | ---: |",
        ]
    )
    for key, value in matrix["success_criteria"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Experiments",
            "",
            "| Priority | ID | Status | Goal | Decision |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for exp in sorted(matrix["experiments"], key=lambda item: int(item["priority"])):
        goal = exp.get("goal", exp.get("entry_condition", ""))
        decision = exp.get("decision", exp.get("entry_condition", ""))
        lines.append(
            f"| `{exp['priority']}` | `{exp['id']}` | `{exp['status']}` | "
            f"{goal} | {decision} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    matrix = read_json(args.matrix)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(matrix), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": "ok"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
