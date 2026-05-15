#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


DEFAULT_CONFIG = "/zhutingqi/song/Plan_gpt55/configs/direction_audit_config.json"
ROOT_DIR = Path("/zhutingqi/song")
INTERVENTION_A_DIR = ROOT_DIR / "ITI_for_entropy" / "Intervention_A"
sys.path.insert(0, str(INTERVENTION_A_DIR))

from intervention_utils import (  # noqa: E402
    entropy_weighted_pool,
    load_rows,
    load_split_ids,
    parse_feature_sites,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ITI direction assets and split/gate structure.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(torch.quantile(torch.tensor(values, dtype=torch.float32), q).item())


def cohen_d(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    pooled_var = 0.5 * (stdev(a) ** 2 + stdev(b) ** 2)
    if pooled_var <= 1e-12:
        return 0.0
    return (mean(a) - mean(b)) / (pooled_var ** 0.5)


def question_sort_key(question_id: str) -> tuple[int, str]:
    try:
        suffix = int(question_id.rsplit("_", 1)[-1])
    except ValueError:
        suffix = 10**9
    return suffix, question_id


def resolve_existing_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    text = str(path)
    replacements = [
        ("/public_zhutingqi/song", "/zhutingqi/song"),
        ("/public_zhutingqi", "/zhutingqi"),
    ]
    for old, new in replacements:
        if old in text:
            candidate = Path(text.replace(old, new, 1))
            if candidate.exists():
                return candidate
    return path


def build_direction_assets_local(config: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(config["run_dir"])
    split_manifest_path = Path(config["split_manifest"])
    feature_sites = parse_feature_sites(config["site_specs"])
    rows = load_rows(run_dir)
    row_by_id = {row["id"]: row for row in rows}
    train_ids = load_split_ids(split_manifest_path, "train")
    train_rows = [row_by_id[question_id] for question_id in train_ids]
    semantic_values = [float(row["semantic_entropy_weighted"]) for row in train_rows]
    semantic_threshold = quantile(semantic_values, float(config["semantic_high_quantile"]))
    all_train_step_entropies: list[float] = []
    accumulators: dict[str, dict[str, Any]] = {
        site.key: {
            "location": site.location,
            "layer": site.layer,
            "positive_sum": None,
            "negative_sum": None,
            "positive_count": 0,
            "negative_count": 0,
            "step_norms": [],
        }
        for site in feature_sites
    }
    for row in train_rows:
        question_dir = resolve_existing_path(row["result_dir"])
        raw_activations = torch.load(question_dir / "raw_activations.pt", map_location="cpu")
        trace_payload = read_json(question_dir / "layer_entropy_trace.json")
        steps = trace_payload.get("steps", [])
        if not steps:
            continue
        token_entropies = torch.tensor([float(step["actual_token_entropy"]) for step in steps], dtype=torch.float32)
        all_train_step_entropies.extend(token_entropies.tolist())
        class_name = "positive" if float(row["semantic_entropy_weighted"]) >= semantic_threshold else "negative"
        for site in feature_sites:
            vectors = raw_activations[site.location][str(site.layer)].float()
            pooled = entropy_weighted_pool(vectors, token_entropies)
            stats = accumulators[site.key]
            sum_key = f"{class_name}_sum"
            count_key = f"{class_name}_count"
            if stats[sum_key] is None:
                stats[sum_key] = pooled.clone()
            else:
                stats[sum_key] += pooled
            stats[count_key] += 1
            stats["step_norms"].extend(vectors.norm(dim=-1).tolist())

    gate_threshold = quantile(all_train_step_entropies, float(config["gate_entropy_quantile"]))
    site_payloads: dict[str, dict[str, Any]] = {}
    for site in feature_sites:
        stats = accumulators[site.key]
        positive_mean = stats["positive_sum"] / float(stats["positive_count"])
        negative_mean = stats["negative_sum"] / float(stats["negative_count"])
        raw_direction = positive_mean - negative_mean
        raw_norm = float(raw_direction.norm().item())
        site_payloads[site.key] = {
            "location": site.location,
            "layer": site.layer,
            "raw_direction": raw_direction.to(torch.float32),
            "unit_direction": (raw_direction / max(raw_norm, 1e-8)).to(torch.float32),
            "raw_norm": raw_norm,
            "positive_count": int(stats["positive_count"]),
            "negative_count": int(stats["negative_count"]),
            "median_step_norm": float(torch.tensor(stats["step_norms"], dtype=torch.float32).median().item()),
            "dim": int(raw_direction.shape[0]),
        }
    return {
        "metadata": {
            "run_dir": str(run_dir),
            "split_manifest_path": str(split_manifest_path),
            "semantic_high_quantile": float(config["semantic_high_quantile"]),
            "semantic_high_threshold": semantic_threshold,
            "gate_entropy_quantile": float(config["gate_entropy_quantile"]),
            "gate_entropy_threshold": gate_threshold,
            "feature_sites": [{"location": site.location, "layer": site.layer} for site in feature_sites],
            "train_question_count": len(train_rows),
        },
        "sites": site_payloads,
    }


def selected_all_rows_ids(rows: list[dict[str, Any]], offset: int, num_questions: int) -> set[str]:
    ordered = sorted(rows, key=lambda row: question_sort_key(str(row["id"])))
    return {str(row["id"]) for row in ordered[offset : offset + num_questions]}


def split_overlap_report(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_path = Path(config["split_manifest"])
    deploy_cfg = config.get("deploy_selection", {})
    split_ids = {
        split_name: set(load_split_ids(manifest_path, split_name))
        for split_name in ["train", "val", "test"]
    }
    if deploy_cfg.get("mode") != "all_rows":
        return {"mode": deploy_cfg.get("mode", ""), "note": "only all_rows overlap is audited"}
    selected_ids = selected_all_rows_ids(
        rows,
        int(deploy_cfg.get("offset", 0)),
        int(deploy_cfg.get("num_questions", len(rows))),
    )
    return {
        "mode": "all_rows",
        "selected_count": len(selected_ids),
        "overlap": {
            split_name: len(selected_ids & ids)
            for split_name, ids in split_ids.items()
        },
    }


def collect_projection_rows(
    config: dict[str, Any],
    assets: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    run_dir = Path(config["run_dir"])
    rows = load_rows(run_dir)
    row_by_id = {row["id"]: row for row in rows}
    train_ids = load_split_ids(Path(config["split_manifest"]), "train")
    train_rows = [row_by_id[question_id] for question_id in train_ids]
    semantic_threshold = float(assets["metadata"]["semantic_high_threshold"])

    projection_rows: dict[str, list[dict[str, Any]]] = {site_key: [] for site_key in assets["sites"]}
    for row in train_rows:
        question_dir = resolve_existing_path(row["result_dir"])
        raw_activations = torch.load(question_dir / "raw_activations.pt", map_location="cpu")
        trace_payload = read_json(question_dir / "layer_entropy_trace.json")
        steps = trace_payload.get("steps", [])
        if not steps:
            continue
        token_entropies = torch.tensor([float(step["actual_token_entropy"]) for step in steps], dtype=torch.float32)
        semantic_value = float(row["semantic_entropy_weighted"])
        is_positive = semantic_value >= semantic_threshold
        for site_key, payload in assets["sites"].items():
            vectors = raw_activations[payload["location"]][str(payload["layer"])].float()
            pooled = entropy_weighted_pool(vectors, token_entropies)
            direction = payload["unit_direction"].float()
            projection = float(torch.dot(pooled, direction).item())
            projection_rows[site_key].append(
                {
                    "question_id": row["id"],
                    "semantic_entropy_weighted": semantic_value,
                    "is_positive": is_positive,
                    "projection": projection,
                    "token_entropy_mean": float(token_entropies.mean().item()),
                    "token_entropy_max": float(token_entropies.max().item()),
                }
            )
    return projection_rows


def summarize_site(site_key: str, payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    projections = [float(row["projection"]) for row in rows]
    pos = [float(row["projection"]) for row in rows if bool(row["is_positive"])]
    neg = [float(row["projection"]) for row in rows if not bool(row["is_positive"])]
    ordered_abs = sorted(rows, key=lambda row: abs(float(row["projection"])), reverse=True)
    return {
        "site_key": site_key,
        "location": payload["location"],
        "layer": int(payload["layer"]),
        "raw_norm": float(payload["raw_norm"]),
        "median_step_norm": float(payload["median_step_norm"]),
        "positive_count": int(payload["positive_count"]),
        "negative_count": int(payload["negative_count"]),
        "positive_ratio": int(payload["positive_count"]) / max(int(payload["positive_count"]) + int(payload["negative_count"]), 1),
        "projection_mean": mean(projections),
        "projection_std": stdev(projections),
        "projection_q05": quantile(projections, 0.05),
        "projection_q50": quantile(projections, 0.50),
        "projection_q95": quantile(projections, 0.95),
        "positive_projection_mean": mean(pos),
        "negative_projection_mean": mean(neg),
        "projection_separation": mean(pos) - mean(neg),
        "projection_cohen_d": cohen_d(pos, neg),
        "top_abs_projection_questions": [
            {
                "question_id": row["question_id"],
                "projection": float(row["projection"]),
                "semantic_entropy_weighted": float(row["semantic_entropy_weighted"]),
                "is_positive": bool(row["is_positive"]),
            }
            for row in ordered_abs[:8]
        ],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Direction Asset Audit",
        "",
        "## Metadata",
        "",
        f"- Run dir: `{payload['metadata']['run_dir']}`",
        f"- Train question count: `{payload['metadata']['train_question_count']}`",
        f"- Semantic threshold: `{payload['metadata']['semantic_high_threshold']:.6f}`",
        f"- Gate threshold: `{payload['metadata']['gate_entropy_threshold']:.6f}`",
        "",
        "## Deploy Selection Split Overlap",
        "",
    ]
    overlap = payload["split_overlap"]
    if "overlap" in overlap:
        lines.extend(
            [
                f"- Selected count: `{overlap['selected_count']}`",
                f"- Train overlap: `{overlap['overlap'].get('train', 0)}`",
                f"- Val overlap: `{overlap['overlap'].get('val', 0)}`",
                f"- Test overlap: `{overlap['overlap'].get('test', 0)}`",
                "",
            ]
        )
    else:
        lines.append(f"- {overlap.get('note', 'No overlap report')}")
        lines.append("")
    lines.extend(
        [
            "## Site Summaries",
            "",
            "| Site | Pos/Neg | Raw norm | Median step norm | Projection separation | Cohen d |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for site in payload["sites"]:
        lines.append(
            f"| `{site['site_key']}` | `{site['positive_count']}/{site['negative_count']}` | "
            f"`{site['raw_norm']:.4f}` | `{site['median_step_norm']:.4f}` | "
            f"`{site['projection_separation']:.4f}` | `{site['projection_cohen_d']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Strong projection separation supports using the site as a diagnostic probe.",
            "- Weak or outlier-dominated separation argues against fixed-vector deployment control.",
            "- If deploy selection overlaps train heavily, deployment-style claims should be separated from held-out claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = build_direction_assets_local(config)
    rows = load_rows(Path(config["run_dir"]))
    projection_rows = collect_projection_rows(config, assets)
    site_summaries = [
        summarize_site(site_key, payload, projection_rows[site_key])
        for site_key, payload in assets["sites"].items()
    ]
    payload = {
        "metadata": assets["metadata"],
        "split_overlap": split_overlap_report(config, rows),
        "sites": site_summaries,
    }
    (output_dir / "direction_asset_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "direction_asset_audit.md").write_text(
        build_markdown(payload),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "status": "ok"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
