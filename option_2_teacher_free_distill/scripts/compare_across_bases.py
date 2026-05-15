"""Aggregate results across base models into a single cross-base report.

Reads:
- ``results/<tag>/best_per_probe.csv``           (ID best per probe)
- ``results/<tag>/all_metrics_long.csv``         (ID layer sweep)
- ``results/<tag>/bootstrap_pairs.csv``          (ID paired bootstrap)
- ``results/<tag>/ood_<name>_best_per_probe.csv``      (per OOD set)
- ``results/<tag>/ood_<name>_metrics_long.csv``        (per OOD set layer sweep)
- ``results/<tag>/ood_<name>_bootstrap_pairs.csv``     (per OOD set bootstrap)

For each base we read the registry in ``run_all_models.py`` to know
``num_hidden_layers`` so we can plot normalised-depth (= layer / N) tables.

Output:
- ``reports/CROSS_BASE_REPORT.md``       (master comparison)
- ``results/cross_base/<table>.csv``     (per-table CSVs for sharing)

Tables produced:
1. AUROC matrix:  rows = probe, cols = (base x dataset)
2. Best-layer matrix (raw + normalised depth)
3. OOD AUROC drop matrix
4. ID bootstrap (DCP vs SEPs-Ridge / SEPs-LR) per base
5. OOD bootstrap (DCP vs SEPs-Ridge / SEPs-LR) per (base, OOD set)
6. Normalised-depth alignment: AUROC at each probe's "ID best layer" mapped
   to nearest layer at each other base.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR / "scripts"))
# Reuse the BASES registry from the orchestrator
from run_all_models import BASES  # noqa: E402

PROBES_OF_INTEREST = ["DCP_mlp", "SEPs_logreg", "SEPs_ridge", "ARD_mlp", "ARD_ridge"]
DATASETS = ["id", "hotpotqa", "nq"]


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4f}"
    return str(v)


def load_base_results(tag: str) -> dict[str, Any]:
    """Load all CSVs we need for one base. Missing files are tolerated."""
    res_dir = THIS_DIR / "results" / tag
    out: dict[str, Any] = {"tag": tag}

    def maybe(path: Path) -> pd.DataFrame | None:
        return pd.read_csv(path) if path.exists() else None

    out["id_best"] = maybe(res_dir / "best_per_probe.csv")
    out["id_long"] = maybe(res_dir / "all_metrics_long.csv")
    out["id_boot"] = maybe(res_dir / "bootstrap_pairs.csv")
    for d in ("hotpotqa", "nq"):
        out[f"{d}_best"] = maybe(res_dir / f"ood_{d}_best_per_probe.csv")
        out[f"{d}_long"] = maybe(res_dir / f"ood_{d}_metrics_long.csv")
        out[f"{d}_boot"] = maybe(res_dir / f"ood_{d}_bootstrap_pairs.csv")
    return out


def build_id_best_long(blobs: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for blob in blobs:
        df = blob.get("id_best")
        if df is None:
            continue
        # Use cv_5fold rows only (the honest regime)
        sub = df[df["regime"] == "cv_5fold"]
        for _, r in sub.iterrows():
            rows.append({
                "base": blob["tag"],
                "dataset": "id",
                "probe": r["probe"],
                "layer": int(r["layer"]),
                "auroc": float(r["auroc"]),
                "aurc": float(r["aurc"]),
                "sel_acc@0.50": float(r.get("sel_acc@0.50", np.nan)),
                "brier": float(r.get("brier", np.nan)) if not pd.isna(r.get("brier", np.nan)) else np.nan,
                "ece": float(r.get("ece", np.nan)) if not pd.isna(r.get("ece", np.nan)) else np.nan,
            })
    return pd.DataFrame(rows)


def build_ood_best_long(blobs: list[dict], dataset: str) -> pd.DataFrame:
    rows: list[dict] = []
    for blob in blobs:
        df = blob.get(f"{dataset}_best")
        if df is None:
            continue
        for _, r in df.iterrows():
            rows.append({
                "base": blob["tag"],
                "dataset": dataset,
                "probe": r["probe"],
                "layer": int(r["layer"]),
                "auroc": float(r["auroc"]),
                "aurc": float(r["aurc"]),
                "sel_acc@0.50": float(r.get("sel_acc@0.50", np.nan)),
                "brier": float(r.get("brier", np.nan)) if not pd.isna(r.get("brier", np.nan)) else np.nan,
                "ece": float(r.get("ece", np.nan)) if not pd.isna(r.get("ece", np.nan)) else np.nan,
            })
    return pd.DataFrame(rows)


def render_md_table(df: pd.DataFrame, cols: list[str], header: list[str] | None = None) -> str:
    if df.empty:
        return "(empty)"
    head = header or cols
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def pivot_auroc_matrix(df_all: pd.DataFrame) -> pd.DataFrame:
    """Build (probe x (base, dataset)) matrix of best-layer AUROC."""
    df = df_all.copy()
    df["base_dataset"] = df["base"] + " | " + df["dataset"]
    pv = df.pivot_table(index="probe", columns="base_dataset", values="auroc", aggfunc="first")
    pv = pv.reindex([p for p in PROBES_OF_INTEREST if p in pv.index])
    return pv


def pivot_layer_matrix(df_all: pd.DataFrame, base_to_n: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_all.copy()
    df["base_dataset"] = df["base"] + " | " + df["dataset"]
    raw = df.pivot_table(index="probe", columns="base_dataset", values="layer", aggfunc="first")
    norm = raw.copy()
    for col in norm.columns:
        base = col.split(" | ")[0]
        N = base_to_n.get(base)
        if N:
            norm[col] = norm[col] / N
    raw = raw.reindex([p for p in PROBES_OF_INTEREST if p in raw.index])
    norm = norm.reindex([p for p in PROBES_OF_INTEREST if p in norm.index])
    return raw, norm


def build_ood_drop_table(df_all: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (base, probe), grp in df_all.groupby(["base", "probe"]):
        id_row = grp[grp["dataset"] == "id"]
        hpq_row = grp[grp["dataset"] == "hotpotqa"]
        nq_row = grp[grp["dataset"] == "nq"]
        if id_row.empty:
            continue
        id_au = float(id_row.iloc[0]["auroc"])
        rows.append({
            "base": base,
            "probe": probe,
            "id_auroc": id_au,
            "hotpotqa_auroc": float(hpq_row.iloc[0]["auroc"]) if not hpq_row.empty else np.nan,
            "nq_auroc": float(nq_row.iloc[0]["auroc"]) if not nq_row.empty else np.nan,
            "drop_hotpotqa": (float(hpq_row.iloc[0]["auroc"]) - id_au) if not hpq_row.empty else np.nan,
            "drop_nq": (float(nq_row.iloc[0]["auroc"]) - id_au) if not nq_row.empty else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["base", "probe"]).reset_index(drop=True)


def build_bootstrap_table(blobs: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for blob in blobs:
        if blob.get("id_boot") is not None:
            for _, r in blob["id_boot"].iterrows():
                rows.append({
                    "base": blob["tag"], "dataset": "id",
                    "probe_a": r["probe_a"], "probe_b": r["probe_b"],
                    "mean_diff": float(r["mean_diff"]),
                    "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
                    "p_value": float(r["p_value"]),
                    "regime": r.get("regime", ""),
                })
        for d in ("hotpotqa", "nq"):
            if blob.get(f"{d}_boot") is not None:
                for _, r in blob[f"{d}_boot"].iterrows():
                    rows.append({
                        "base": blob["tag"], "dataset": d,
                        "probe_a": r["probe_a"], "probe_b": r["probe_b"],
                        "mean_diff": float(r["mean_diff"]),
                        "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
                        "p_value": float(r["p_value"]),
                        "regime": r.get("regime", ""),
                    })
    return pd.DataFrame(rows)


def aligned_normalised_depth_sweep(blobs: list[dict], base_to_n: dict[str, int], target_norm: float) -> pd.DataFrame:
    """For each (base, probe, dataset), pick the layer whose normalised depth
    is closest to ``target_norm``, and report its AUROC. Useful for testing
    whether a single fixed normalised depth (e.g. 0.7) is a good universal
    deployment default."""
    rows: list[dict] = []
    for blob in blobs:
        tag = blob["tag"]
        N = base_to_n.get(tag)
        if N is None:
            continue
        for dataset in ("id", "hotpotqa", "nq"):
            df = blob.get("id_long") if dataset == "id" else blob.get(f"{dataset}_long")
            if df is None:
                continue
            sub = df.copy()
            if dataset == "id":
                sub = sub[sub["regime"] == "cv_5fold"]
            for probe in PROBES_OF_INTEREST:
                psub = sub[sub["probe"] == probe]
                if psub.empty:
                    continue
                psub = psub.assign(norm_depth=lambda x: x["layer"].astype(float) / N)
                # Pick layer closest to target_norm
                psub = psub.assign(diff=lambda x: (x["norm_depth"] - target_norm).abs())
                best = psub.sort_values("diff").iloc[0]
                rows.append({
                    "base": tag,
                    "dataset": dataset,
                    "probe": probe,
                    "layer": int(best["layer"]),
                    "norm_depth": float(best["norm_depth"]),
                    "auroc": float(best["auroc"]),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bases", nargs="*", default=None, help="Subset of base tags")
    p.add_argument(
        "--out-md",
        default=str(THIS_DIR / "reports" / "CROSS_BASE_REPORT.md"),
    )
    p.add_argument(
        "--out-csv-dir",
        default=str(THIS_DIR / "results" / "cross_base"),
    )
    p.add_argument("--target-norm", type=float, default=0.71, help="Normalised depth to align across bases")
    args = p.parse_args()

    selected = args.bases or [b.tag for b in BASES]
    base_to_n = {b.tag: b.num_hidden_layers for b in BASES if b.tag in selected}
    blobs = [load_base_results(tag) for tag in selected]
    blobs = [b for b in blobs if any(b.get(k) is not None for k in ("id_best", "hotpotqa_best", "nq_best"))]
    if not blobs:
        print("No usable results found. Did you run run_all_models.py first?")
        sys.exit(1)

    out_csv_dir = Path(args.out_csv_dir)
    out_csv_dir.mkdir(parents=True, exist_ok=True)

    # Build long tables
    id_long = build_id_best_long(blobs)
    hpq_long = build_ood_best_long(blobs, "hotpotqa")
    nq_long = build_ood_best_long(blobs, "nq")
    df_all = pd.concat([id_long, hpq_long, nq_long], ignore_index=True)
    df_all.to_csv(out_csv_dir / "best_per_probe_all_bases.csv", index=False)

    # Pivots
    auroc_pv = pivot_auroc_matrix(df_all)
    layer_raw, layer_norm = pivot_layer_matrix(df_all, base_to_n)
    auroc_pv.to_csv(out_csv_dir / "auroc_matrix.csv")
    layer_raw.to_csv(out_csv_dir / "best_layer_raw.csv")
    layer_norm.to_csv(out_csv_dir / "best_layer_norm_depth.csv")

    # Drop table
    drop_df = build_ood_drop_table(df_all)
    drop_df.to_csv(out_csv_dir / "ood_drop_table.csv", index=False)

    # Bootstrap aggregation
    boot_df = build_bootstrap_table(blobs)
    boot_df.to_csv(out_csv_dir / "bootstrap_all_bases.csv", index=False)

    # Aligned normalised-depth sweep
    align_df = aligned_normalised_depth_sweep(blobs, base_to_n, args.target_norm)
    align_df.to_csv(out_csv_dir / f"aligned_norm_depth_{args.target_norm:.2f}.csv", index=False)

    # ---- Render markdown ----
    md: list[str] = []
    md.append("# Cross-base teacher-free probe comparison\n")
    md.append("Generated by `scripts/compare_across_bases.py`. "
              "All AUROC values use sklearn's tie-corrected Mann-Whitney U.\n")

    md.append("\n## 0. Setup\n")
    md.append("| base | model dir | num_hidden_layers | hidden_size | layer sweep |")
    md.append("| --- | --- | --- | --- | --- |")
    for b in BASES:
        if b.tag not in selected:
            continue
        md.append(f"| {b.tag} | `{b.model_dir}` | {b.num_hidden_layers} | {b.hidden_size} | {b.layers} |")

    md.append("\n## 1. Best-layer AUROC matrix (rows = probe, cols = base × dataset)\n")
    md.append("Values are best-layer AUROC selected per (base, dataset, probe) on the cv_5fold regime (ID) or full-train→OOD apply (HotpotQA / NQ).\n")
    md.append("| probe | " + " | ".join(auroc_pv.columns) + " |")
    md.append("| --- | " + " | ".join(["---"] * len(auroc_pv.columns)) + " |")
    for probe, row in auroc_pv.iterrows():
        md.append("| " + str(probe) + " | " + " | ".join(fmt(row[c]) for c in auroc_pv.columns) + " |")

    md.append("\n## 2. Best layer (raw + normalised depth)\n")
    md.append("**Raw layer index:**\n")
    md.append("| probe | " + " | ".join(layer_raw.columns) + " |")
    md.append("| --- | " + " | ".join(["---"] * len(layer_raw.columns)) + " |")
    for probe, row in layer_raw.iterrows():
        md.append("| " + str(probe) + " | " + " | ".join(fmt(row[c]) for c in layer_raw.columns) + " |")
    md.append("\n**Normalised depth (layer / N):**\n")
    md.append("| probe | " + " | ".join(layer_norm.columns) + " |")
    md.append("| --- | " + " | ".join(["---"] * len(layer_norm.columns)) + " |")
    for probe, row in layer_norm.iterrows():
        md.append("| " + str(probe) + " | " + " | ".join(fmt(row[c]) for c in layer_norm.columns) + " |")

    md.append("\n## 3. OOD AUROC drop (per base × probe)\n")
    md.append(render_md_table(
        drop_df,
        ["base", "probe", "id_auroc", "hotpotqa_auroc", "drop_hotpotqa", "nq_auroc", "drop_nq"],
        ["base", "probe", "ID AUROC", "HQA AUROC", "Δ HQA", "NQ AUROC", "Δ NQ"],
    ))

    md.append("\n## 4. Paired-bootstrap pairwise comparisons (DCP vs baselines)\n")
    md.append(render_md_table(
        boot_df,
        ["base", "dataset", "regime", "probe_a", "probe_b", "mean_diff", "ci_low", "ci_high", "p_value"],
        ["base", "dataset", "regime", "A", "B", "ΔAUROC (A−B)", "CI lo", "CI hi", "p"],
    ))

    md.append(f"\n## 5. Aligned normalised-depth sweep at target = {args.target_norm:.2f}\n")
    md.append(
        "For each (base, probe, dataset) we pick the layer whose normalised "
        "depth is closest to the target and report its AUROC. This tests "
        "whether a single fixed normalised depth is a good universal "
        "deployment default across base models.\n"
    )
    md.append(render_md_table(
        align_df,
        ["base", "dataset", "probe", "layer", "norm_depth", "auroc"],
    ))

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[done] cross-base report -> {out_md}")
    print(f"[done] cross-base CSVs   -> {out_csv_dir}/")
    print(f"[summary] bases included: {[b['tag'] for b in blobs]}")
    print("\nAUROC matrix preview:")
    print(auroc_pv.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
