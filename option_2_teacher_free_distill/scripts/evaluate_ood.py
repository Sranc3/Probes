"""Evaluate teacher-free probes on one or more OOD datasets.

For each (probe family, layer) and each OOD dataset:

1. Train on the FULL TriviaQA cache (no folds) using the same target as in
   ``train_probes.py`` - SEPs-Ridge predicts ``semantic_entropy_weighted_set``,
   SEPs-LR / DCP-MLP predict ``strict_correct``.
2. Apply the trained probe to the OOD hidden states.
3. Compute selective-prediction metrics on the OOD set (labels = strict_correct
   of greedy sample0 against ideals).
4. Run paired bootstrap (DCP-MLP vs each baseline) to test significance.

Outputs (per OOD dataset ``<name>``):
- ``results/ood_<name>_metrics_long.csv``
- ``results/ood_<name>_best_per_probe.csv``
- ``results/ood_<name>_bootstrap_pairs.csv``

Combined:
- ``results/ood_combined_table.md``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR / "shared"))

from data_utils import (  # noqa: E402
    attach_targets,
    load_anchor_rows,
    question_index_map,
    select_sample0_rows,
)
from probe_utils import (  # noqa: E402
    LogRegProbe,
    MLPProbe,
    RidgeProbe,
    safe_auroc,
    summarize_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--triviaqa-anchor",
        default="/zhutingqi/song/Plan_gpt55/cross_model_anchor/runs/run_20260513_012413_gptoss_anchor_triviaqa_full500_k4_tok256/qwen_candidate_anchor_rows_final_only.csv",
    )
    p.add_argument(
        "--triviaqa-hidden",
        default=str(THIS_DIR / "runs" / "hidden_states.npz"),
    )
    p.add_argument(
        "--ood-cache",
        action="append",
        default=None,
        help=(
            "OOD dataset cache as 'name=path/to/cache.npz'. Repeatable. "
            "Example: --ood-cache hotpotqa=runs/hotpotqa_ood.npz "
            "         --ood-cache nq=runs/nq_ood.npz"
        ),
    )
    p.add_argument(
        "--out-dir",
        default=str(THIS_DIR / "results"),
    )
    p.add_argument("--mlp-hidden", type=int, default=128)
    p.add_argument("--mlp-epochs", type=int, default=200)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--logreg-c", type=float, default=1.0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not args.ood_cache:
        # backwards-compat default: just HotpotQA
        args.ood_cache = [f"hotpotqa={THIS_DIR / 'runs' / 'hotpotqa_ood.npz'}"]
    return args


def parse_ood_specs(specs: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--ood-cache expects 'name=path', got {s!r}")
        name, path = s.split("=", 1)
        out.append((name.strip(), Path(path.strip())))
    return out


def load_triviaqa_training(
    args: argparse.Namespace,
) -> dict[str, Any]:
    df = load_anchor_rows(args.triviaqa_anchor)
    sample0 = select_sample0_rows(df)
    targets = attach_targets(sample0)
    qid_to_local = question_index_map(sample0)

    blob = np.load(args.triviaqa_hidden, allow_pickle=True)
    cached_qids = list(blob["question_ids"])
    h_all = blob["hidden_prompt_last"]
    layers = list(map(int, blob["layer_indices"]))

    cache_idx = {qid: i for i, qid in enumerate(cached_qids)}
    qid_order = list(qid_to_local.keys())
    h_per_question = np.stack([h_all[cache_idx[q]] for q in qid_order], axis=0)
    rows_by_question = sample0["question_id"].map(qid_to_local).to_numpy()

    return {
        "h_per_question": h_per_question,
        "h_rows": h_per_question[rows_by_question],
        "y_corr": targets["y_corr"],
        "y_se": targets["y_se"],
        "y_anchor": targets["y_anchor"],
        "layers": layers,
    }


def load_ood_test(cache_path: Path) -> dict[str, Any]:
    blob = np.load(cache_path, allow_pickle=True)
    layers = list(map(int, blob["layer_indices"]))
    return {
        "h_per_question": blob["hidden_prompt_last"],
        "y_corr": blob["strict_correct"].astype(np.float32),
        "qids": list(blob["question_ids"]),
        "layers": layers,
        "answers": list(blob["answers"]) if "answers" in blob.files else [],
    }


def fit_and_predict(
    layer_idx: int,
    train: dict[str, Any],
    test: dict[str, Any],
    args: argparse.Namespace,
    layer_int: int,
) -> dict[str, np.ndarray]:
    """Train all probes at this layer on TriviaQA and predict on the OOD set."""
    X_tr = train["h_rows"][:, layer_idx, :].astype(np.float32)
    X_te = test["h_per_question"][:, layer_idx, :].astype(np.float32)
    y_corr = train["y_corr"]
    y_se = train["y_se"]

    out: dict[str, np.ndarray] = {}

    p_sep_r = RidgeProbe(alpha=args.ridge_alpha, seed=args.seed)
    p_sep_r.fit(X_tr, y_se)
    out["SEPs_ridge"] = -p_sep_r.predict_score(X_te)

    p_sep_l = LogRegProbe(C=args.logreg_c, seed=args.seed)
    p_sep_l.fit(X_tr, y_corr)
    out["SEPs_logreg"] = p_sep_l.predict_score(X_te)

    p_dcp = MLPProbe(
        hidden=args.mlp_hidden, out_dim=1, loss="bce",
        epochs=args.mlp_epochs, seed=args.seed,
    )
    p_dcp.fit(X_tr, y_corr)
    out["DCP_mlp"] = p_dcp.predict_score(X_te)

    return out


def paired_bootstrap_diff(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    n = len(y_true)
    diffs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        au_a = safe_auroc(y_true[idx], score_a[idx])
        au_b = safe_auroc(y_true[idx], score_b[idx])
        diffs[b] = au_a - au_b
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(diffs))
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    pval = float(2 * min(np.mean(diffs > 0), np.mean(diffs < 0)))
    return mean, lo, hi, pval


CALIBRATED = {"SEPs_logreg", "DCP_mlp"}


def evaluate_one_ood(
    name: str,
    cache_path: Path,
    train: dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    print(f"\n[{name}] load OOD cache: {cache_path}")
    test = load_ood_test(cache_path)
    if train["layers"] != test["layers"]:
        raise ValueError(
            f"[{name}] Layer mismatch: train={train['layers']} vs test={test['layers']}"
        )
    layers = train["layers"]
    y_te = test["y_corr"]
    print(
        f"[{name}] n={len(y_te)}, base_acc={float(y_te.mean()):.4f}, layers={layers}"
    )

    metric_rows: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, int], np.ndarray] = {}

    for li, layer in enumerate(layers):
        print(f"[{name}][layer {layer}] fitting probes...")
        preds = fit_and_predict(li, train, test, args, layer)
        for probe, scores in preds.items():
            score_cache[(probe, layer)] = scores
            metrics = summarize_metrics(y_te, scores, calibrated=(probe in CALIBRATED))
            metric_rows.append({
                "ood": name,
                "probe": probe,
                "layer": int(layer),
                "n": int(len(y_te)),
                "base_acc": float(y_te.mean()),
                **metrics,
            })

    metrics_df = pd.DataFrame(metric_rows)
    metrics_csv = out_dir / f"ood_{name}_metrics_long.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[{name}] wrote {metrics_csv}")

    best_rows: list[dict[str, Any]] = []
    for probe, grp in metrics_df.groupby("probe", sort=False):
        best = grp.sort_values("auroc", ascending=False).iloc[0]
        best_rows.append(best.to_dict())
    best_df = pd.DataFrame(best_rows)
    best_csv = out_dir / f"ood_{name}_best_per_probe.csv"
    best_df.to_csv(best_csv, index=False)

    rng = np.random.default_rng(args.seed)
    dcp_best_layer = int(
        metrics_df[metrics_df["probe"] == "DCP_mlp"].sort_values("auroc", ascending=False).iloc[0]["layer"]
    )
    boot_rows: list[dict[str, Any]] = []
    for probe in ["SEPs_ridge", "SEPs_logreg"]:
        other_best_layer = int(
            metrics_df[metrics_df["probe"] == probe].sort_values("auroc", ascending=False).iloc[0]["layer"]
        )
        a = score_cache[("DCP_mlp", dcp_best_layer)]
        b = score_cache[(probe, other_best_layer)]
        mean, lo, hi, pval = paired_bootstrap_diff(y_te, a, b, args.n_boot, rng)
        boot_rows.append({
            "ood": name,
            "probe_a": f"DCP_mlp@L{dcp_best_layer}",
            "probe_b": f"{probe}@L{other_best_layer}",
            "mean_diff": mean, "ci_low": lo, "ci_high": hi, "p_value": pval,
        })
    boot_df = pd.DataFrame(boot_rows)
    boot_csv = out_dir / f"ood_{name}_bootstrap_pairs.csv"
    boot_df.to_csv(boot_csv, index=False)
    print(f"[{name}] wrote {boot_csv}")
    print(boot_df.to_string(index=False))

    return {
        "name": name,
        "n": int(len(y_te)),
        "base_acc": float(y_te.mean()),
        "metrics_df": metrics_df,
        "best_df": best_df,
        "boot_df": boot_df,
    }


def render_combined_md(
    results: list[dict[str, Any]],
    out_path: Path,
    train_n_rows: int,
) -> None:
    md: list[str] = []
    md.append("# Teacher-Free Probes: TriviaQA -> Multi-OOD Evaluation\n")
    md.append(
        f"Trained on full TriviaQA ({train_n_rows} rows). "
        f"Each OOD dataset: greedy sample0 with `strict_correct` against ideal answers.\n"
    )

    md.append("\n## OOD setup\n")
    md.append("| dataset | n | base_acc |")
    md.append("| --- | --- | --- |")
    for r in results:
        md.append(f"| {r['name']} | {r['n']} | {r['base_acc']:.4f} |")

    md.append("\n## Best AUROC per probe (per OOD dataset)\n")
    cols = ["probe", "layer", "auroc", "aurc", "sel_acc@0.25", "sel_acc@0.50", "sel_acc@0.75", "brier", "ece"]
    for r in results:
        md.append(f"\n### {r['name']}  (n={r['n']}, base_acc={r['base_acc']:.4f})\n")
        md.append("| " + " | ".join(cols) + " |")
        md.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in r["best_df"].sort_values("auroc", ascending=False).iterrows():
            md.append("| " + " | ".join(
                f"{row[c]:.4f}" if isinstance(row[c], (int, float, np.floating)) and not pd.isna(row[c]) else str(row[c])
                for c in cols
            ) + " |")

    md.append("\n## DCP-MLP vs SEPs (paired bootstrap, per OOD dataset)\n")
    md.append("| OOD | comparison | mean ΔAUROC | 95% CI | p-value |")
    md.append("| --- | --- | --- | --- | --- |")
    for r in results:
        for _, row in r["boot_df"].iterrows():
            md.append(
                f"| {r['name']} | {row['probe_a']} vs {row['probe_b']} | "
                f"{row['mean_diff']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
                f"{row['p_value']:.3f} |"
            )

    md.append("\n## Per-probe AUROC across OOD datasets (best-layer)\n")
    pivot_rows: list[dict[str, Any]] = []
    for r in results:
        for _, row in r["best_df"].iterrows():
            pivot_rows.append({
                "ood": r["name"],
                "probe": row["probe"],
                "best_layer": int(row["layer"]),
                "auroc": float(row["auroc"]),
            })
    p_df = pd.DataFrame(pivot_rows)
    pivot = p_df.pivot_table(index="probe", columns="ood", values="auroc")
    pivot_layer = p_df.pivot_table(index="probe", columns="ood", values="best_layer")
    md.append("AUROC table:\n")
    md.append("| probe | " + " | ".join(pivot.columns) + " |")
    md.append("| --- | " + " | ".join(["---"] * len(pivot.columns)) + " |")
    for probe, row in pivot.iterrows():
        md.append(
            "| " + str(probe) + " | "
            + " | ".join(f"{row[c]:.4f}" for c in pivot.columns)
            + " |"
        )
    md.append("\nBest-layer table:\n")
    md.append("| probe | " + " | ".join(pivot_layer.columns) + " |")
    md.append("| --- | " + " | ".join(["---"] * len(pivot_layer.columns)) + " |")
    for probe, row in pivot_layer.iterrows():
        md.append(
            "| " + str(probe) + " | "
            + " | ".join(f"{int(row[c])}" for c in pivot_layer.columns)
            + " |"
        )

    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[done] wrote combined markdown -> {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] TriviaQA training cache + anchor labels")
    train = load_triviaqa_training(args)
    print(f"[load] training rows={train['h_rows'].shape[0]}, layers={train['layers']}")

    ood_specs = parse_ood_specs(args.ood_cache)
    print(f"[load] OOD datasets: {[(n, str(p)) for n, p in ood_specs]}")

    results: list[dict[str, Any]] = []
    for name, path in ood_specs:
        results.append(evaluate_one_ood(name, path, train, args, out_dir))

    render_combined_md(
        results,
        out_dir / "ood_combined_table.md",
        train_n_rows=train["h_rows"].shape[0],
    )


if __name__ == "__main__":
    main()
