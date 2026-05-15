"""Compute extended selective-prediction metrics for ID + OOD across all bases.

For each base in {qwen7b, llama3b, qwen72b}:

ID  (TriviaQA, cv_5fold, n=1000):
    - Read ``runs/<base>/probe_predictions.csv`` directly (no re-fitting).
    - For each probe pick the best layer by AUROC.
    - Save per-item predictions and compute extended metrics.

OOD (HotpotQA + NQ-Open, n=500 each):
    - Re-fit DCP-MLP / SEPs-LR / SEPs-Ridge on the FULL TriviaQA cache
      (no folds), using cached hidden states.
    - Apply on the OOD hidden-state cache.
    - For each probe pick the best layer by OOD AUROC and dump per-item
      predictions for that layer.

Outputs (per base):
    results/<base>/extended_metrics_long.csv
    results/<base>/per_item_predictions_<dataset>_best.csv
        cols: probe, layer, dataset, base, question_id, y_true, y_score

Top-level:
    results/extended_metrics_long.csv          (concat of all bases)
    results/per_item_predictions_all.csv       (concat for plotting)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR / "shared"))
sys.path.insert(0, str(THIS_DIR / "scripts"))

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
    extended_metrics,
    safe_auroc,
)
from run_all_models import BASES  # noqa: E402

CALIBRATED_PROBES = {"SEPs_logreg", "DCP_mlp", "ARD_ridge", "ARD_mlp"}
ID_PROBES = ["DCP_mlp", "SEPs_logreg", "SEPs_ridge", "ARD_mlp", "ARD_ridge"]
OOD_PROBES = ["DCP_mlp", "SEPs_logreg", "SEPs_ridge"]
TRIVIA_ANCHOR = (
    "/zhutingqi/song/Plan_gpt55/cross_model_anchor/runs/"
    "run_20260513_012413_gptoss_anchor_triviaqa_full500_k4_tok256/"
    "qwen_candidate_anchor_rows_final_only.csv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def best_layer_per_probe(df: pd.DataFrame, group_col: str = "probe") -> dict[str, int]:
    """Return {probe -> best layer by AUROC} from a per-item CSV (regime-filtered)."""
    rows = []
    for (probe, layer), grp in df.groupby([group_col, "layer"]):
        au = safe_auroc(grp["y_true"].to_numpy(), grp["y_score"].to_numpy())
        rows.append({"probe": probe, "layer": int(layer), "auroc": au})
    summary = pd.DataFrame(rows)
    out = {}
    for probe, sub in summary.groupby("probe"):
        out[probe] = int(sub.sort_values("auroc", ascending=False).iloc[0]["layer"])
    return out


def collapse_seeds(grp: pd.DataFrame) -> pd.DataFrame:
    """Average y_score across seeds per question (cv_5fold typically has 2 seeds)."""
    out = (
        grp.groupby("question_id", as_index=False)
        .agg(y_true=("y_true", "first"), y_score=("y_score", "mean"))
    )
    return out


# ---------------------------------------------------------------------------
# Per-base ID processing
# ---------------------------------------------------------------------------


def process_id(tag: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = THIS_DIR / "runs" / tag / "probe_predictions.csv"
    print(f"\n[{tag}] reading {pred_path}")
    df = pd.read_csv(pred_path)
    df = df[df["regime"] == "cv_5fold"].copy()

    # Pick best layer per probe by AUROC over (averaged-across-seeds) per item
    avg_rows: list[dict[str, Any]] = []
    for (probe, layer), grp in df.groupby(["probe", "layer"]):
        sub = collapse_seeds(grp)
        au = safe_auroc(sub["y_true"].to_numpy(), sub["y_score"].to_numpy())
        avg_rows.append({"probe": probe, "layer": int(layer), "auroc": au})
    summary = pd.DataFrame(avg_rows)
    best_layers = {
        probe: int(sub.sort_values("auroc", ascending=False).iloc[0]["layer"])
        for probe, sub in summary.groupby("probe")
    }
    print(f"[{tag}] best ID layers: {best_layers}")

    metric_rows: list[dict[str, Any]] = []
    per_item_rows: list[pd.DataFrame] = []
    for probe in ID_PROBES:
        if probe not in best_layers:
            continue
        L = best_layers[probe]
        sub = df[(df["probe"] == probe) & (df["layer"] == L)]
        sub = collapse_seeds(sub)
        y = sub["y_true"].to_numpy()
        s = sub["y_score"].to_numpy()
        m = extended_metrics(y, s, calibrated=(probe in CALIBRATED_PROBES))
        metric_rows.append({
            "base": tag, "dataset": "id", "probe": probe, "layer": L,
            "n": int(len(y)), "base_acc": float(np.mean(y)), **m,
        })
        per_item_rows.append(sub.assign(probe=probe, layer=L,
                                        dataset="id", base=tag))

    metrics_df = pd.DataFrame(metric_rows)
    per_item_df = pd.concat(per_item_rows, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_item_df.to_csv(out_dir / "per_item_predictions_id_best.csv", index=False)
    print(f"[{tag}] wrote per-item ID predictions ({len(per_item_df)} rows)")
    return metrics_df, per_item_df


# ---------------------------------------------------------------------------
# Per-base OOD processing (re-fit on TriviaQA, predict on OOD)
# ---------------------------------------------------------------------------


def load_triviaqa_training(tag: str) -> dict[str, Any]:
    df = load_anchor_rows(TRIVIA_ANCHOR)
    sample0 = select_sample0_rows(df)
    targets = attach_targets(sample0)
    qid_to_local = question_index_map(sample0)

    blob = np.load(THIS_DIR / "runs" / tag / "hidden_states.npz", allow_pickle=True)
    cached_qids = list(blob["question_ids"])
    h_all = blob["hidden_prompt_last"]
    layers = list(map(int, blob["layer_indices"]))

    cache_idx = {qid: i for i, qid in enumerate(cached_qids)}
    qid_order = list(qid_to_local.keys())
    h_per_question = np.stack([h_all[cache_idx[q]] for q in qid_order], axis=0)
    rows_by_question = sample0["question_id"].map(qid_to_local).to_numpy()
    return {
        "h_rows": h_per_question[rows_by_question],
        "y_corr": targets["y_corr"],
        "y_se": targets["y_se"],
        "layers": layers,
    }


def load_ood_cache(path: Path) -> dict[str, Any]:
    blob = np.load(path, allow_pickle=True)
    return {
        "h_per_question": blob["hidden_prompt_last"],
        "y_corr": blob["strict_correct"].astype(np.float32),
        "qids": list(blob["question_ids"]),
        "layers": list(map(int, blob["layer_indices"])),
    }


def fit_ood_probes(
    train: dict[str, Any], test: dict[str, Any],
    layer_idx: int, seed: int = 0,
) -> dict[str, np.ndarray]:
    X_tr = train["h_rows"][:, layer_idx, :].astype(np.float32)
    X_te = test["h_per_question"][:, layer_idx, :].astype(np.float32)
    y_corr = train["y_corr"]
    y_se = train["y_se"]

    out: dict[str, np.ndarray] = {}
    p_sep_r = RidgeProbe(alpha=1.0, seed=seed)
    p_sep_r.fit(X_tr, y_se)
    out["SEPs_ridge"] = -p_sep_r.predict_score(X_te)

    p_sep_l = LogRegProbe(C=1.0, seed=seed)
    p_sep_l.fit(X_tr, y_corr)
    out["SEPs_logreg"] = p_sep_l.predict_score(X_te)

    p_dcp = MLPProbe(hidden=128, out_dim=1, loss="bce", epochs=200, seed=seed)
    p_dcp.fit(X_tr, y_corr)
    out["DCP_mlp"] = p_dcp.predict_score(X_te)
    return out


def process_ood(
    tag: str, ood_name: str, ood_cache_path: Path,
    train: dict[str, Any], out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n[{tag}|{ood_name}] loading {ood_cache_path}")
    test = load_ood_cache(ood_cache_path)
    layers = train["layers"]
    if layers != test["layers"]:
        raise ValueError(f"layer mismatch ID={layers} OOD={test['layers']}")
    y_te = test["y_corr"]

    score_cache: dict[tuple[str, int], np.ndarray] = {}
    layer_metrics: list[dict[str, Any]] = []
    for li, layer in enumerate(layers):
        preds = fit_ood_probes(train, test, li)
        for probe, scores in preds.items():
            score_cache[(probe, layer)] = scores
            au = safe_auroc(y_te, scores)
            layer_metrics.append({"probe": probe, "layer": int(layer), "auroc": au})
        print(f"[{tag}|{ood_name}] layer {layer}: " +
              ", ".join(f"{p}={safe_auroc(y_te, preds[p]):.3f}" for p in preds))

    layer_df = pd.DataFrame(layer_metrics)
    best_layers = {
        probe: int(sub.sort_values("auroc", ascending=False).iloc[0]["layer"])
        for probe, sub in layer_df.groupby("probe")
    }
    print(f"[{tag}|{ood_name}] best layers: {best_layers}")

    metric_rows: list[dict[str, Any]] = []
    per_item_rows: list[pd.DataFrame] = []
    for probe in OOD_PROBES:
        if probe not in best_layers:
            continue
        L = best_layers[probe]
        s = score_cache[(probe, L)]
        m = extended_metrics(y_te, s, calibrated=(probe in CALIBRATED_PROBES))
        metric_rows.append({
            "base": tag, "dataset": ood_name, "probe": probe, "layer": L,
            "n": int(len(y_te)), "base_acc": float(np.mean(y_te)), **m,
        })
        per_item_rows.append(pd.DataFrame({
            "question_id": test["qids"],
            "y_true": y_te,
            "y_score": s,
            "probe": probe, "layer": L,
            "dataset": ood_name, "base": tag,
        }))

    metrics_df = pd.DataFrame(metric_rows)
    per_item_df = pd.concat(per_item_rows, ignore_index=True)
    per_item_df.to_csv(out_dir / f"per_item_predictions_{ood_name}_best.csv",
                       index=False)
    print(f"[{tag}|{ood_name}] wrote per-item OOD predictions ({len(per_item_df)} rows)")
    return metrics_df, per_item_df


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    base_tags = [b.tag for b in BASES]
    print(f"[run] bases={base_tags}")

    all_metrics: list[pd.DataFrame] = []
    all_per_item: list[pd.DataFrame] = []

    for tag in base_tags:
        out_dir = THIS_DIR / "results" / tag
        # ID
        m_id, p_id = process_id(tag, out_dir)
        all_metrics.append(m_id)
        all_per_item.append(p_id)
        # OOD
        train = load_triviaqa_training(tag)
        for ood_name, fname in [("hotpotqa", "hotpotqa_ood.npz"),
                                ("nq", "nq_ood.npz")]:
            ood_path = THIS_DIR / "runs" / tag / fname
            m_ood, p_ood = process_ood(tag, ood_name, ood_path, train, out_dir)
            all_metrics.append(m_ood)
            all_per_item.append(p_ood)
        # per-base extended metrics
        per_base = pd.concat([m_id] +
                             [m for m in all_metrics
                              if not m.empty and m.iloc[0]["base"] == tag and
                              m.iloc[0]["dataset"] != "id"],
                             ignore_index=True)
        per_base.to_csv(out_dir / "extended_metrics_long.csv", index=False)
        print(f"[{tag}] wrote extended_metrics_long.csv ({len(per_base)} rows)")

    full_metrics = pd.concat(all_metrics, ignore_index=True)
    full_per_item = pd.concat(all_per_item, ignore_index=True)
    out_root = THIS_DIR / "results"
    full_metrics.to_csv(out_root / "extended_metrics_long.csv", index=False)
    full_per_item.to_csv(out_root / "per_item_predictions_all.csv", index=False)
    print(f"\n[done] {out_root / 'extended_metrics_long.csv'} ({len(full_metrics)} rows)")
    print(f"[done] {out_root / 'per_item_predictions_all.csv'} ({len(full_per_item)} rows)")


if __name__ == "__main__":
    main()
