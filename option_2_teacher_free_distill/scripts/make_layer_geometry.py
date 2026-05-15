"""Layer-wise geometric analysis: linear separability + CKA between layers.

Two complementary measures, both computed on the cached prompt-last hidden
states for each base model on TriviaQA ID:

1. Linear separability vs depth
   -- Train logistic regression on each layer's hidden state to predict
      strict_correct (Qwen-7B's anchor labels).  Report cv_5fold AUROC.
   -- Reveals "where confidence becomes linearly readable."

2. Centered Kernel Alignment (CKA) between adjacent layers
   -- CKA(L_i, L_j) measures representation similarity.
   -- Adjacent-layer CKA tends to be ~1 in late layers (representations
      stop changing once the answer is locked in).  The point at which
      CKA(L,L+1) starts approaching 1 is the "answer-locked" depth, often
      where probe AUROC peaks then degrades.
   -- We use linear CKA (Kornblith+ 2019) for speed.

Output:
  results/layer_geometry_{base}.csv   -- per-layer rows (auroc + cka)
  reports/dashboard_layer_geometry.png -- 2-panel dashboard for all 3 bases
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR / "shared"))
sys.path.insert(0, str(THIS_DIR / "scripts"))
from data_utils import (  # noqa: E402
    attach_targets,
    load_anchor_rows,
    question_index_map,
    select_sample0_rows,
)
from run_all_models import BASES  # noqa: E402

ANCHOR_CSV = (
    "/zhutingqi/song/Plan_gpt55/cross_model_anchor/runs/"
    "run_20260513_012413_gptoss_anchor_triviaqa_full500_k4_tok256/"
    "qwen_candidate_anchor_rows_final_only.csv"
)


def load_aligned_hidden(tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Return (h[Q, L, D], y_corr[Q], groups[Q], layer_indices)."""
    df = load_anchor_rows(ANCHOR_CSV)
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
    h_rows = h_per_question[rows_by_question]
    y_corr = targets["y_corr"]
    groups = sample0["question_id"].to_numpy()
    return h_rows, y_corr, groups, layers


def linear_separability_per_layer(
    h: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5,
) -> list[float]:
    """Run group-aware 5-fold logreg on each layer and return AUROC per layer."""
    aurocs = []
    L = h.shape[1]
    kf = GroupKFold(n_splits=n_splits)
    for li in range(L):
        X = h[:, li, :].astype(np.float32)
        if not np.isfinite(X).all():
            X = np.nan_to_num(X)
        y_score = np.zeros(len(y))
        for tr, te in kf.split(X, y, groups=groups):
            scaler = StandardScaler().fit(X[tr])
            Xs_tr = scaler.transform(X[tr])
            Xs_te = scaler.transform(X[te])
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
            clf.fit(Xs_tr, y[tr].astype(int))
            y_score[te] = clf.predict_proba(Xs_te)[:, 1]
        aurocs.append(float(roc_auc_score(y, y_score)))
    return aurocs


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith+ 2019). X: [N, D1], Y: [N, D2]."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # Use the "biased" Hilbert-Schmidt formulation for speed.
    XY = X.T @ Y                 # [D1, D2]
    XX = X.T @ X                 # [D1, D1]
    YY = Y.T @ Y                 # [D2, D2]
    num = float(np.sum(XY * XY))   # ||X^T Y||_F^2
    den = float(np.sqrt(np.sum(XX * XX)) * np.sqrt(np.sum(YY * YY)))
    if den == 0:
        return float("nan")
    return num / den


def adjacent_cka_per_layer(h_per_question: np.ndarray) -> list[float]:
    """For adjacent layers (L, L+1), compute linear CKA on per-question reps.

    h_per_question: [Q, L, D]
    Returns CKA values of length L-1.
    """
    L = h_per_question.shape[1]
    out: list[float] = []
    for li in range(L - 1):
        out.append(linear_cka(h_per_question[:, li, :].astype(np.float32),
                              h_per_question[:, li + 1, :].astype(np.float32)))
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

BASE_PRETTY = {"qwen7b": "Qwen2.5-7B", "llama3b": "Llama-3.2-3B", "qwen72b": "Qwen2.5-72B"}
BASE_COLORS = {"qwen7b": "#3182bd", "llama3b": "#9ecae1", "qwen72b": "#08519c"}
BASE_MARKERS = {"qwen7b": "o", "llama3b": "s", "qwen72b": "^"}
BASE_N_LAYERS = {b.tag: b.num_hidden_layers for b in BASES}


def main() -> None:
    rows: list[dict] = []
    base_to_data: dict[str, dict] = {}

    for tag in ["qwen7b", "llama3b", "qwen72b"]:
        print(f"\n[{tag}] loading hidden states...")
        h_rows, y, groups, layers = load_aligned_hidden(tag)
        # h_rows is [N=1000, L=8, D]; we need per-question reps for CKA.
        # Get per-question reps via blob:
        blob = np.load(THIS_DIR / "runs" / tag / "hidden_states.npz", allow_pickle=True)
        h_per_question = blob["hidden_prompt_last"]  # [Q=500, L=8, D]
        print(f"[{tag}] h_rows={h_rows.shape}, h_per_q={h_per_question.shape}, "
              f"y={y.shape}, layers={layers}")

        print(f"[{tag}] computing linear separability per layer...")
        au = linear_separability_per_layer(h_rows, y, groups)
        print(f"[{tag}] AUROC per layer: " +
              ", ".join(f"L{layers[i]}={au[i]:.3f}" for i in range(len(au))))

        print(f"[{tag}] computing adjacent CKA per layer...")
        cka = adjacent_cka_per_layer(h_per_question)
        print(f"[{tag}] CKA(L,L+1): " +
              ", ".join(f"{layers[i]}->{layers[i+1]}={cka[i]:.3f}" for i in range(len(cka))))

        n_lay = BASE_N_LAYERS[tag]
        for i, L in enumerate(layers):
            rows.append({
                "base": tag, "layer": int(L),
                "norm_depth": L / n_lay,
                "linear_sep_auroc": au[i],
                "cka_to_next": cka[i] if i < len(cka) else float("nan"),
            })
        base_to_data[tag] = {
            "layers": layers, "auroc": au, "cka": cka,
        }

    df = pd.DataFrame(rows)
    out_csv = THIS_DIR / "results" / "layer_geometry.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[done] {out_csv}")
    print(df.to_string(index=False))

    # 2-panel dashboard: AUROC vs depth, CKA vs depth.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Layer Geometry: Where Hidden State Becomes Linearly Readable\n"
        "Linear separability (left) & adjacent-layer CKA (right) vs normalised depth",
        fontsize=12, fontweight="bold", y=1.02,
    )

    ax = axes[0]
    for tag in ["qwen7b", "llama3b", "qwen72b"]:
        d = base_to_data[tag]
        depths = [L / BASE_N_LAYERS[tag] for L in d["layers"]]
        ax.plot(depths, d["auroc"],
                marker=BASE_MARKERS[tag], color=BASE_COLORS[tag],
                linewidth=2.0, markersize=7, label=BASE_PRETTY[tag])
    ax.axvline(0.71, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(0.715, 0.55, "  ref 0.71", fontsize=8, alpha=0.7)
    ax.set_xlabel("Normalised depth = layer / N")
    ax.set_ylabel("Linear separability  (AUROC of logreg on this layer)")
    ax.set_title("Linear separability — where does confidence become linearly readable?")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.5, 0.85)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    for tag in ["qwen7b", "llama3b", "qwen72b"]:
        d = base_to_data[tag]
        # CKA(L_i, L_{i+1}) plotted at midpoint of normalised depths
        midpoints = [
            ((d["layers"][i] + d["layers"][i + 1]) / 2.0) / BASE_N_LAYERS[tag]
            for i in range(len(d["cka"]))
        ]
        ax.plot(midpoints, d["cka"],
                marker=BASE_MARKERS[tag], color=BASE_COLORS[tag],
                linewidth=2.0, markersize=7, label=BASE_PRETTY[tag])
    ax.axvline(0.71, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Normalised depth (midpoint of L → L+1)")
    ax.set_ylabel("Linear CKA(L, L+1)")
    ax.set_title("Adjacent-layer CKA — where does the representation \"lock in\"?")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(0.02, 0.96, "high similarity (locked in)", fontsize=7, color="gray", alpha=0.7)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    out = THIS_DIR / "reports" / "dashboard_layer_geometry.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[done] dashboard_layer_geometry -> {out}")


if __name__ == "__main__":
    main()
