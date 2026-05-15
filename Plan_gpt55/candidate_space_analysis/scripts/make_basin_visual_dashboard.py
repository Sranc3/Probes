#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


OUTPUT_DIR = Path("/zhutingqi/song/memo_for_paper/figures")
OUTPUT_PNG = OUTPUT_DIR / "basin_visual_dashboard.png"
OUTPUT_PDF = OUTPUT_DIR / "basin_visual_dashboard.pdf"
OUTPUT_MD = OUTPUT_DIR / "basin_visual_dashboard_explanation_zh.md"

PALETTE = {
    "blue": "#069DFF",
    "gray": "#808080",
    "green": "#A4E048",
    "black": "#010101",
}

PANELS = [
    {
        "title": "A. Candidate Space / Basin Map",
        "path": Path("/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25/plots/07_attractor_basin_map.png"),
        "note": "语义候选会聚成多个 answer basins；rescue/damage 不是随机散点。",
        "figure_note": "Candidates form structured answer basins; rescue/damage are not random outliers.",
        "remap_red_green": True,
        "remap_profile": "damage_light_red",
    },
    {
        "title": "B. Rescue Attractor Micrographs",
        "path": Path("/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25/plots/08_rescue_attractor_micrographs.png"),
        "note": "sample0 错时，附近或替代 basin 中确实存在可救回答案。",
        "figure_note": "When sample0 is wrong, alternative basins often contain rescue answers.",
        "remap_red_green": True,
    },
    {
        "title": "C. Entropy Regime Map",
        "path": Path("/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260429_063326_entropy_anatomy/plots/01_entropy_regime_map.png"),
        "note": "不同 basin regime 在 entropy/logprob/mass 空间中有结构差异。",
        "figure_note": "Basin regimes show structure in entropy / logprob / mass space.",
        "remap_red_green": True,
    },
    {
        "title": "D. Stable Correct vs Hallucination",
        "path": Path("/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260429_063326_entropy_anatomy/plots/02_stable_correct_vs_hallucination.png"),
        "note": "低 entropy 有两张脸：稳定正确，也可能稳定幻觉。",
        "figure_note": "Low entropy has two faces: stable correctness and stable hallucination.",
        "remap_red_green": True,
    },
    {
        "title": "E. Rescue / Damage Anatomy",
        "path": Path("/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260429_063326_entropy_anatomy/plots/03_rescue_damage_entropy_anatomy.png"),
        "note": "rescue 与 damage 的区分不能只靠单一置信度信号。",
        "figure_note": "Rescue vs damage cannot be separated by confidence alone.",
        "remap_red_green": True,
        "remap_profile": "rescue_green",
    },
    {
        "title": "F. Token Entropy Waveform Bands",
        "path": Path("/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260430_021911_entropy_waveform_analysis_v2/plots/06_correct_vs_wrong_waveform_bands.png"),
        "note": "正确/错误 basin 的 token-level entropy 波形存在可视化差异。",
        "figure_note": "Correct and wrong basins differ in token-level entropy waveforms.",
    },
    {
        "title": "G. Waveform Feature Distributions",
        "path": Path("/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260430_021911_entropy_waveform_analysis_v2/plots/07_entropy_feature_distributions.png"),
        "note": "waveform summary 有信号，但当前不足以单独做强 controller。",
        "figure_note": "Waveform summaries carry signal, but overlap limits controller strength.",
    },
    {
        "title": "H. Spike vs Min-Prob Scatter",
        "path": Path("/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260430_021911_entropy_waveform_analysis_v2/plots/09_entropy_spike_vs_minprob_scatter.png"),
        "note": "stable hallucination 可能在局部 token 上仍出现犹豫峰。",
        "figure_note": "Stable hallucinations may still contain local token-level uncertainty spikes.",
    },
    {
        "title": "I. No-Leak Cost / Accuracy Frontier",
        "path": Path("/zhutingqi/song/memo_for_paper/figures/cost_accuracy_pareto_frontier.png"),
        "note": "合规 controller 目前只有小幅收益；adaptive 比固定高成本数值 verifier 更有效。",
        "figure_note": "Compliant controllers show modest gains; adaptive policies are the current lead.",
    },
]


def make_dashboard() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(18, 14.5))
    fig.suptitle(
        "Answer-Basin Phenomena Dashboard: Structure, Entropy Anatomy, Waveforms, and No-Leak Control",
        fontsize=16,
        fontweight="bold",
        color=PALETTE["black"],
        y=0.995,
    )
    for ax, panel in zip(axes.flat, PANELS):
        ax.axis("off")
        ax.set_title(panel["title"], fontsize=11, fontweight="bold", color=PALETTE["black"], pad=6)
        ax.set_facecolor("#ffffff")
        path = panel["path"]
        if path.exists():
            image = mpimg.imread(path)
            if panel.get("remap_red_green"):
                image = remap_dashboard_pixels(image, str(panel.get("remap_profile", "default")))
            ax.imshow(image)
        else:
            ax.text(0.5, 0.5, f"Missing:\n{path}", ha="center", va="center", wrap=True)
        ax.text(
            0.01,
            -0.05,
            panel["figure_note"],
            transform=ax.transAxes,
            fontsize=8,
            color=PALETTE["gray"],
            va="top",
            wrap=True,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.8, w_pad=0.8)
    fig.savefig(OUTPUT_PNG)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)


def remap_dashboard_pixels(image, profile: str = "default"):
    import numpy as np
    from matplotlib import colors as mcolors

    arr = np.asarray(image).astype("float32")
    if arr.max() > 1.0:
        arr /= 255.0
    if arr.shape[-1] == 4:
        rgb = arr[..., :3]
        alpha = arr[..., 3:]
    else:
        rgb = arr[..., :3]
        alpha = None
    hsv = mcolors.rgb_to_hsv(rgb)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]
    visible_color = (sat > 0.08) & (val > 0.12) & (val < 0.98)

    # Source plots use muted browns/pinks/reds and greens, not only saturated red/green.
    red_or_pink = visible_color & ((hue < 0.075) | (hue > 0.90) | ((hue > 0.80) & (sat > 0.10)))
    brown_or_orange = visible_color & (hue >= 0.075) & (hue < 0.17) & (sat > 0.12)
    green_or_yellow_green = visible_color & (hue >= 0.17) & (hue < 0.47)
    blue_or_cyan = visible_color & (hue >= 0.47) & (hue < 0.72)

    palette = {
        "blue": np.array([0x06, 0x9D, 0xFF], dtype="float32") / 255.0,
        "gray": np.array([0x80, 0x80, 0x80], dtype="float32") / 255.0,
        "green": np.array([0xA4, 0xE0, 0x48], dtype="float32") / 255.0,
        "light_red": np.array([0xFF, 0xA0, 0xA0], dtype="float32") / 255.0,
    }
    luminance = val[..., None]
    red_palette = palette["light_red"] if profile == "damage_light_red" else palette["blue"]
    red_target = 0.34 * luminance + 0.66 * red_palette
    brown_target = 0.40 * luminance + 0.60 * palette["gray"]
    green_target = 0.34 * luminance + 0.66 * palette["green"]
    yellow_green_target = 0.28 * luminance + 0.72 * palette["green"]
    blue_to_green_target = 0.30 * luminance + 0.70 * palette["green"]

    red_strength = np.clip(sat * 1.35, 0.0, 1.0)[..., None]
    brown_strength = np.clip(sat * 1.20, 0.0, 1.0)[..., None]
    green_strength = np.clip(sat * 1.30, 0.0, 1.0)[..., None]
    remapped = rgb.copy()
    remapped = np.where(red_or_pink[..., None], (1.0 - red_strength) * remapped + red_strength * red_target, remapped)
    remapped = np.where(brown_or_orange[..., None], (1.0 - brown_strength) * remapped + brown_strength * brown_target, remapped)
    remapped = np.where(green_or_yellow_green[..., None], (1.0 - green_strength) * remapped + green_strength * green_target, remapped)
    if profile == "rescue_green":
        blue_strength = np.clip(sat * 1.35, 0.0, 1.0)[..., None]
        remapped = np.where(blue_or_cyan[..., None], (1.0 - blue_strength) * remapped + blue_strength * blue_to_green_target, remapped)
    yellow_green_mask = green_or_yellow_green & (hue < 0.27)
    remapped = np.where(yellow_green_mask[..., None], 0.50 * remapped + 0.50 * yellow_green_target, remapped)

    # Keep non-targeted colors intact; only soften very loud residual colors in targeted panels.
    loud_color = visible_color & (sat > 0.62)
    remapped = np.where(loud_color[..., None], 0.90 * remapped + 0.10 * palette["gray"], remapped)
    rgb = np.clip(remapped, 0.0, 1.0)
    if alpha is not None:
        return np.concatenate([rgb, alpha], axis=-1)
    return rgb


def write_explanation() -> None:
    lines = [
        "# Basin Visual Dashboard 图解",
        "",
        "这张大图只整合现象图和 no-leak 合规结果，不展示已经排除的 leaky fixed-8 方法。",
        "",
        "## 怎么读这张图",
        "",
        "| Panel | 看什么 | 对 controller 的启发 |",
        "| --- | --- | --- |",
    ]
    controller_hints = {
        "A": "controller 不应只看单个答案，应判断当前答案所在 basin 是否可信。",
        "B": "rescue 机会真实存在，但需要判断 alternative basin 是否 factual，而不只是更稳定。",
        "C": "数值 geometry 有结构信号，可作为轻量输入，但不能直接等同 correctness。",
        "D": "低熵不是充分条件；必须区分 stable correct 与 stable hallucination。",
        "E": "damage veto 需要比 confidence 更强的 factual/risk signal。",
        "F": "waveform 可作为 prefix 或 sequence feature，而不是简单均值阈值。",
        "G": "summary feature 有分布差异，但重叠仍大，解释了 controller 效果有限。",
        "H": "局部 spike/min-prob 可能帮助识别稳定幻觉内部的微观不稳定。",
        "I": "当前合规主线应优先优化 adaptive / prefix policy，而不是 fixed high-compute numeric verifier。",
    }
    for panel in PANELS:
        key = panel["title"].split(".")[0]
        lines.append(f"| {key} | {panel['note']} | {controller_hints[key]} |")
    lines.extend(
        [
            "",
            "## 当前仍成立的 basin 发现",
            "",
            "- answer candidates 会形成语义 basin，而不是完全随机散布。",
            "- sample0 错误时，候选空间中经常存在可 rescue 的 alternative basin。",
            "- stable hallucination basin 仍成立：错误答案也会低熵、高一致性地聚集。",
            "- entropy waveform 仍是有用的机制信号，但当前 summary 版本不是强 controller。",
            "",
            "## 需要降级的结论",
            "",
            "- 原始 fixed-8 learned verifier 的强收益来自 label-derived feature，不能作为方法结果。",
            "- 现有 no-leak 数值 basin geometry 只能带来小幅收益，说明 controller 需要更强的 semantic/factual verifier。",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    make_dashboard()
    write_explanation()
    print(f"Wrote dashboard to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
