"""Render results.json into PNG charts for the report / slides."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

RANDOM = "#2a78d6"  # series 1: random window split (current train_user.py protocol)
HONEST = "#eb6834"  # series 2: grouped / held-out split
NEUTRAL = "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": ["Hiragino Sans GB", "PingFang HK", "Arial Unicode MS", "sans-serif"],
    "axes.edgecolor": "#d6d5d0", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#ecebe7", "grid.linewidth": 0.8, "font.size": 11,
})


def _bar_labels(ax, bars):
    for bar in bars:
        ax.annotate(f"{bar.get_height() * 100:.0f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)


def _chance(ax, y, label, x_right=None):
    ax.axhline(y, color=NEUTRAL, linestyle="--", linewidth=1.2, zorder=3)
    ax.text(0.005, y, label, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9, color=INK2,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"), zorder=4)


def leakage_chart(r):
    groups = [
        ("自采数据\n左/右/静息 (3类)", r["own_data"]["FinalUnifiedModel"]["random_window_split"]["acc"],
         r["own_data"]["FinalUnifiedModel"]["block_grouped_cv"]["acc"], 1 / 3),
        ("Mendeley MindWave\n注视 左/右/前 (3类)", r["mendeley"]["gaze_left_right_front"]["FinalUnifiedModel"]["random_window_split"]["acc"],
         r["mendeley"]["gaze_left_right_front"]["FinalUnifiedModel"]["trial_grouped_cv"]["acc"], 1 / 3),
        ("PhysioNet Fp1\n左/右/静息 (3类)", r["physionet"]["leakage_fp1_3class"]["random_window_split"]["acc"],
         r["physionet"]["leakage_fp1_3class"]["subject_grouped"]["acc"], 1 / 3),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
    xs = range(len(groups))
    w = 0.36
    b1 = ax.bar([x - w / 2 - 0.01 for x in xs], [g[1] for g in groups], w, color=RANDOM, label="随机窗口划分（当前 train_user.py）")
    b2 = ax.bar([x + w / 2 + 0.01 for x in xs], [g[2] for g in groups], w, color=HONEST, label="按试次/被试分组划分（真实在线场景）")
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    _chance(ax, 1 / 3, "随机猜测 33%", len(groups) - 0.45)
    ax.set_xticks(list(xs), [g[0] for g in groups])
    ax.set_ylim(0, 1.08); ax.set_ylabel("测试准确率")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("同一模型（FinalUnifiedModel），只改变训练/测试划分方式", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "leakage.png"); plt.close(fig)


def physionet_channel_chart(r):
    within = r["physionet"]["within_subject_lr"]
    rows = [
        ("Fp1\n(脑环位置)", within["FP1/neurosky_style"]["mean"]),
        ("C3", within["C3/neurosky_style"]["mean"]),
        ("C4", within["C4/neurosky_style"]["mean"]),
        ("Cz", within["CZ/neurosky_style"]["mean"]),
        ("C3+C4\n细粒度频谱", within["C3+C4/fine_psd"]["mean"]),
        ("64 导\nCSP", r["physionet"]["within_subject_lr_csp64"]["mean"]),
    ]
    sig = r["physionet"]["within_subject_lr_sig_threshold"]
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
    colors = [HONEST] * 4 + [NEUTRAL, RANDOM]
    bars = ax.bar(range(len(rows)), [v for _, v in rows], 0.6, color=colors)
    _bar_labels(ax, bars)
    _chance(ax, 0.5, "随机猜测 50%", len(rows) - 0.6)
    ax.axhline(sig, color=INK2, linestyle=":", linewidth=1)
    ax.text(1.5, sig + 0.01, f"单人显著性阈值 {sig * 100:.0f}%（p<0.05）", fontsize=9, color=INK2)
    ax.set_xticks(range(len(rows)), [n for n, _ in rows])
    ax.set_ylim(0.3, 0.8); ax.set_ylabel("被试内 5 折平均准确率")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("PhysioNet 左手 vs 右手运动想象（105 名被试）", color=INK, fontsize=12, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "physionet_channels.png"); plt.close(fig)


def mendeley_chart(r):
    m = r["mendeley"]
    tasks = [("专注 vs 走神\n(2类)", "focus_vs_notfocus", 0.5), ("注视 左/右/前\n(3类)", "gaze_left_right_front", 1 / 3)]
    protocols = [("random_window_split", "随机窗口划分", RANDOM), ("trial_grouped_cv", "按试次分组", HONEST), ("cross_subject", "跨被试", NEUTRAL)]
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
    w = 0.26
    for j, (key, name, color) in enumerate(protocols):
        bars = ax.bar([i + (j - 1) * (w + 0.02) for i in range(len(tasks))],
                      [m[t]["FinalUnifiedModel"][key]["acc"] for _, t, _ in tasks], w, color=color, label=name)
        _bar_labels(ax, bars)
    for i, (_, _, chance) in enumerate(tasks):
        ax.hlines(chance, i - 0.45, i + 0.45, color=INK2, linestyle="--", linewidth=1.2, zorder=3)
        ax.text(i - 0.45, chance, f"随机 {chance * 100:.0f}%", ha="right", va="center", fontsize=9, color=INK2)
    ax.set_xticks(range(len(tasks)), [t[0] for t in tasks])
    ax.set_xlim(-0.75, len(tasks) - 0.5)
    ax.set_ylim(0, 1.08); ax.set_ylabel("测试准确率")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Mendeley NeuroSky 数据集（真实 MindWave，2 名被试）", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "mendeley.png"); plt.close(fig)


def capability_chart(r):
    """What one frontal channel can / cannot do, honest (held-out) accuracy only."""

    p, m = r["physionet"], r["mendeley"]
    rows = [
        ("专注 vs 走神\nMendeley·真实脑环", m["focus_vs_notfocus"]["FinalUnifiedModel"]["trial_grouped_cv"]["acc"], 0.5),
        ("想象任务 vs 静息\nPhysioNet·Fp1", p["within_subject_mi_vs_rest"]["FP1/neurosky_log"]["mean"], 0.5),
        ("注视 左/右/前\nMendeley·真实脑环", m["gaze_left_right_front"]["FinalUnifiedModel"]["trial_grouped_cv"]["acc"], 1 / 3),
        ("想象 左手 vs 右手\nPhysioNet·Fp1", p["within_subject_lr"]["FP1/neurosky_style"]["mean"], 0.5),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
    colors = [HONEST, HONEST, NEUTRAL, NEUTRAL]
    bars = ax.bar(range(len(rows)), [v for _, v, _ in rows], 0.6, color=colors)
    _bar_labels(ax, bars)
    for i, (_, _, chance) in enumerate(rows):
        ax.hlines(chance, i - 0.36, i + 0.36, color=INK, linestyle="--", linewidth=1.4, zorder=3)
    ax.text(3.36, 0.98, "虚线 = 该任务的随机猜测水平", ha="right", fontsize=9, color=INK2)
    ax.set_xticks(range(len(rows)), [n for n, _, _ in rows], fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel("留出测试准确率（无泄漏）")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("单通道前额脑环：能做什么、不能做什么", color=INK, fontsize=12, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "capability.png"); plt.close(fig)


def main() -> int:
    results = json.loads((HERE / "results.json").read_text())
    OUT.mkdir(exist_ok=True)
    leakage_chart(results)
    physionet_channel_chart(results)
    mendeley_chart(results)
    capability_chart(results)
    print(f"图表已保存：{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
