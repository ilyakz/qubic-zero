#!/usr/bin/env python
"""Рисует графики для README из runs/ и results/findings.json в docs/img/.

    python tools/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "img")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C1, C2, C3, C4, MAIN, CONTROL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#898781"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
})


def metrics(run):
    path = os.path.join(ROOT, "runs", run, "metrics.jsonl")
    if not os.path.exists(path):  # в репозитории лежат только копии метрик
        path = os.path.join(ROOT, "results", "metrics", run + ".jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def series(records, fn):
    pairs = [(m["iter"], fn(m)) for m in records]
    pairs = [(x, y) for x, y in pairs if y is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def win_rate(opponent):
    def fn(m):
        e = m.get("eval", {}).get(opponent)
        return None if not e else 100 * (e["win"] + e["draw"] / 2) / e["games"]
    return fn


def second_rate(opponent):
    def fn(m):
        e = m.get("eval", {}).get(opponent)
        if not e:
            return None
        w, d, l = e["as_second"]
        return 100 * (w + d / 2) / max(w + d + l, 1)
    return fn


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("docs/img/" + name)


def figure_bug():
    """Главная ошибка 2019 года: ошибка оценки падает к нулю, а прогноз исхода становится хуже монетки."""
    fixed = [m for m in metrics("main_v1") if m["iter"] <= 10]
    broken = metrics("control_2019")
    if not fixed or not broken:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    get_loss = lambda m: m.get("train", {}).get("value_loss")
    get_acc = lambda m: 100 * m["value_check"]["accuracy"] if "value_check" in m else None
    for ax, fn, title in ((axes[0], get_loss, "Value loss: the broken run looks great"), (axes[1], get_acc, "Predicting who wins: it is actually worse than a coin")):
        ax.plot(*series(fixed, fn), color=MAIN, label="labels fixed")
        ax.plot(*series(broken, fn), color=CONTROL, label="labels as in 2019")
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.legend(loc="center right")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 100)
    axes[1].axhline(50, color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    axes[1].set_ylabel("%")
    fig.tight_layout()
    save(fig, "bug_2019.png")


def figure_tree():
    with open(os.path.join(ROOT, "results", "findings.json")) as f:
        data = json.load(f)["tree_vcf"]
    labels = ["no check", "budget 40", "budget 150", "budget 400"]
    values = [r["rate"] for r in data["vs_tsp"]] + [data["budget_400_100_games_vs_tsp"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.bar(labels, values, color=MAIN, width=0.42)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 108)
    ax.grid(axis="x", visible=False)
    ax.set_title("Same weights, same 100 simulations, wins against tsp")
    fig.tight_layout()
    save(fig, "tree_check.png")


def figure_networks():
    arms = [("arch_2019", "2019 network, 8M weights", C2), ("arch_resnet", "convolutions, 3.1M", C1), ("arch_lines", "convolutions + line features, 3.1M", C3), ("arch_linenet", "line network, 0.4M", MAIN)]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for run, label, color in arms:
        records = [m for m in metrics(run) if m["iter"] <= 40]
        if records:
            ax.plot(*series(records, win_rate("tsp")), color=color, label=label)
    ax.set_ylim(0, 103)
    ax.set_ylabel("% wins")
    ax.set_xlabel("iteration")
    ax.set_title("Trained from scratch in equal conditions, wins against tsp")
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    save(fig, "networks.png")


def figure_first_move(run="main"):
    records = [m for m in metrics(run) if m.get("first_move")]
    if not records:
        return
    policy = np.array(records[-1]["first_move"]["policy"])
    if len(policy) != 64:
        return
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.6))
    top = policy.max()
    for z, ax in enumerate(axes):
        layer = policy[z * 16:(z + 1) * 16].reshape(4, 4)
        ax.imshow(layer, cmap="Blues", vmin=0, vmax=top)
        ax.set_title(f"layer {z + 1}", fontsize=10.5, fontweight="normal")
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for y in range(4):
            for x in range(4):
                if layer[y, x] >= 0.01:
                    ax.text(x, y, f"{layer[y, x] * 100:.0f}", ha="center", va="center", fontsize=9, color="white" if layer[y, x] > top * 0.55 else INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    share = policy[[0, 3, 12, 15, 21, 22, 25, 26, 37, 38, 41, 42, 48, 51, 60, 63]].sum()
    fig.suptitle(f"Where the network wants to move first, %: {share * 100:.0f}% on the 16 cells that lie on 7 lines", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "first_move.png")


def figure_progress(run="main"):
    records = metrics(run)
    if not records:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, color, label in (("pure", C1, "search without a network"), ("tsp", C2, "tsp"), ("hsp", C3, "hsp"), ("osp", C4, "osp")):
        axes[0].plot(*series(records, win_rate(name)), color=color, label=label)
    axes[0].axhline(35, color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    axes[0].text(records[-1]["iter"], 37, "2019: 35% vs hsp", ha="right", color=INK2, fontsize=9.5)
    axes[0].set_title("Wins against the opponents, %")
    axes[0].legend(loc="lower right", fontsize=9.5)
    for name, color, label in (("pure", C1, "search without a network"), ("tsp", C2, "tsp")):
        axes[1].plot(*series(records, second_rate(name)), color=color, label=label)
    axes[1].set_title("Playing second (the hard side), %")
    axes[1].legend(loc="lower right", fontsize=9.5)
    for ax in axes:
        ax.set_ylim(0, 103)
        ax.set_xlabel("iteration")
    fig.tight_layout()
    save(fig, "progress.png")


if __name__ == "__main__":
    figure_bug()
    figure_tree()
    figure_networks()
    figure_first_move()
    figure_progress()
