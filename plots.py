"""Графики по results/*.csv -> figures/*.png"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SETS = ["E", "F", "M", "P"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.linewidth": 1,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#FBFBF8", "axes.facecolor": "#FBFBF8", "savefig.dpi": 200,
})


def by_n_and_set(summ):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), sharey="row", sharex=True)
    for j, s in enumerate(SETS):
        d = summ[summ.set == s].sort_values("n")
        ax = axes[0, j]
        ax.errorbar(d.n, d.mean_gap, yerr=[d.mean_gap - d.best_gap, d.max_gap - d.mean_gap],
                    fmt="o", ms=5, color=BLUE, ecolor=BLUE, elinewidth=1.2, capsize=0,
                    markeredgecolor="white", markeredgewidth=1)
        ax.set_title(f"Набор {s}  ({len(d)} задач)", fontsize=15, color=INK, loc="left")
        ax2 = axes[1, j]
        ax2.plot(d.n, d.mean_time, "o", ms=5, color=BLUE, markeredgecolor="white", markeredgewidth=1)
        ax2.set_xlabel("n (клиентов)")
    axes[0, 0].set_ylabel("Отклонение, %")
    axes[1, 0].set_ylabel("Время запуска, с")
    axes[0, 0].set_ylim(bottom=-0.2)
    fig.text(0.01, 0.005, "Точка — среднее по 10 запускам; отрезок — от лучшего до худшего запуска.",
             color=INK2, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/gap_time_vs_n.png")
    plt.close(fig)


def by_set_bars(by):
    by = by.loc[SETS]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = range(len(by))
    w = 0.38
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], by.cw_gap, w, color=ORANGE, label="Кларк–Райт (начальное)")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], by.mean_gap, w, color=BLUE, label="Отжиг (среднее по запускам)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.15, f"{r.get_height():.1f}",
                    ha="center", va="bottom", fontsize=13, color=INK)
    ax.set_xticks(list(x), [f"{s}" for s in by.index])
    ax.set_ylabel("Среднее отклонение, %")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right", fontsize=13)
    ax.set_ylim(0, max(by.cw_gap.max(), by.mean_gap.max()) * 1.25)
    fig.tight_layout()
    fig.savefig("figures/gap_by_set.png")
    plt.close(fig)


def convergence(conv, name, opt):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(conv["iter"] / 1e6, conv["current"], color=ORANGE, lw=1.5, label="текущее решение")
    ax.plot(conv["iter"] / 1e6, conv["best"], color=BLUE, lw=2, label="лучшее допустимое")
    ax.axhline(opt, color=INK2, lw=1, ls="--")
    ax.text(conv["iter"].max() / 1e6, opt, f"оптимум {opt:.0f}", ha="right", va="bottom", fontsize=12, color=INK2)
    ax.set_xlabel("Итерации, млн")
    ax.set_ylabel("Стоимость")
    ax.set_title(name, loc="left", fontsize=15)
    lo = opt * 0.97
    ax.set_ylim(lo, conv["best"].replace(float("inf"), float("nan")).max() * 1.15)
    ax.legend(frameon=False, fontsize=12, loc="upper left", handlelength=1.5)
    fig.tight_layout()
    fig.savefig("figures/convergence.png")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    summ = pd.read_csv("results/summary.csv")
    by = pd.read_csv("results/by_set.csv", index_col=0)
    by_n_and_set(summ)
    by_set_bars(by)
    conv = pd.read_csv("results/convergence.csv")
    opt = summ.loc[summ.instance == "M-n200-k17", "opt"].iloc[0]
    convergence(conv, "M-n200-k17: сходимость отжига", opt)

    t = pd.read_csv("results/tuning.csv")
    g = t.groupby(["alpha", "epoch_factor", "p0"], as_index=False).agg(gap=("gap", "mean"), time=("time", "mean"))
    g.sort_values("gap").to_csv("results/tuning_summary.csv", index=False, float_format="%.3f")
    print("figures saved")
