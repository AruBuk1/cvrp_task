"""Прогон отжига на всех задачах наборов E, F, M, P.

python run_all.py --seeds 10 --jobs 2
Результаты:
  results/runs.csv        — каждый запуск (задача × seed)
  results/summary.csv     — по задачам: лучшее/среднее отклонение, время
  results/by_set.csv      — средние по наборам
  results/convergence.csv — траектория одного запуска (для графика сходимости)
  solutions/<name>.sol    — лучшее найденное решение в формате CVRPLIB
"""
import argparse
import csv
import os
from multiprocessing import Pool

import pandas as pd

from cvrp.instance import check_solution, load_all, read_vrp
from cvrp.sa import SAParams, solve

CONV_INSTANCE = "M-n200-k17"


def job(args):
    path, seed = args
    inst = read_vrp(path)
    r = solve(inst, SAParams(), seed=seed)
    cost, errors = check_solution(inst, r["routes"])
    assert not errors and cost == r["cost"], (inst.name, errors, cost, r["cost"])
    row = {
        "instance": inst.name, "set": inst.group, "n": inst.n - 1, "k": inst.k, "seed": seed,
        "opt": inst.opt, "cw_cost": r["cw_cost"], "cost": r["cost"],
        "gap": 100 * (r["cost"] - inst.opt) / inst.opt,
        "cw_gap": 100 * (r["cw_cost"] - inst.opt) / inst.opt,
        "routes_used": len(r["routes"]),
        "time": r["time"], "iters": r["iters"], "epochs": r["epochs"], "stop": r["stop"],
    }
    hist = r["history"] if (inst.name == CONV_INSTANCE and seed == 0) else None
    return row, r["routes"], hist


def write_sol(path, routes, cost):
    with open(path, "w") as f:
        for i, r in enumerate(routes, 1):
            f.write(f"Route #{i}: {' '.join(map(str, r))}\n")
        f.write(f"Cost {cost}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()

    insts = load_all("data")
    paths = {i.name: os.path.join("data", i.group, i.name + ".vrp") for i in insts}
    # крупные задачи первыми — лучше загрузка процессов
    order = sorted(insts, key=lambda i: -i.n)
    tasks = [(paths[i.name], s) for i in order for s in range(a.seeds)]

    os.makedirs("results", exist_ok=True)
    os.makedirs("solutions", exist_ok=True)
    rows, best = [], {}
    with Pool(a.jobs) as pool:
        for row, routes, hist in pool.imap_unordered(job, tasks):
            rows.append(row)
            nm = row["instance"]
            if nm not in best or row["cost"] < best[nm][0]:
                best[nm] = (row["cost"], routes)
            if hist:
                with open("results/convergence.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["iter", "best", "current", "T"])
                    w.writerows(hist)
            print(f"{nm:12s} seed={row['seed']} cost={row['cost']} gap={row['gap']:.2f}% t={row['time']:.1f}s")

    for nm, (cost, routes) in best.items():
        write_sol(os.path.join("solutions", nm + ".sol"), routes, cost)

    df = pd.DataFrame(rows).sort_values(["set", "n", "instance", "seed"])
    df.to_csv("results/runs.csv", index=False)
    summ = df.groupby(["set", "instance", "n", "k", "opt"], as_index=False).agg(
        cw_gap=("cw_gap", "first"),
        best_cost=("cost", "min"), mean_cost=("cost", "mean"),
        best_gap=("gap", "min"), mean_gap=("gap", "mean"), max_gap=("gap", "max"),
        std_gap=("gap", "std"), mean_time=("time", "mean"), mean_iters=("iters", "mean"),
    ).sort_values(["set", "n", "instance"])
    summ.to_csv("results/summary.csv", index=False, float_format="%.3f")
    by_set = summ.groupby("set").agg(
        instances=("instance", "count"), n_min=("n", "min"), n_max=("n", "max"),
        cw_gap=("cw_gap", "mean"), best_gap=("best_gap", "mean"), mean_gap=("mean_gap", "mean"),
        worst_gap=("max_gap", "max"), optimal_found=("best_gap", lambda g: int((g < 1e-9).sum())),
        mean_time=("mean_time", "mean"),
    )
    tot = summ.agg({"instance": "count", "n": ["min", "max"]})
    by_set.loc["ALL"] = [len(summ), summ.n.min(), summ.n.max(), summ.cw_gap.mean(), summ.best_gap.mean(),
                         summ.mean_gap.mean(), summ.max_gap.max(), int((summ.best_gap < 1e-9).sum()),
                         summ.mean_time.mean()]
    by_set.to_csv("results/by_set.csv", float_format="%.3f")
    print(by_set.round(2).to_string())
