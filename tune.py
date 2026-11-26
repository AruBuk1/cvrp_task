"""Подбор параметров отжига по сетке на нескольких задачах разного размера и типа.

python tune.py --seeds 3 --jobs 2
Результат: results/tuning.csv
"""
import argparse
import csv
import itertools
import os
from multiprocessing import Pool

from cvrp.instance import read_vrp
from cvrp.sa import SAParams, solve

TUNE_SET = ["E-n51-k5", "P-n55-k15", "E-n76-k10", "M-n101-k10", "E-n101-k14", "M-n151-k12"]
GRID = {
    "alpha": [0.95, 0.98, 0.99, 0.995],
    "epoch_factor": [10, 50],
    "p0": [0.05, 0.3],
}


def job(args):
    name, alpha, ef, p0, seed = args
    inst = read_vrp(os.path.join("data", name.split("-")[0], name + ".vrp"))
    r = solve(inst, SAParams(alpha=alpha, epoch_factor=ef, p0=p0), seed=seed)
    return {
        "instance": name, "alpha": alpha, "epoch_factor": ef, "p0": p0, "seed": seed,
        "cost": r["cost"], "opt": inst.opt,
        "gap": 100 * (r["cost"] - inst.opt) / inst.opt,
        "time": round(r["time"], 3), "iters": r["iters"], "stop": r["stop"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    tasks = [(nm, al, ef, p0, s)
             for al, ef, p0 in itertools.product(GRID["alpha"], GRID["epoch_factor"], GRID["p0"])
             for nm in TUNE_SET for s in range(a.seeds)]
    os.makedirs("results", exist_ok=True)
    with Pool(a.jobs) as pool:
        rows = pool.map(job, tasks, chunksize=1)
    with open("results/tuning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("saved results/tuning.csv,", len(rows), "runs")
