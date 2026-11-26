"""Чтение задач CVRPLIB (формат TSPLIB) и оптимальных решений."""
import math
import os
import re
from dataclasses import dataclass, field


@dataclass
class Instance:
    name: str
    n: int                      # число вершин вместе со складом (DIMENSION)
    Q: int                      # грузоподъёмность
    k: int                      # число машин (из имени файла ...-kK)
    q: list                     # спрос, q[0] = 0 (склад)
    D: list                     # матрица расстояний n x n (целые, TSPLIB)
    coords: list = field(default_factory=list)
    opt: float | None = None    # оптимальная стоимость из .sol
    opt_routes: list | None = None

    @property
    def group(self):
        return self.name.split("-")[0]


def _tokens(lines, i, count):
    vals = []
    while len(vals) < count:
        vals += lines[i].split()
        i += 1
    return vals, i


def read_vrp(path):
    lines = open(path).read().replace("\r", "").split("\n")
    meta, coords, demand, weights = {}, [], [], None
    n = 0
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s == "EOF":
            i += 1
            continue
        if s.startswith("NODE_COORD_SECTION"):
            for j in range(n):
                a = lines[i + 1 + j].split()
                coords.append((float(a[1]), float(a[2])))
            i += n + 1
        elif s.startswith("DEMAND_SECTION"):
            for j in range(n):
                a = lines[i + 1 + j].split()
                demand.append(int(a[1]))
            i += n + 1
        elif s.startswith("EDGE_WEIGHT_SECTION"):
            fmt = meta.get("EDGE_WEIGHT_FORMAT", "LOWER_ROW")
            assert fmt == "LOWER_ROW", fmt
            vals, i = _tokens(lines, i + 1, n * (n - 1) // 2)
            weights = [float(v) for v in vals]
        elif s.startswith("DEPOT_SECTION"):
            depot = int(lines[i + 1].split()[0])
            assert depot == 1, "склад должен быть вершиной 1"
            i += 2
        elif ":" in s:
            key, val = s.split(":", 1)
            meta[key.strip()] = val.strip()
            if key.strip() == "DIMENSION":
                n = int(val)
            i += 1
        else:
            i += 1

    name = meta.get("NAME", os.path.basename(path)[:-4])
    Q = int(meta["CAPACITY"])
    k = int(re.search(r"-k(\d+)", name).group(1))
    D = [[0] * n for _ in range(n)]
    if weights is not None:                       # EXPLICIT, LOWER_ROW
        t = 0
        for a in range(1, n):
            for b in range(a):
                D[a][b] = D[b][a] = int(round(weights[t]))
                t += 1
    else:                                          # EUC_2D: округление до целого
        for a in range(n):
            xa, ya = coords[a]
            for b in range(a + 1, n):
                xb, yb = coords[b]
                D[a][b] = D[b][a] = int(math.floor(math.hypot(xa - xb, ya - yb) + 0.5))
    inst = Instance(name=name, n=n, Q=Q, k=k, q=demand, D=D, coords=coords)

    sol = path[:-4] + ".sol"
    if os.path.exists(sol):
        inst.opt, inst.opt_routes = read_sol(sol)
    return inst


def read_sol(path):
    routes, cost = [], None
    for line in open(path):
        if line.startswith("Route"):
            routes.append([int(x) for x in line.split(":")[1].split()])
        elif line.lower().startswith("cost"):
            cost = float(line.split()[1])
    return cost, routes


def route_cost(inst, routes):
    D = inst.D
    total = 0
    for r in routes:
        prev = 0
        for c in r:
            total += D[prev][c]
            prev = c
        total += D[prev][0]
    return total


def check_solution(inst, routes):
    """Возвращает (стоимость, список нарушений). Пустой список = решение допустимо."""
    errors = []
    seen = [c for r in routes for c in r]
    if sorted(seen) != list(range(1, inst.n)):
        errors.append("каждый клиент должен быть посещён ровно один раз")
    for i, r in enumerate(routes):
        load = sum(inst.q[c] for c in r)
        if load > inst.Q:
            errors.append(f"маршрут {i}: загрузка {load} > {inst.Q}")
    used = sum(1 for r in routes if r)
    if used != inst.k:
        errors.append(f"маршрутов {used} != k={inst.k}")
    return route_cost(inst, routes), errors


def load_all(data_dir, groups=("E", "F", "M", "P")):
    out = []
    for g in groups:
        d = os.path.join(data_dir, g)
        for f in sorted(os.listdir(d)):
            if f.endswith(".vrp"):
                out.append(read_vrp(os.path.join(d, f)))
    out.sort(key=lambda x: (x.group, x.n, x.name))
    return out
