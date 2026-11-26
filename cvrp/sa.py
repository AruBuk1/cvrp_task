"""Имитация отжига для CVRP.

Решение — ровно k маршрутов (часть может быть пустой). Окрестность — 4 хода:
  0  перенос клиента (relocate): u ставится до/после соседа v, в т.ч. в другой маршрут
  1  обмен клиентов (swap) u <-> v
  2  2-opt: удаление рёбер (u,u+), (v,v+) и добавление (u,v), (u+,v+);
     внутри маршрута — разворот отрезка, между маршрутами — перекрещивание с разворотом
  3  2-opt* (перекрещивание рёбер двух маршрутов): (u-,u),(v,v+) -> (v,u),(u-,v+)
v выбирается из G ближайших к u клиентов (гранулярная окрестность).

Нарушение грузоподъёмности разрешено, но штрафуется:
    f = стоимость + lam * суммарный_перегруз,
lam адаптируется, чтобы примерно половина посещённых решений была допустимой.
Ответом считается лучшее ДОПУСТИМОЕ решение ровно из k непустых маршрутов
(пустой маршрут может только заполниться, опустошать маршрут запрещено).
"""
import math
import random
import time
from dataclasses import dataclass

from .construct import clarke_wright, to_k_routes
from .instance import route_cost


@dataclass
class SAParams:
    p0: float = 0.05         # вероятность принять "среднее" ухудшение в начале -> T0
    alpha: float = 0.99      # коэффициент охлаждения за эпоху: T <- alpha*T
    epoch_factor: int = 50   # длина эпохи L = epoch_factor * n итераций
    t_end: float = 1e-3      # остановка, когда T < t_end * T0 ...
    patience: int = 30       # ... или система "замёрзла": доля принятых ходов < frozen
    frozen: float = 0.01     #     и лучшее решение не улучшалось patience эпох подряд
    granular: int = 20       # размер списка ближайших соседей
    time_limit: float = 300  # страховочный лимит, с
    move_weights: tuple = (0.35, 0.25, 0.2, 0.2)


class Solution:
    def __init__(self, inst, routes):
        self.inst = inst
        self.routes = [list(r) for r in routes]
        n = inst.n
        self.rid = [0] * n
        self.pos = [0] * n
        self.load = [0] * len(self.routes)
        self.cum = [None] * len(self.routes)
        for r in range(len(self.routes)):
            self.rebuild(r)
        self.cost = route_cost(inst, self.routes)
        self.over = sum(max(0, L - inst.Q) for L in self.load)

    def rebuild(self, r):
        q = self.inst.q
        rid, pos = self.rid, self.pos
        R = self.routes[r]
        acc = 0
        cum = []
        for i, c in enumerate(R):
            rid[c] = r
            pos[c] = i
            acc += q[c]
            cum.append(acc)
        self.cum[r] = cum
        self.load[r] = acc


def solve(inst, params=SAParams(), seed=0, log_every=0):
    rng = random.Random(seed)
    rnd = rng.random
    t_start = time.perf_counter()

    D, q, Q, n = inst.D, inst.q, inst.Q, inst.n
    G = min(params.granular, n - 2)
    neigh = [[]] + [sorted((c for c in range(1, n) if c != u), key=lambda c: D[u][c])[:G]
                    for u in range(1, n)]

    init_routes = to_k_routes(inst, clarke_wright(inst))
    S = Solution(inst, init_routes)
    routes, rid, pos, load, cum = S.routes, S.rid, S.pos, S.load, S.cum
    K = len(routes)

    cw_cost = route_cost(inst, [r for r in clarke_wright(inst)])
    ok0 = S.over == 0 and all(routes)
    best_cost = S.cost if ok0 else math.inf
    best_routes = [list(r) for r in routes] if ok0 else None

    # штраф за единицу перегруза: стартовое значение ~ средняя длина ребра / средний спрос
    avg_d = sum(D[0][i] for i in range(1, n)) / (n - 1)
    avg_q = max(1, sum(q) / (n - 1))
    lam = avg_d / avg_q
    w = params.move_weights
    c1, c2, c3 = w[0], w[0] + w[1], w[0] + w[1] + w[2]

    def over_of(L):
        return L - Q if L > Q else 0

    # ---------- оценка хода: возвращает (dcost, dover, данные для применения) ----------
    def propose():
        u = int(rnd() * (n - 1)) + 1
        v = neigh[u][int(rnd() * G)]
        ru, rv = rid[u], rid[v]
        Ru, Rv = routes[ru], routes[rv]
        iu, iv = pos[u], pos[v]
        pu = Ru[iu - 1] if iu > 0 else 0
        nu = Ru[iu + 1] if iu + 1 < len(Ru) else 0
        pv = Rv[iv - 1] if iv > 0 else 0
        nv = Rv[iv + 1] if iv + 1 < len(Rv) else 0
        x = rnd()

        if x < c1:                                   # relocate
            if rnd() < 0.05:                         # иногда — в пустой маршрут
                empty = [r for r in range(K) if not routes[r]]
                if not empty:
                    return None
                rt, idx, a, b = empty[0], 0, 0, 0
            elif rnd() < 0.5:                        # после v
                rt, idx, a, b = rv, iv + 1, v, nv
            else:                                    # перед v
                rt, idx, a, b = rv, iv, pv, v
            if a == u or b == u:
                return None
            if rt != ru and len(Ru) == 1:           # маршрут не должен опустеть (k фиксировано)
                return None
            dc = D[pu][nu] - D[pu][u] - D[u][nu] + D[a][u] + D[u][b] - D[a][b]
            if rt == ru:
                return dc, 0, (0, u, rt, idx)
            Lu, Lt = load[ru], load[rt]
            do = (over_of(Lu - q[u]) + over_of(Lt + q[u])) - (over_of(Lu) + over_of(Lt))
            return dc, do, (0, u, rt, idx)

        if x < c2:                                   # swap
            if nu == v and ru == rv:
                dc = D[pu][v] + D[u][nv] - D[pu][u] - D[v][nv]
            elif nv == u and ru == rv:
                dc = D[pv][u] + D[v][nu] - D[pv][v] - D[u][nu]
            else:
                dc = (D[pu][v] + D[v][nu] - D[pu][u] - D[u][nu]
                      + D[pv][u] + D[u][nv] - D[pv][v] - D[v][nv])
            if ru == rv:
                return dc, 0, (1, u, v)
            Lu, Lv = load[ru], load[rv]
            d = q[v] - q[u]
            do = (over_of(Lu + d) + over_of(Lv - d)) - (over_of(Lu) + over_of(Lv))
            return dc, do, (1, u, v)

        if x < c3:                                   # 2-opt: (u,nu),(v,nv) -> (u,v),(nu,nv)
            if ru == rv:
                if iu > iv:
                    u, v, iu, iv, nu, nv = v, u, iv, iu, nv, nu
                if iv == iu + 1:
                    return None
                dc = D[u][v] + D[nu][nv] - D[u][nu] - D[v][nv]
                return dc, 0, (2, ru, iu, iv)
            if iu + 1 == len(Ru) and iv + 1 == len(Rv):   # второй маршрут стал бы пустым
                return None
            dc = D[u][v] + D[nu][nv] - D[u][nu] - D[v][nv]
            cu, cv = cum[ru][iu], cum[rv][iv]
            Lu, Lv = load[ru], load[rv]
            nlu, nlv = cu + cv, (Lu - cu) + (Lv - cv)
            do = (over_of(nlu) + over_of(nlv)) - (over_of(Lu) + over_of(Lv))
            return dc, do, (3, ru, rv, iu, iv, nlu, nlv)

        # 2-opt*: (pu,u),(v,nv) -> (v,u),(pu,nv); маршруты: Ru[:iu]+Rv[iv+1:], Rv[:iv+1]+Ru[iu:]
        if ru == rv or (iu == 0 and iv + 1 == len(Rv)):  # первый маршрут стал бы пустым
            return None
        dc = D[v][u] + D[pu][nv] - D[pu][u] - D[v][nv]
        cu = cum[ru][iu] - q[u]
        cv = cum[rv][iv]
        Lu, Lv = load[ru], load[rv]
        nlu, nlv = cu + (Lv - cv), cv + (Lu - cu)
        do = (over_of(nlu) + over_of(nlv)) - (over_of(Lu) + over_of(Lv))
        return dc, do, (4, ru, rv, iu, iv)

    def apply(mv):
        t = mv[0]
        if t == 0:
            _, u, rt, idx = mv
            ru = rid[u]
            iu = pos[u]
            routes[ru].pop(iu)
            if rt == ru and iu < idx:
                idx -= 1
            routes[rt].insert(idx, u)
            S.rebuild(ru)
            if rt != ru:
                S.rebuild(rt)
        elif t == 1:
            _, u, v = mv
            ru, rv = rid[u], rid[v]
            iu, iv = pos[u], pos[v]
            routes[ru][iu] = v
            routes[rv][iv] = u
            S.rebuild(ru)
            if rv != ru:
                S.rebuild(rv)
        elif t == 2:
            _, r, i, j = mv
            R = routes[r]
            R[i + 1:j + 1] = R[i + 1:j + 1][::-1]
            S.rebuild(r)
        elif t == 3:
            _, ru, rv, iu, iv, _, _ = mv
            Ru, Rv = routes[ru], routes[rv]
            new_u = Ru[:iu + 1] + Rv[:iv + 1][::-1]
            new_v = Ru[iu + 1:][::-1] + Rv[iv + 1:]
            routes[ru], routes[rv] = new_u, new_v
            S.rebuild(ru)
            S.rebuild(rv)
        else:
            _, ru, rv, iu, iv = mv
            Ru, Rv = routes[ru], routes[rv]
            new_u = Ru[:iu] + Rv[iv + 1:]
            new_v = Rv[:iv + 1] + Ru[iu:]
            routes[ru], routes[rv] = new_u, new_v
            S.rebuild(ru)
            S.rebuild(rv)

    # ---------- начальная температура: по выборке ухудшающих ходов ----------
    ups = []
    for _ in range(2000):
        m = propose()
        if m:
            df = m[0] + lam * m[1]
            if df > 0:
                ups.append(df)
    T0 = (sum(ups) / len(ups)) / -math.log(params.p0) if ups else 1.0
    T = T0
    T_min = params.t_end * T0
    L = params.epoch_factor * n

    cur_cost, cur_over = S.cost, S.over
    n_empty = sum(1 for r in routes if not r)
    epoch = 0
    stale = 0
    iters = 0
    history = []
    exp = math.exp
    while T > T_min and stale < params.patience:
        improved = False
        feas_cnt = 0
        acc_cnt = 0
        for _ in range(L):
            m = propose()
            iters += 1
            if m is None:
                continue
            dc, do, mv = m
            df = dc + lam * do
            if df <= 0 or rnd() < exp(-df / T):
                if mv[0] == 0 and not routes[mv[2]]:
                    n_empty -= 1
                apply(mv)
                acc_cnt += 1
                cur_cost += dc
                cur_over += do
                if cur_over == 0 and n_empty == 0 and cur_cost < best_cost:
                    best_cost = cur_cost
                    best_routes = [list(r) for r in routes]
                    improved = True
            if cur_over == 0:
                feas_cnt += 1
        # адаптация штрафа
        frac = feas_cnt / L
        if frac < 0.4:
            lam *= 1.3
        elif frac > 0.6:
            lam = max(lam / 1.3, 1e-3)
        epoch += 1
        stale = 0 if (improved or acc_cnt / L >= params.frozen) else stale + 1
        T *= params.alpha
        history.append((iters, best_cost, cur_cost, T))
        if log_every and epoch % log_every == 0:
            print(f"ep {epoch:4d} T={T:9.3f} cur={cur_cost} over={cur_over} best={best_cost} lam={lam:.3f}")
        if time.perf_counter() - t_start > params.time_limit:
            break

    elapsed = time.perf_counter() - t_start
    if best_routes is not None:
        best_routes = [r for r in best_routes if r]
    return {
        "cost": best_cost,
        "routes": best_routes,
        "time": elapsed,
        "iters": iters,
        "epochs": epoch,
        "cw_cost": cw_cost,
        "T0": T0,
        "history": history,
        "stop": "T_min" if T <= T_min else ("patience" if stale >= params.patience else "time"),
    }
