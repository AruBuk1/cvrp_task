"""Начальное решение: параллельный алгоритм сбережений Кларка–Райта."""


def clarke_wright(inst):
    n, D, q, Q = inst.n, inst.D, inst.q, inst.Q
    # сначала каждый клиент обслуживается отдельным маршрутом (0, i, 0)
    route_of = list(range(n))                 # id маршрута клиента
    routes = {i: [i] for i in range(1, n)}
    load = {i: q[i] for i in range(1, n)}

    # сбережения s_ij = d(0,i) + d(0,j) - d(i,j), по убыванию
    savings = [(D[0][i] + D[0][j] - D[i][j], i, j)
               for i in range(1, n) for j in range(i + 1, n)]
    savings.sort(reverse=True)

    for s, i, j in savings:
        if s <= 0:
            break
        ri, rj = route_of[i], route_of[j]
        if ri == rj or load[ri] + load[rj] > Q:
            continue
        a, b = routes[ri], routes[rj]
        # i и j должны быть концевыми клиентами своих маршрутов (смежны со складом)
        if a[-1] == i and b[0] == j:
            new = a + b
        elif a[0] == i and b[-1] == j:
            new = b + a
        elif a[-1] == i and b[-1] == j:
            new = a + b[::-1]
        elif a[0] == i and b[0] == j:
            new = a[::-1] + b
        else:
            continue
        routes[ri] = new
        load[ri] += load[rj]
        for c in b:
            route_of[c] = ri
        del routes[rj], load[rj]
    return list(routes.values())


def to_k_routes(inst, routes):
    """Приводит решение ровно к k маршрутам (Number of Routes: Fixed).

    Если маршрутов больше k, наименее загруженные маршруты расформировываются,
    а их клиенты вставляются в самые дешёвые позиции оставшихся (перегруз
    допускается — его потом устраняет отжиг через штраф).
    Если маршрутов меньше k, добавляются пустые маршруты.
    """
    D, q = inst.D, inst.q
    routes = [list(r) for r in routes if r]
    while len(routes) > inst.k:
        routes.sort(key=lambda r: sum(q[c] for c in r))
        orphan = routes.pop(0)
        for c in sorted(orphan, key=lambda c: -q[c]):
            loads = [sum(q[x] for x in r) for r in routes]
            best = None
            for ri, r in enumerate(routes):
                over = max(0, loads[ri] + q[c] - inst.Q)
                for p in range(len(r) + 1):
                    a = r[p - 1] if p > 0 else 0
                    b = r[p] if p < len(r) else 0
                    cost = D[a][c] + D[c][b] - D[a][b] + 1000 * over
                    if best is None or cost < best[0]:
                        best = (cost, ri, p)
            _, ri, p = best
            routes[ri].insert(p, c)
    while len(routes) < inst.k:
        routes.append([])
    return routes
