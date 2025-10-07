from collections import deque, defaultdict
from math import inf
import timeit

from numpy.ma.core import empty
import numpy as np
import math

def parse_input(data):
    """Parse input tree and budget."""
    lines = [l for l in data.strip().splitlines() if l.strip()]
    n, B = map(int, lines[0].split())
    node_info, idx = {}, 1
    for _ in range(n):
        parts = lines[idx].split();
        idx += 1
        u = int(parts[0])
        if parts[1] == 'f':
            node_info[u] = ('f', 0)
        else:
            node_info[u] = ('c', int(parts[2]))
    m = int(lines[idx]);
    idx += 1
    edges = []
    for _ in range(m):
        u, v = map(int, lines[idx].split());
        idx += 1
        edges.append((u, v))
    return n, B, node_info, edges


def build_adj(n, edges):
    adj = {i: [] for i in range(1, n + 1)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def make_clusters(n, edges, node_info):
    """
    Partition the tree into clusters: connected components of customers,
    plus any facilities adjacent to them.
    Returns list of (nodeset, cluster_edges).
    """
    # Build adjacency among customers and map customers to adjacent facilities
    cust_adj = defaultdict(list)
    cust_to_facs = defaultdict(set)
    for u, v in edges:
        tu, tv = node_info[u][0], node_info[v][0]
        if tu == 'c' and tv == 'c':
            cust_adj[u].append(v)
            cust_adj[v].append(u)
        elif tu == 'c' and tv == 'f':
            cust_to_facs[u].add(v)
        elif tu == 'f' and tv == 'c':
            cust_to_facs[v].add(u)

    visited = set()
    clusters = []
    for u, (t, _) in node_info.items():
        if t != 'c' or u in visited:
            continue
        # DFS to collect one customer-component
        stack = [u]
        visited.add(u)
        custs = {u}
        while stack:
            v = stack.pop()
            for w in cust_adj[v]:
                if w not in visited:
                    visited.add(w)
                    custs.add(w)
                    stack.append(w)
        # Gather facilities adjacent to any customer in this component
        facs = set()
        for c in custs:
            facs |= cust_to_facs[c]
        nodeset = custs | facs
        # Extract edges inside the cluster
        adj = build_adj(n, edges)
        cl_edges = []
        for v in nodeset:
            for w in adj[v]:
                if w in nodeset and v < w:
                    cl_edges.append((v, w))
        clusters.append((nodeset, cl_edges))
    return clusters


def build_candidate_tree(adj, node_info, x):
    parent, depth = {x: None}, {x: 0}
    dq = deque([x])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                depth[v] = depth[u] + 1
                dq.append(v)

    cand = set()

    children = {u: [] for u in adj}
    for v in adj:
        p = parent.get(v)
        if p is not None:
            children[p].append(v)

    facilities = [u for u, (t, _) in node_info.items() if t == 'f']
    is_fac = {u: (node_info[u][0] == 'f') for u in adj}

    has_fac = {u: False for u in adj}
    for u in sorted(adj, key=lambda t: depth[t], reverse=True):
        has_fac[u] = is_fac[u] or any(has_fac[c] for c in children[u])

    for u in adj:
        p = parent.get(u)
        if p is None:
            continue
        fac_child_subtrees = 0
        for c in children[u]:
            if has_fac[c]:
                fac_child_subtrees += 1
                if fac_child_subtrees >= 2:
                    break
        if is_fac[u] or fac_child_subtrees >= 2:
            cand.add((p, u))

    cand = sorted(cand, key=lambda e: depth[e[0]])

    def dfs(u, rep, visited):
        for v in adj[u]:
            if v in visited:
                continue
            if (u, v) in cand or (v, u) in cand:
                e_std = (u, v) if (u, v) in cand else (v, u)
                rep.append(e_std)
                visited.add(v)
            else:
                visited.add(v)
                rep = dfs(v, rep, visited)
        return rep

    rep_x = dfs(x, [], {x})

    # print(rep_x)

    def edge_facility(edge):
        # print(edge)
        (u, v) = edge
        if node_info[u][0] == 'f' or node_info[v][0] == 'f':
            return True
        else:
            return False

    def edge_id(edge):
        u, v = edge
        return idx_map.get((u, v), idx_map.get((v, u)))

    def edge_id_if_facility(edge, idx_map, node_info):

        u, v = edge
        if node_info[u][0] == 'f' or node_info[v][0] == 'f':
            return None
        if (u, v) in idx_map:
            return idx_map[(u, v)]
        elif (v, u) in idx_map:
            return idx_map[(v, u)]
        else:
            return None

    def enclosed(cut, x):
        vis = {x}
        dq = deque([x])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if (u, v) in cut or (v, u) in cut:
                    continue
                if v not in vis:
                    vis.add(v)
                    dq.append(v)
        return vis

    cost, profit, Xset, Min_b, parent_state = {}, {}, {}, {}, {}
    children = defaultdict(list)
    # root state

    cut0 = set(rep_x)
    X0 = enclosed(cut0, x)
    cost[0] = len(cut0)
    profit[0] = sum(w for u, (t, w) in node_info.items() if u in X0 and t == 'c')
    Min_b[0] = cost[0]
    parent_state[0] = None
    Xset[0] = X0.copy()
    strategies = {}
    counter = 0
    # print(cand)
    for e in cand:
        if edge_facility(e) == False:
            counter = counter + 1
            strategies[e] = counter
    # print(strategies)

    idx_map = {e: i + 1 for i, e in enumerate(cand)}

    # print(idx_map)
    rep_ids = [edge_id_if_facility(e, strategies, node_info) for e in rep_x]
    for id in rep_ids:
        if id != None:
            children[0].append(id)
            parent_state[id] = 0
    repnum = len(cand) - len(facilities)
    # print(repnum)
    j = 1
    for e, i in strategies.items():
        if (j > repnum):
            break
        rep_e = dfs(e[1], [], {e[0]})
        if (len(rep_e) == 0):
            continue
        cut_new = {e} | set(rep_e)
        Xj = enclosed(cut_new, e[1])
        cost[j] = len(rep_e) - 1

        profit[j] = sum(w for u, (t, w) in node_info.items() if u in Xj and t == 'c')
        Xset[j] = Xj - X0
        rep_ids = [edge_id_if_facility(e, strategies, node_info) for e in rep_e]
        # print(rep_ids)
        for id in rep_ids:
            if id != None:
                children[j].append(id)
                parent_state[id] = j
        j = j + 1
    min_b = [None] * len(cost)

    def min_b_calculater(i):
        if min_b[i] is not None:
            return min_b[i]
        parent = parent_state.get(i)
        if parent is None:
            min_b[i] = cost[i]
        else:
            min_b[i] = min_b_calculater(parent) - 1 + cost[i]

        return min_b[i]

    for i in range(0, len(cost)):
        min_b_calculater(i)
    return cand, cost, profit, Xset, children, parent_state, min_b


from collections import deque, defaultdict


def build_idx_map(adj):
    """Canonical, global edge IDs independent of root x."""
    edges = set()
    for u in adj:
        for v in adj[u]:
            a, b = (u, v) if u <= v else (v, u)
            edges.add((a, b))
    return {e: i + 1 for i, e in enumerate(sorted(edges))}  # 1-based IDs


def canon(edge):
    u, v = edge
    return (u, v) if u <= v else (v, u)


from collections import defaultdict


def mark_customers_by_cand(adj, node_info, idx_map=None):
    # 1) Collect customers
    customers = [u for u, (t, _) in node_info.items() if t == 'c']

    # 2) Build a canonical signature from each X_map to allow dict comparison
    #    We preserve component ids (keys) exactly as they come.
    def xmap_signature(X_map):
        # signature: sorted list of (comp_id, sorted tuple of nodes)
        return tuple(sorted(
            (int(k), tuple(sorted(int(x) for x in v)))
            for k, v in X_map.items()
        ))

    xmap_of = {}
    sig_of = {}
    for u in customers:
        _, _, _, X_map, _, _, _ = build_candidate_tree(adj, node_info, u)
        xmap_of[u] = X_map
        sig_of[u] = xmap_signature(X_map)

    # 3) Group by identical X_map signatures
    groups = defaultdict(list)
    for u in customers:
        groups[sig_of[u]].append(u)

    # 4) Pick a representative per group; mark others visited
    def rank(x):
        return idx_map.get(x, x) if isinstance(idx_map, dict) else x

    visited = {u: False for u in customers}
    reps = {u: False for u in customers}
    for custs in groups.values():
        rep = min(custs, key=rank)
        reps[rep] = True
        for v in custs:
            if v != rep:
                visited[v] = True

    return visited


def solve_subtree_opt(n_c, B, node_info_c, edges_c):
    """Solve one cluster’s interdiction via DP (same logic, faster data structures)."""
    adj = build_adj(n_c, edges_c)
    cost, profit, Xset, min_b, parent = {0: 0}, {0: 0}, {0: set()}, {0: 0}, {0: 0}
    children = defaultdict(list)
    offset = 1
    idx_map = build_idx_map(adj)
    visited_c = mark_customers_by_cand(adj, node_info_c, idx_map)

    cost[0] = 0
    min_b[0] = 0
    profit[0] = 0
    parent[0] = None
    Xset[0] = None

    for x in sorted(u for u, (t, _) in node_info_c.items() if t == 'c'):
        if visited_c[x]:
            continue

        cand, c_map, p_map, X_map, ch_map, parent_state, min_b_map = build_candidate_tree(adj, node_info_c, x)
        block = max(c_map.keys()) + 1

        for i in c_map:
            ni = offset + i
            cost[ni] = c_map[i]
            profit[ni] = p_map[i]
            Xset[ni] = X_map[i]
            if parent_state[i] is None:
                parent[ni] = 0
                children[0].append(ni)
            else:
                parent[ni] = offset + parent_state[i]
            min_b[ni] = min_b_map[i]
        for i, lst in ch_map.items():
            ni = offset + i
            for j in lst:
                children[ni].append(offset + j)
        offset += block

    P = offset  # number of DP indices (0..P-1 valid; your states use 0..P-1)
    # ---- Build bitmasks for Xset (for fast disjointness) ----
    # Map all groups appearing in Xset to bit positions
    groups = sorted({g for k in range(1, P) for g in (Xset[k] or ())})
    gid = {g: i for i, g in enumerate(groups)}
    G = len(groups)
    xmask = [0] * P
    for k in range(1, P):
        m = 0
        if Xset[k]:
            for g in Xset[k]:
                m |= (1 << gid[g])
        xmask[k] = m

    # ---- Allocate triangular DP arrays ----
    # W[i][j][k]  for 0<=i<=B, 0<=j<P, 0<=k<=j
    def tri_init(val_fn):
        # returns list size (B+1) where each item is [ [*] for j in 0..P-1 ]
        arr = []
        for _i in range(B + 1):
            level = []
            for j in range(P):
                level.append([val_fn() for _k in range(j + 1)])
            arr.append(level)
        return arr

    neg_inf = float('-inf')
    W = tri_init(lambda: neg_inf)  # floats
    Vmask = tri_init(lambda: 0)  # ints (bitmasks for coverage)
    Iset = tri_init(lambda: frozenset())  # frozensets of chosen indices

    # ---- Base initialization for i=0 ----
    # If min_b[k] > 0, not feasible at budget 0
    # Else W=0, Iset contains {k} (your original logic had I[(0,j,k)] = {k})
    for j in range(P):
        # k=0..j
        row_min_b = min_b  # local alias
        W0j = W[0][j]
        V0j = Vmask[0][j]
        I0j = Iset[0][j]
        for k in range(j + 1):
            if row_min_b.get(k, 0) > 0:
                W0j[k] = neg_inf
                V0j[k] = 0
                I0j[k] = frozenset()
            else:
                W0j[k] = 0.0
                V0j[k] = 0

                I0j[k] = frozenset({k})

    row_parent = parent
    row_cost = cost
    row_profit = profit
    row_min_b = min_b
    row_xmask = xmask

    for i in range(1, B + 1):
        Wi = W[i];
        Vim = Vmask[i];
        Ii = Iset[i]
        Wim1 = W[i - 1];
        Vim1 = Vmask[i - 1];
        Iim1 = Iset[i - 1]
        for j in range(P):
            Wij = Wi[j];
            Vij = Vim[j];
            Iij = Ii[j]
            Wim1j = Wim1[j];
            Vim1j = Vim1[j];
            Iim1j = Iim1[j]
            # only define these if j>0
            if j > 0:
                Wijm1 = Wi[j - 1];
                Vijm1 = Vim[j - 1];
                Iijm1 = Ii[j - 1]
            for k in range(j + 1):
                if row_min_b.get(k, 0) > i:
                    Wij[k] = neg_inf;
                    Vij[k] = 0;
                    Iij[k] = frozenset()
                    continue

                bestW = 0.0
                bestV = 0
                bestI = frozenset()

                # skip budget
                w = Wim1j[k]
                if w > bestW:
                    bestW, bestV, bestI = w, Vim1j[k], Iim1j[k]

                # skip node index (j-1): only valid if k <= j-1
                if j > 0 and k <= j - 1:
                    w = Wijm1[k]
                    if w > bestW:
                        bestW, bestV, bestI = w, Vijm1[k], Iijm1[k]

                # take k ...
                ck = row_cost.get(k, inf)
                if k > 0 and ck <= i:
                    prev = i - ck
                    p = row_parent.get(k, None)
                    xk = row_xmask[k]
                    Wprev = W[prev][j];
                    Vprev = Vmask[prev][j];
                    Iprev = Iset[prev][j]
                    for kp in range(j + 1):
                        wbase = Wprev[kp]
                        if wbase == neg_inf:
                            continue
                        Ibase = Iprev[kp]
                        if p is not None and p not in Ibase:
                            continue
                        if k in Ibase:
                            continue
                        if xk and (Vprev[kp] & xk):
                            continue
                        cand = wbase + row_profit[k]
                        if cand > bestW:
                            bestW = cand
                            bestV = Vprev[kp] | xk
                            bestI = Ibase | {k}

                Wij[k] = bestW
                Vij[k] = bestV
                Iij[k] = bestI
    out = []
    last_j = P - 1
    Wlast = W
    for i in range(B + 1):
        best = neg_inf
        Wi_last = Wlast[i][last_j]
        for k in range(P):
            v = Wi_last[k] if k <= last_j else neg_inf
            if v > best:
                best = v
        out.append(best)
    return out


def solve_subtree(n_c, B, node_info_c, edges_c):
    """Solve one cluster’s interdiction via DP."""
    adj = build_adj(n_c, edges_c)
    cost, profit, Xset, min_b, parent = {0: 0}, {0: 0}, {0: set()}, {0: 0}, {0: 0}
    children = defaultdict(list)
    offset = 1
    idx_map = build_idx_map(adj)
    # print(idx_map)
    visited_c = mark_customers_by_cand(adj, node_info_c, idx_map)
    # print(visited_c)
    counter = 0

    cost[0] = 0
    min_b[0] = 0
    profit[0] = 0
    parent[0] = None
    Xset[0] = None

    for x in sorted(u for u, (t, _) in node_info_c.items() if t == 'c'):
        if visited_c[x] == True:
            counter = counter + 1
            continue

        cand, c_map, p_map, X_map, ch_map, parent_state, min_b_map = build_candidate_tree(adj, node_info_c, x)
        # print(cand, c_map, p_map, X_map, ch_map, parent_state , min_b_map)
        # print(len(ch_map))
        block = max(c_map.keys()) + 1
        # print(block)
        # children[0].append(offset)
        # print(len(c_map))

        for i in c_map:
            ni = offset + i
            cost[ni] = c_map[i]
            profit[ni] = p_map[i]
            Xset[ni] = X_map[i]
            if parent_state[i] == None:
                parent[ni] = 0
                children[0].append(ni)
            else:
                parent[ni] = offset + parent_state[i]

            # parent_state[ni] = parent_state[i]
            min_b[ni] = min_b_map[i]
        for i, lst in ch_map.items():
            ni = offset + i
            for j in lst:
                children[ni].append(offset + j)
        offset += block

    #print(min_b)
    P = offset
    W = defaultdict(lambda: -inf)
    V = {}
    I = {}
    W[(0, 0, 0)] = 0
    V[(0, 0, 0)] = set()
    I[(0, 0, 0)] = set() | {0}

    for j in range(P):
        for k in range(j + 1):
            if min_b[k] > 0:
                W[(0, j, k)] = float(-inf)
                V[(0, j, k)] = set()
                I[(0, j, k)] = set()

            else:
                W[(0, j, k)] = 0
                V[(0, j, k)] = set()
                I[(0, j, k)] = {k}
    #print("check", min_b[2])
    for i in range(1, B + 1):
        for j in range(P):
            for k in range(j + 1):
                # print(i,j,k)
                if min_b[k] > i:
                    W[(i, j, k)] = float(-inf)
                    V[(i, j, k)] = set()
                    I[(i, j, k)] = set()
                    continue
                if j == 0:
                    W[(i, j, k)] = 0
                    V[(i, j, k)] = set()
                    I[(i, j, k)] = {0}

                key = (i, j, k)
                W[(i, j, k)] = 0
                V[(i, j, k)] = set()
                I[(i, j, k)] = set()

                # skip budgets or nodes
                if i > 0 and W[(i - 1, j, k)] > W[key]:
                    W[key], V[key], I[key] = W[(i - 1, j, k)], V[(i - 1, j, k)], I[(i - 1, j, k)]
                if j > 0 and W[(i, j - 1, k)] > W[key]:
                    W[key], V[key], I[key] = W[(i, j - 1, k)], V[(i, j - 1, k)], I[(i, j - 1, k)]
                # take this node
                if k > 0 and cost[k] <= i:
                    prev = i - cost[k]
                    for kp in range(j + 1):
                        base = (prev, j, kp)


                        Ibase = I[base]
                        Vbase = V[base]


                        p = parent[k]


                        if p is not None and p not in Ibase:  # two O(1) ops on average
                            continue
                        if k in Ibase:
                            continue

                        Xk = Xset[k]
                        if Xk and not Vbase.isdisjoint(Xk):
                            continue

                        candv = W[base] + profit[k]
                        if candv > W[key]:
                            W[key], V[key], I[key] = candv, V[base] | Xset[k], I[base] | {k}




    return [max(W[(i, P - 1, k)] for k in range(P)) for i in range(B + 1)]


from collections import defaultdict


def solve_subtree_ultra_optimized(n_c, B, node_info_c, edges_c):
    """
    Ultra-optimized single-function cluster interdiction solver.
    Combines all operations for maximum performance.
    """
    # Build adjacency and process candidates in one pass
    adj = build_adj(n_c, edges_c)
    idx_map = build_idx_map(adj)
    visited_customers = mark_customers_by_cand(adj, node_info_c, idx_map)

    # Pre-allocate all data structures
    cost = {0: 0}
    profit = {0: 0}
    xset = {0: set()}
    min_budget = {0: 0}
    parent = {0: None}
    children = defaultdict(list)
    offset = 1

    # Process candidates and build state in single loop
    candidates = sorted(u for u, (t, _) in node_info_c.items() if t == 'c')
    for candidate in candidates:
        if visited_customers[candidate]:
            continue

        # Inline candidate tree building
        tree_data = build_candidate_tree(adj, node_info_c, candidate)
        cand, c_map, p_map, x_map, ch_map, parent_state, min_b_map = tree_data
        block_size = max(c_map.keys()) + 1

        # Direct state merging (no function calls)
        for i in c_map:
            ni = offset + i
            cost[ni] = c_map[i]
            profit[ni] = p_map[i]
            xset[ni] = x_map[i]
            min_budget[ni] = min_b_map[i]

            if parent_state[i] is None:
                parent[ni] = 0
                children[0].append(ni)
            else:
                parent[ni] = offset + parent_state[i]

        for i, lst in ch_map.items():
            ni = offset + i
            children[ni].extend(offset + j for j in lst)

        offset += block_size

    P = offset

    # Setup bitmasks inline for maximum speed
    all_groups = set()
    for k in range(1, P):
        if xset[k]:
            all_groups.update(xset[k])

    groups = sorted(all_groups)
    group_bits = {g: i for i, g in enumerate(groups)}

    # Pre-compute all bitmasks
    bitmasks = [0] * P
    for k in range(1, P):
        if xset[k]:
            mask = 0
            for g in xset[k]:
                mask |= (1 << group_bits[g])
            bitmasks[k] = mask

    # Allocate triangular DP arrays with optimal memory layout
    NEG_INF = float('-inf')

    # Use list comprehensions for fastest initialization
    W = [[[NEG_INF for _ in range(k + 1)] for k in range(P)] for _ in range(B + 1)]
    V = [[[0 for _ in range(k + 1)] for k in range(P)] for _ in range(B + 1)]
    I = [[[frozenset() for _ in range(k + 1)] for k in range(P)] for _ in range(B + 1)]

    # Initialize base case (budget = 0) with direct indexing
    for j in range(P):
        W0j, V0j, I0j = W[0][j], V[0][j], I[0][j]
        for k in range(j + 1):
            if min_budget.get(k, 0) > 0:
                W0j[k] = NEG_INF
                V0j[k] = 0
                I0j[k] = frozenset()
            else:
                W0j[k] = 0.0
                V0j[k] = 0
                I0j[k] = frozenset({k})

    # Main DP with aggressive optimizations
    for i in range(1, B + 1):
        Wi, Vim, Ii = W[i], V[i], I[i]
        Wim1, Vim1, Iim1 = W[i - 1], V[i - 1], I[i - 1]

        for j in range(P):
            Wij, Vij, Iij = Wi[j], Vim[j], Ii[j]
            Wim1j, Vim1j, Iim1j = Wim1[j], Vim1[j], Iim1[j]

            # Cache for j-1 access (only when needed)
            if j > 0:
                Wijm1, Vijm1, Iijm1 = Wi[j - 1], Vim[j - 1], Ii[j - 1]

            for k in range(j + 1):
                min_b_k = min_budget.get(k, 0)
                if min_b_k > i:
                    Wij[k], Vij[k], Iij[k] = NEG_INF, 0, frozenset()
                    continue

                # Initialize with defaults
                best_w, best_v, best_i = 0.0, 0, frozenset()

                # Transition 1: Skip budget (most common case first)
                w1 = Wim1j[k]
                if w1 > best_w:
                    best_w, best_v, best_i = w1, Vim1j[k], Iim1j[k]

                # Transition 2: Skip node index
                if j > 0 and k < j:  # k <= j-1
                    w2 = Wijm1[k]
                    if w2 > best_w:
                        best_w, best_v, best_i = w2, Vijm1[k], Iijm1[k]

                # Transition 3: Take node k (most expensive, check last)
                if k > 0:
                    cost_k = cost.get(k, float('inf'))
                    if cost_k <= i:
                        prev_budget = i - cost_k
                        parent_k = parent.get(k)
                        xmask_k = bitmasks[k]
                        profit_k = profit[k]

                        # Inline the base state search for maximum speed
                        Wprev, Vprev, Iprev = W[prev_budget][j], V[prev_budget][j], I[prev_budget][j]

                        for kp in range(j + 1):
                            w_base = Wprev[kp]
                            if w_base == NEG_INF:
                                continue

                            i_base = Iprev[kp]

                            # Fast constraint checking with short-circuit evaluation
                            if (parent_k is not None and parent_k not in i_base or
                                    k in i_base or
                                    xmask_k and (Vprev[kp] & xmask_k)):
                                continue

                            candidate_profit = w_base + profit_k
                            if candidate_profit > best_w:
                                best_w = candidate_profit
                                best_v = Vprev[kp] | xmask_k
                                best_i = i_base | {k}

                # Store results
                Wij[k], Vij[k], Iij[k] = best_w, best_v, best_i

    # Extract results inline
    results = []
    last_j = P - 1
    for i in range(B + 1):
        max_profit = NEG_INF
        Wi_last = W[i][last_j]
        for k in range(P):
            if k <= last_j:
                max_profit = max(max_profit, Wi_last[k])
        results.append(max_profit)

    return results
def solve_subtree_ultra_optimized_B(n_c, B, node_info_c, edges_c):

    from collections import defaultdict  # in case not imported outside

    # ---- Cap budget by number of facilities ----
    orig_B = B
    num_facilities = sum(1 for t, _ in node_info_c.values() if t == 'f')
    B_eff = min(B, num_facilities)

    # Build adjacency and process candidates in one pass
    adj = build_adj(n_c, edges_c)
    idx_map = build_idx_map(adj)
    visited_customers = mark_customers_by_cand(adj, node_info_c, idx_map)

    # Pre-allocate all data structures
    cost = {0: 0}
    profit = {0: 0}
    xset = {0: set()}
    min_budget = {0: 0}
    parent = {0: None}
    children = defaultdict(list)
    offset = 1

    # Process candidates and build state in single loop
    candidates = sorted(u for u, (t, _) in node_info_c.items() if t == 'c')
    for candidate in candidates:
        if visited_customers[candidate]:
            continue

        # Inline candidate tree building
        tree_data = build_candidate_tree(adj, node_info_c, candidate)
        cand, c_map, p_map, x_map, ch_map, parent_state, min_b_map = tree_data
        block_size = max(c_map.keys()) + 1

        # Direct state merging (no function calls)
        for i in c_map:
            ni = offset + i
            cost[ni] = c_map[i]
            profit[ni] = p_map[i]
            xset[ni] = x_map[i]
            min_budget[ni] = min_b_map[i]

            if parent_state[i] is None:
                parent[ni] = 0
                children[0].append(ni)
            else:
                parent[ni] = offset + parent_state[i]

        for i, lst in ch_map.items():
            ni = offset + i
            children[ni].extend(offset + j for j in lst)

        offset += block_size

    P = offset

    # Setup bitmasks inline for maximum speed
    all_groups = set()
    for k in range(1, P):
        if xset[k]:
            all_groups.update(xset[k])

    groups = sorted(all_groups)
    group_bits = {g: i for i, g in enumerate(groups)}

    # Pre-compute all bitmasks
    bitmasks = [0] * P
    for k in range(1, P):
        if xset[k]:
            mask = 0
            for g in xset[k]:
                mask |= (1 << group_bits[g])
            bitmasks[k] = mask

    # Allocate triangular DP arrays with optimal memory layout
    NEG_INF = float('-inf')

    # NOTE: allocate using B_eff (capped by number of facilities)
    W = [[[NEG_INF for _ in range(k + 1)] for k in range(P)] for _ in range(B_eff + 1)]
    V = [[[0 for _ in range(k + 1)] for k in range(P)] for _ in range(B_eff + 1)]
    I = [[[frozenset() for _ in range(k + 1)] for k in range(P)] for _ in range(B_eff + 1)]

    # Initialize base case (budget = 0) with direct indexing
    for j in range(P):
        W0j, V0j, I0j = W[0][j], V[0][j], I[0][j]
        for k in range(j + 1):
            if min_budget.get(k, 0) > 0:
                W0j[k] = NEG_INF
                V0j[k] = 0
                I0j[k] = frozenset()
            else:
                W0j[k] = 0.0
                V0j[k] = 0
                I0j[k] = frozenset({k})

    # Main DP with aggressive optimizations (iterate only up to B_eff)
    for i in range(1, B_eff + 1):
        Wi, Vim, Ii = W[i], V[i], I[i]
        Wim1, Vim1, Iim1 = W[i - 1], V[i - 1], I[i - 1]

        for j in range(P):
            Wij, Vij, Iij = Wi[j], Vim[j], Ii[j]
            Wim1j, Vim1j, Iim1j = Wim1[j], Vim1[j], Iim1[j]

            # Cache for j-1 access (only when needed)
            if j > 0:
                Wijm1, Vijm1, Iijm1 = Wi[j - 1], Vim[j - 1], Ii[j - 1]

            for k in range(j + 1):
                min_b_k = min_budget.get(k, 0)
                if min_b_k > i:
                    Wij[k], Vij[k], Iij[k] = NEG_INF, 0, frozenset()
                    continue

                # Initialize with defaults
                best_w, best_v, best_i = 0.0, 0, frozenset()

                # Transition 1: Skip budget
                w1 = Wim1j[k]
                if w1 > best_w:
                    best_w, best_v, best_i = w1, Vim1j[k], Iim1j[k]

                # Transition 2: Skip node index
                if j > 0 and k < j:
                    w2 = Wijm1[k]
                    if w2 > best_w:
                        best_w, best_v, best_i = w2, Vijm1[k], Iijm1[k]

                # Transition 3: Take node k
                if k > 0:
                    cost_k = cost.get(k, float('inf'))
                    if cost_k <= i:
                        prev_budget = i - cost_k
                        parent_k = parent.get(k)
                        xmask_k = bitmasks[k]
                        profit_k = profit[k]

                        Wprev, Vprev, Iprev = W[prev_budget][j], V[prev_budget][j], I[prev_budget][j]

                        for kp in range(j + 1):
                            w_base = Wprev[kp]
                            if w_base == NEG_INF:
                                continue

                            i_base = Iprev[kp]

                            # Fast constraint checks
                            if (parent_k is not None and parent_k not in i_base or
                                k in i_base or
                                (xmask_k and (Vprev[kp] & xmask_k))):
                                continue

                            candidate_profit = w_base + profit_k
                            if candidate_profit > best_w:
                                best_w = candidate_profit
                                best_v = Vprev[kp] | xmask_k
                                best_i = i_base | {k}

                # Store results
                Wij[k], Vij[k], Iij[k] = best_w, best_v, best_i

    # Extract results up to B_eff
    results = []
    last_j = P - 1
    for i in range(B_eff + 1):
        max_profit = NEG_INF
        Wi_last = W[i][last_j]
        for k in range(P):
            if k <= last_j:
                if Wi_last[k] > max_profit:
                    max_profit = Wi_last[k]
        results.append(max_profit)

    # ---- Pad results if original B > B_eff ----
    if orig_B > B_eff:
        tail = results[-1]
        results.extend([tail] * (orig_B - B_eff))

    return results

def get_cluster_subproblems(n, node_info, edges):

    clusters = make_clusters(n, edges, node_info)
    subs = []
    for nodeset, cl_edges in clusters:
        mapping = {u: idx + 1 for idx, u in enumerate(sorted(nodeset))}
        node_info_c = {mapping[u]: node_info[u] for u in nodeset}
        edges_c = [(mapping[u], mapping[v]) for u, v in cl_edges]
        subs.append((node_info_c, edges_c))
    return subs


def full_interdiction(cluster_subproblems, B):

    cluster_profits = [
        solve_subtree_ultra_optimized_B(len(nc), B, nc, ec)
        for nc, ec in cluster_subproblems
    ]
    #print(cluster_profits)
    # global knapsack
    dp = [0] * (B + 1)
    for profs in cluster_profits:
        newdp = [0] * (B + 1)
        for b in range(B + 1):
            for k in range(b + 1):
                newdp[b] = max(newdp[b], dp[b - k] + profs[k])
        dp = newdp
    return dp[B]


def convert_text_file_to_array(filename):
    with open(filename, "r") as f:
        return [block for block in f.read().strip().split("\n\n") if block.strip()]


def main():
    tree_str = convert_text_file_to_array("newData/trees_Edge_1000Total_600N_0.4F_E_Integer.txt")
    start = timeit.default_timer()
    results = []
    i=0
    time = []
    for data in tree_str:
        i=i+1
        temp1 = timeit.default_timer()
        n, B, node_info, edges = parse_input(data)
        subs = get_cluster_subproblems(n, node_info, edges)
        ans = full_interdiction(subs, B)
        print(ans)
        results.append(ans)
        temp2 = timeit.default_timer()
        time.append(temp2 - temp1)
        print(i, "process time",temp2 - temp1 , "Running time",temp2 - start)
    #print(results)
    total = timeit.default_timer() - start
    average_time = total / len(results)
    pop_var = np.var(time)
    sigma_time = math.sqrt(pop_var)


    print(total)
    with open("new_results/Final_Result_FR_Final_T_1000_N_600_0.4_f.txt", "w") as f:
        for i, ans in enumerate(results):
            f.write(f"Answer {i}: {ans} , Time : {time[i]}\n")
        f.write(f"\n Average run time: {average_time:.4f} seconds\n")
        f.write(f"\n Run time SD: {sigma_time:.4f} seconds\n")


if __name__ == "__main__":
    main()
