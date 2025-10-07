from collections import deque
from itertools import combinations
import timeit
import numpy as np
import math

def constrained_mckp_dp(costs, values, properties, B):
    # V0: none of the chosen options so far had p==1
    # V1: at least one chosen option so far had p==1
    V0_prev = np.full(B + 1, -np.inf, dtype=np.float64)
    V1_prev = np.full(B + 1, -np.inf, dtype=np.float64)
    V0_prev[0] = 0.0

    for k in range(len(costs)):  # for each child/group
        # IMPORTANT: start as -inf so you MUST choose exactly one option in this group
        V0 = np.full(B + 1, -np.inf, dtype=np.float64)
        V1 = np.full(B + 1, -np.inf, dtype=np.float64)

        c_k = costs[k]; v_k = values[k]; p_k = properties[k]
        for j in range(len(c_k)):
            c = c_k[j]; v = v_k[j]; p = p_k[j]
            if c <= B:
                base = slice(0, B + 1 - c)
                bs   = slice(c, B + 1)

                # choose this option after a state with no 'p==1' so far
                cand0 = V0_prev[base] + v
                if p == 0:
                    np.maximum(V0[bs], cand0, out=V0[bs])
                else:
                    np.maximum(V1[bs], cand0, out=V1[bs])

                # choose this option after a state that already had 'p==1'
                cand1 = V1_prev[base] + v
                np.maximum(V1[bs], cand1, out=V1[bs])

        V0_prev, V1_prev = V0, V1

    return V1_prev  # final vector length B+1

def constrained_mckp_dp_classic(costs, values, properties, B):

    M = len(costs)
    # DP tables
    V0 = np.full((M + 1, B + 1), -np.inf)
    V1 = np.full((M + 1, B + 1), -np.inf)
    # Selection tracking for V1
    sel1 = np.full((M + 1, B + 1), -1, dtype=int)

    V0[0, 0] = 0

    for k in range(1, M + 1):
        for j in range(len(costs[k - 1])):
            c = costs[k - 1][j]
            v = values[k - 1][j]
            p = properties[k - 1][j]

            if c <= B:
                b_slice = slice(c, B + 1)

                # Transition from V0[k-1]
                cand0 = V0[k - 1, :B + 1 - c] + v
                if p == 0:
                    mask0 = cand0 > V0[k, b_slice]
                    V0[k, b_slice][mask0] = cand0[mask0]
                else:
                    mask1 = cand0 > V1[k, b_slice]
                    V1[k, b_slice][mask1] = cand0[mask1]
                    sel1[k, b_slice][mask1] = c

                # Transition from V1[k-1]
                cand1 = V1[k - 1, :B + 1 - c] + v
                mask1_2 = cand1 > V1[k, b_slice]
                V1[k, b_slice][mask1_2] = cand1[mask1_2]
                sel1[k, b_slice][mask1_2] = c

    return V1[M, :]

def constrained_mckp_dp_efficient(costs, values, properties, B):
    M = len(costs)
    V0 = np.full((M + 1, B + 1), -np.inf)
    V1 = np.full((M + 1, B + 1), -np.inf)
    V0[0, 0] = 0

    for k in range(1, M + 1):
        for j in range(len(costs[k - 1])):
            cost = costs[k - 1][j]
            value = values[k - 1][j]
            prop = properties[k - 1][j]
            if cost <= B:
                # Create a vector for all budgets b where b >= cost.
                b_slice = slice(cost, B + 1)
                # Candidate values computed from V0[k-1] for the valid range.
                candidate = V0[k - 1, :B + 1 - cost] + value

                if prop == 0:
                    V0[k, b_slice] = np.maximum(V0[k, b_slice], candidate)
                else:
                    V1[k, b_slice] = np.maximum(V1[k, b_slice], candidate)

                # Also update using values from V1[k-1]
                candidate2 = V1[k - 1, :B + 1 - cost] + value
                V1[k, b_slice] = np.maximum(V1[k, b_slice], candidate2)
    return V1[M, :]


def read_input(data):
    lines = data.strip().split('\n')
    n, r = map(int, lines[0].split())
    # print(r)
    node_info = {}
    idx = 1
    for _ in range(n):
        parts = lines[idx].split()
        idx += 1
        u = int(parts[0])
        typ = parts[1]
        if typ == 'f':
            node_info[u] = ('f', 0)
        else:
            w = int(parts[2])
            node_info[u] = ('c', w)

    m = int(lines[idx]);
    idx += 1
    edges = []
    for _ in range(m):
        u, v = map(int, lines[idx].split())
        idx += 1
        edges.append((u, v))
    return n, r, node_info, edges


def build_tree(n, edges, root=1):
    # Create the adjacency list.
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Initialize parent, children, and height arrays.
    height = [0] * (n + 1)
    parent = [0] * (n + 1)
    children = [[] for _ in range(n + 1)]

    # Set the root's parent to -1 and its height (0 or 1, here we choose 0).
    parent[root] = -1
    height[root] = 0

    q = deque([root])
    visited = [False] * (n + 1)
    visited[root] = True

    while q:
        u = q.popleft()
        for w in adj[u]:
            if not visited[w]:
                visited[w] = True
                parent[w] = u
                # Set the child's height as parent's height + 1.
                height[w] = height[u] + 1
                children[u].append(w)
                q.append(w)
    # print(parent, children)
    return parent, children, height


def dfs(node, children, visited, node_info, d0, d1, d2, budget , budgetarray):
    visited[node] = True
    # print(f'Visited node: {node}, Type: {node_info[node][0]}, Weight: {node_info[node][1]}')
    for child in children[node]:
        if not visited[child]:
            dfs(child, children, visited, node_info, d0, d1, d2, budget , budgetarray)
    compute_dp(node, budget, node_info, children, d0, d1, d2,budgetarray)
    #print(node)
    #print(d0[10], d1[10], d2[10])


def perform_dfs(n, budget, node_info, edges):
    _, children, _ = build_tree(n, edges, root=1)
    visited = [False] * (n + 1)
    d0, d1, d2 , sel = initialize_dp(n, budget, node_info, children)
    dfs(1, children, visited, node_info, d0, d1, d2, budget , sel)
    #print(sel)
    return max(d0[1][budget], d1[1][budget], d2[1][budget])


def initialize_dp(n, budget, node_info, children):
    INF = float('-inf')
    d0 = [[INF] * (budget + 1) for _ in range(n + 1)]
    d1 = [[INF] * (budget + 1) for _ in range(n + 1)]
    d2 = [[INF] * (budget + 1) for _ in range(n + 1)]
    sel = [[INF] * (budget + 1) for _ in range(n + 1)]

    return d0, d1, d2 , sel


def compute_dp(u, budget, node_info, children, d0, d1, d2 , sel):
    typ, w = node_info[u]
    INF = float('-inf')

    if len(children[u]) == 0:
        for i in range(budget + 1):
            # print(i)
            if typ == 'f':

                d0[u][i] = INF
                d1[u][i] = 0
                d2[u][i] = INF
                sel[u][i] = -1


            else:

                d0[u][i] = w
                d1[u][i] = INF
                d2[u][i] = 0
                sel[u][i] = -1

        return
    if typ == 'f':
        for i in range(budget + 1):
            d0[u][i] = INF
            d2[u][i] = INF
        child_options_for_d0_f = []
        child_options_for_d0_f_c = []
        child_options_for_d0_f_p = []
        for cnode in children[u]:
            arr_v = []
            arr_c = []
            arr_p = []
            for B in range(budget + 1):
                if B >= 1:
                    arr_v.append(max(d0[cnode][B - 1], d1[cnode][B], d2[cnode][B]))
                else:
                    arr_v.append(max(d1[cnode][B], d2[cnode][B]))
                arr_c.append(B)
                arr_p.append(1)
            # print(arr_v,arr_c,arr_p)
            child_options_for_d0_f.append(arr_v)
            child_options_for_d0_f_c.append(arr_c)
            child_options_for_d0_f_p.append(arr_p)
        # print("hello")
        result= constrained_mckp_dp(child_options_for_d0_f_c, child_options_for_d0_f, child_options_for_d0_f_p, B)

        for B in range(budget + 1):
            d1[u][B] = result[B]

    if typ == 'c':
        child_options_for_d0 = []
        child_options_for_d0_c = []
        child_options_for_d0_p = []
        for cnode in children[u]:
            arr_v = []
            arr_c = []
            arr_p = []
            for B in range(budget + 1):
                if B >= 1:
                    arr_v.append(max(d0[cnode][B], d1[cnode][B - 1]))
                else:
                    arr_v.append(d0[cnode][B])
                arr_c.append(B)
                arr_p.append(1)
            child_options_for_d0.append(arr_v)
            child_options_for_d0_c.append(arr_c)
            child_options_for_d0_p.append(arr_p)
        result= constrained_mckp_dp(child_options_for_d0_c, child_options_for_d0, child_options_for_d0_p, B)
        #print(selr)
        for B in range(budget + 1):
            d0[u][B] = result[B] + w

    if typ == 'c':
        child_options_for_d1 = []
        child_options_for_d1_c = []
        child_options_for_d1_p = []
        for cnode in children[u]:
            arr_v = []
            arr_c = []
            arr_p = []
            for B in range(budget + 1):
                for z in {0, 1}:
                    if z == 0:
                        if (B >= 1):
                            arr_v.append(max(d0[cnode][B - 1], d1[cnode][B - 1], d2[cnode][B]))
                        else:
                            arr_v.append(d2[cnode][B])
                    else:
                        arr_v.append(d1[cnode][B])

                    arr_c.append(B)
                    arr_p.append(z)
            # print(arr_v,arr_c,arr_p)
            child_options_for_d1.append(arr_v)
            child_options_for_d1_c.append(arr_c)
            child_options_for_d1_p.append(arr_p)
        #print(child_options_for_d1_c, child_options_for_d1, child_options_for_d1_p, B)
        result= constrained_mckp_dp(child_options_for_d1_c, child_options_for_d1, child_options_for_d1_p, B)
       # print(selr)

        for B in range(budget + 1):
            d1[u][B] = result[B]

    if typ == 'c':
        child_options_for_d2 = []
        child_options_for_d2_c = []
        child_options_for_d2_p = []
        for cnode in children[u]:
            arr_v = []
            arr_c = []
            arr_p = []
            for B in range(budget + 1):
                if (B >= 1):
                    arr_v.append(max(d1[cnode][B - 1], d2[cnode][B], d0[cnode][B - 1]))
                else:
                    arr_v.append(d2[cnode][B])
                arr_c.append(B)
                arr_p.append(1)
            child_options_for_d2.append(arr_v)
            child_options_for_d2_c.append(arr_c)
            child_options_for_d2_p.append(arr_p)
        result = constrained_mckp_dp(child_options_for_d2_c, child_options_for_d2, child_options_for_d2_p, B)
        #print(selr)
        for B in range(budget + 1):
            d2[u][B] = result[B]


def convert_text_file_to_array(filename):
    with open(filename, "r") as file:
        content = file.read()
    return [block for block in content.strip().split("\n\n") if block.strip()]


def main():

    tree_str = convert_text_file_to_array("newData/trees_Edge_100Total_1000N_0.4F_E_Integer.txt")

    start_time = timeit.default_timer()
    answers = []
    time = []
    for i in range(len(tree_str)):
        print(i)
        temp_time_s = timeit.default_timer()
        n, budget, node_info, edges = read_input(tree_str[i])

        parent, children , height = build_tree(n, edges, root=1)
        result = perform_dfs(n, budget, node_info, edges)
        answers.append(result)
        print(budget , result)
        temp_time = timeit.default_timer() - temp_time_s
        time.append(temp_time)
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    average_time = total_time / len(answers)
    pop_var = np.var(time)
    sigma_time = math.sqrt(pop_var)

    with open("new_results/Final_Results_DP_T_100_N_1000_P_0.4_test.txt", "w") as f:
        for idx, ans in enumerate(answers):
            f.write(f"Answer {idx + 1}: {ans} , Time : {time[idx]}\n")
        f.write(f"\nAverage run time: {average_time:.4f} seconds\n")
        f.write(f"\n Run time SD: {sigma_time:.4f} seconds\n")


if __name__ == "__main__":
    main()
