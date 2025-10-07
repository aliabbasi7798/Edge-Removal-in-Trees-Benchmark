import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import timeit
import numpy as np
import math

def parse_input(data):
    lines = data.strip().split("\n")
    n, m = map(int, lines[0].split())
    node_data = []
    for line in lines[1:n+1]:
        parts = line.split()
        if parts[1] == 'f':
            node_data.append((int(parts[0]), 'f'))
        else:
            node_data.append((int(parts[0]), 'c', int(parts[2])))
    edge_data = [tuple(map(int, line.split())) for line in lines[n+2:]]
    return n, m, node_data, edge_data

def build_graph(node_data, edge_data):
    V = [node[0] for node in node_data]
    S = [node[0] for node in node_data if node[1] == 'f']
    W = {node[0]: node[2] for node in node_data if node[1] == 'c'}
    E_set = [frozenset(e) for e in edge_data]

    T = nx.Graph()
    T.add_nodes_from(V)
    T.add_edges_from(edge_data)

    paths = {}
    for v in V:
        if v not in S:
            paths[v] = {}
            for s in S:
                try:
                    paths[v][s] = nx.shortest_path(T, source=v, target=s)
                except nx.NetworkXNoPath:
                    paths[v][s] = []

    return V, S, W, E_set, paths

def optimize_model(V, S, W, E_set, paths, B):
    m = gp.Model("max_disconnected_customers")

    x = m.addVars([v for v in V if v not in S], vtype=GRB.BINARY, name="x")
    y = m.addVars(E_set, vtype=GRB.BINARY, name="y")

    m.setObjective(gp.quicksum(W[v] * (1 - x[v]) for v in V if v not in S), GRB.MAXIMIZE)

    for v in V:
        if v not in S:
            for s in S:
                path_edges = [frozenset((paths[v][s][i], paths[v][s][i + 1])) for i in range(len(paths[v][s]) - 1)]
                m.addConstr(x[v] + gp.quicksum(y[e] for e in path_edges) >= 1,
                            name=f"path_constr_{v}_{s}")

    m.addConstr(y.sum() <= B, name="max_edges_removed")

    m.optimize()

    return m, x, y

def solve_with_data(data):
    n, B, node_data, edge_data = parse_input(data)
    V, S, W, E_set, paths = build_graph(node_data, edge_data)
    m, x, y = optimize_model(V, S, W, E_set, paths, B)
    maxp = 0
    if m.status == GRB.OPTIMAL:
        disconnected_customers = [v for v in V if v not in S and x[v].X < 0.5]
        removed_edges = [tuple(e) for e in E_set if y[e].X > 0.5]
        #print(f"Disconnected customers: {disconnected_customers}")
        #print(f"Removed edges: {removed_edges}")
        maxp = sum(W[v] for v in disconnected_customers)
        #print(f"Total weight disconnected: {maxp}")
    return maxp
def convert_text_file_to_array(filename):
    with open(filename, "r") as file:
        content = file.read()
    return [block for block in content.strip().split("\n\n") if block.strip()]

if __name__ == "__main__":
    tree_str = convert_text_file_to_array("old_data/trees_Edge_100Total_300N_0.5F_E_Integer.txt")
    start_time = timeit.default_timer()
    answers = []
    time = []
    for i in range(len(tree_str)):
        temp_time_str = timeit.default_timer()
        answers.append(solve_with_data(tree_str[i]))
        temp_time_str_2 = timeit.default_timer() - temp_time_str
        time.append(temp_time_str_2)


    end_time = timeit.default_timer()
    total_time = end_time - start_time
    average_time = total_time / len(answers)
    pop_var = np.var(time)
    sigma_time = math.sqrt(pop_var)
    with open("oldplots/Final_Results_Groubi_T_100_N_300_0.5.txt", "w") as f:
        for idx, ans in enumerate(answers):
            f.write(f"Answer {idx + 1}: {ans} , Time : {time[idx]}\n")
        f.write(f"\nAverage run time: {average_time:.4f} seconds\n")
        f.write(f"\nRun time Variance: {sigma_time:.4f} seconds\n")
