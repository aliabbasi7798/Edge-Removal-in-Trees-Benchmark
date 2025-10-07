import random

import numpy as np
import networkx as nx

from networkx.readwrite.json_graph.tree import tree_data


def generate_tree_data(n, seed=None):
    tree = nx.random_tree(n, seed=seed)
    p =0.4
    node_attrs = {}
    for node in tree.nodes():
        if random.uniform(0, 1) < p:
            node_attrs[node] = ('f', None)
        else:
            #cost = random.uniform(1, 1_000_000)
            #cost = 0
            cost = random.randint(1 , 1000000)
            node_attrs[node] = ('c', cost)

    count_f = sum(1 for t, _ in node_attrs.values() if t == 'f')

    #rand_b = random.randint(0, n-1)
    rand_b = 100
    header = f"{n} {rand_b}"

    node_lines = []
    for node in sorted(tree.nodes()):
        typ, cost = node_attrs[node]
        if typ == 'f':
            node_lines.append(f"{node + 1} f")
        else:
            node_lines.append(f"{node + 1} c {cost}")

    edge_lines = []
    for u, v in tree.edges():
        a, b = min(u, v), max(u, v)
        edge_lines.append(f"{a + 1} {b + 1}")

    parts = [
        header,
        "\n".join(node_lines),
        f"{n - 1}",
        "\n".join(edge_lines)
    ]
    data_str = "\n".join(parts)
    return data_str


def main():
    tree_str_list = []
    #seed = random.randint(0, 100)
    for i in range(1000):
        for j in [600]:

            n = j
            tree_data_str = generate_tree_data(n, seed=None)
            tree_str_list.append(tree_data_str)


    all_trees_data = "\n\n".join(tree_str_list)

    # Save the combined tree data to one file.
    filename = "newData/trees_Edge_1000Total_600N_0.4F_E_Integer.txt"
    with open(filename, "w") as file:
        file.write(all_trees_data)

    print(f"Saved {len(tree_str_list)} trees in {filename}.")


if __name__ == "__main__":
    main()
