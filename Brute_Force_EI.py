from collections import deque
from itertools import combinations
import timeit


def solve_with_data(data_str):
    lines = data_str.strip().split("\n")


    n, b = map(int, lines[0].split())
    node_type = {}
    node_weight = {}

    idx = 1
    for _ in range(n):
        parts = lines[idx].split()
        idx += 1

        node_id = int(parts[0])
        node_kind = parts[1]

        if node_kind == "f":
            node_type[node_id] = "F"
            node_weight[node_id] = 0
        else:
            node_type[node_id] = "C"
            weight = int(parts[2])
            node_weight[node_id] = weight


    e = int(lines[idx])
    idx += 1

    edges = []
    for _ in range(e):
        u, v = map(int, lines[idx].split())
        idx += 1
        edges.append((u, v))

    # Build the adjacency list
    graph = {i: [] for i in range(1, n + 1)}
    for (u, v) in edges:
        graph[u].append(v)
        graph[v].append(u)


    def compute_lost_weight(removed_edges):
        new_graph = {i: set(graph[i]) for i in range(1, n + 1)}

        # Remove selected edges
        for u, v in removed_edges:
            new_graph[u].discard(v)
            new_graph[v].discard(u)


        def bfs(start):
            queue = deque([start])
            visited = {start}
            component = set()
            while queue:
                cur = queue.popleft()
                component.add(cur)
                for nxt in new_graph[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            return component


        visited_nodes = set()
        connected_nodes = set()
        for node in range(1, n + 1):
            if node not in visited_nodes and node_type[node] == "F":
                component = bfs(node)
                visited_nodes |= component
                connected_nodes |= component


        lost_sum = sum(
            node_weight[node] for node in range(1, n + 1) if node_type[node] == "C" and node not in connected_nodes
        )

        return lost_sum


    best_lost_weight = 0

    for remove_size in range(1, b + 1):
        for subset in combinations(edges, remove_size):
            lost_w = compute_lost_weight(subset)
            if lost_w > best_lost_weight:
                best_lost_weight = lost_w


    return best_lost_weight


def convert_text_file_to_array(filename):
    with open(filename, "r") as file:
        content = file.read()
    return [block for block in content.strip().split("\n\n") if block.strip()]


if __name__ == "__main__":
    tree_str = convert_text_file_to_array("previouspaperresults/testFR.txt")
    start_time = timeit.default_timer()
    answers =[]
    for i in range(len(tree_str)):
        answers.append(solve_with_data(tree_str[i]))
    end_time = timeit.default_timer()

    total_time = end_time - start_time
    print(answers)
    with open("previouspaperresults/Results_Bruteforce_test.txt", "w") as f:
        for idx, ans in enumerate(answers):
            f.write(f"Answer {idx + 1}: {ans}\n")
        f.write(f"\nTotal run time: {total_time:.4f} seconds\n")