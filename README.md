# Edge Removal Interdiction – Algorithmic Comparison

This repository implements and evaluates three algorithms for the **Edge Removal Interdiction Problem** on tree networks:

- **Fröhlich–Ruzika (FR) algorithm** – dynamic programming formulation from *Fröhlich & Ruzika (2022)*  
- **Gurobi optimizer** – exact ILP formulation solved using the Gurobi engine  
- **Proposed Dynamic Programming (DP)** – optimized algorithm with reduced state space and efficient subtree merging  

---

## Overview

The interdiction problem aims to **select a subset of edges under a budget constraint** to **maximize disruption between customers and facilities**.  
All algorithms are tested on identical, synthetically generated tree instances from the edge-removal interdiction setting.

---

## Data and Evaluation

Instances are generated with parameters:

- `N`: number of nodes  
- `B`: interdiction budget  
- `p_f`: facility probability  

Comparisons include interdiction value, runtime, and scalability across different network sizes.

---

## Reference

Fröhlich, F., & Ruzika, S. (2022). *Interdicting Facilities in Tree Networks.*  
*European Journal of Operational Research.*
