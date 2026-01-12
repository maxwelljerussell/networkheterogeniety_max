# Network Heterogeneity - 'HeterogeneityAnalyzer'

This repository provides tools for analysing structural heterogeneity of networks.
The main interface is the `HeterogeneityAnalyzer` class, which combines multiple heterogeneity metrics into a single, user-friendly API.

## Enviroment Setup

### **Create a virtual environment**
```powershell
python -m venv .venv
.venv\Scipts\activate
```

### **Install required dependencies**
```bash
pip install -r requirements.txt
```

## Metrics

Currently supported metrics


### Degree variance  
**Reference:**  
F.K. Bell, *A note on the irregularity of graphs*, Linear Algebra and its Applications, 1992.  
https://doi.org/10.1016/0024-3795(92)90004-T

Degree variance measures how unevenly node degrees are distributed.

Equation:

$$
σ_d^2 = (1/N) * Σ_i (d_i - \overline{d})^2
$$

where the mean degree is:

$$
\overline{d} = (1/N) * Σ_i d_i
$$

---

### Degree distribution heterogeneity  
**Reference:**  
Jacob, R., Harikrishnan, K.P., Misra, R., & Ambika, G. (2017).  
*Measure for degree heterogeneity in complex networks and its application to recurrence network analysis.*  
Royal Society Open Science, 4(1), 160757.  
https://doi.org/10.1098/rsos.160757

Let $P(k)$ be the fraction of nodes with degree $k$.

The intermediate quantity $h^2$ is:

$$
h^2 = (1/N) * Σ_k [ P(k) * (1 - P(k))^2 ]
$$

The theoretical maximum heterogeneity $h_{het}$ is:

$$
h_{het} = sqrt( 1 - 3/N + (N + 2) / N^3 )
$$

The normalized heterogeneity index is:

$$
H_m = h / h_{het}
$$

$H_m$ lies between 0 and 1.

---

### Betweenness centrality heterogeneity

Let $B_i$ be the betweenness centrality of node $i$.

We define:

$$
σ_B^2 = (1/N) * Σ_i (B_i - B̄)^2
$$

where the mean betweenness is:

$$
\overline{B} = (1/N) * Σ_i B_i
$$

This measures how unevenly shortest-path participation is distributed.

---

### Clustering heterogeneity

Let $C_i$ be the local clustering coefficient of node $i$.

Clustering heterogeneity is:

$$
σ_C^2 = (1/N) * Σ_i (C_i - \overline{C})^2
$$

where:

$$
\overline{C} = (1/N) * Σ_i C_i
$$

This quantifies variation in triangle density across nodes.

---

## HetergeneityAnalyzer

The file `heterogeneity/analyzer.py` provides the central interface for computing heterogeneity metrics on NetworkX graphs.

It performs the following tasks:

- imports low-level metric functions from the `metrics/` package  
- wraps them in a unified `HeterogeneityAnalyzer` class  
- organizes metrics into three functional groups:
  - **degree-based metrics**
  - **centrality-based metrics**
  - **clustering-based metrics**

### Methods Overview

The class exposes the following methods:

- `degree_metrics()`  
  Computes:
  - Degree variance  
  - Degree distribution heterogeneity  

- `centrality_metrics()`  
  Computes:
  - Betweenness centrality heterogeneity  

- `clustering_metrics()`  
  Computes:
  - Clustering heterogeneity  

- `all_metrics()`  
  Calls all the above groups and returns a single merged dictionary:

```python
{
  "Degree variance": ...,
  "Degree distribution variance": ...,
  "Betweenness centrality heterogeneity": ...,
  "Clustering heterogeneity": ...
}

```

### Usage Example
```python
import networkx as nx
from heterogeneity.analyzer import HeterogeneityAnalyzer

# Build a graph
G = nx.erdos_renyi_graph(50, 0.1)

# Create analyzer
analyzer = HeterogeneityAnalyzer(G)

# Compute metrics
metrics = analyzer.all_metrics()

# Display results
for name, value in metrics.items():
    print(f"{name}: {value:.6f}")
```