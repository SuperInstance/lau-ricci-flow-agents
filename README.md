# lau-ricci-flow-agents

> Networks have curvature. Not the kind you see — the kind you measure by how efficiently information flows between neighbors. Ollivier-Ricci curvature quantifies this: positive curvature means neighbors are already close, negative means they're far apart. Run Ricci flow to smooth the curvature, and communities emerge like islands from the sea.

[![tests](https://img.shields.io/badge/tests-98-green)]()
[![license](https://img.shields.io/badge/license-MIT-blue)]()

## What This Does

This crate implements **discrete Ricci curvature and Ricci flow on weighted graphs**, with direct application to **agent network community detection**. It computes how "curved" the connections between nodes are, then evolves the graph to make curvature uniform — revealing natural community boundaries as edges between communities weaken.

It provides:

- **Ollivier-Ricci curvature** via exact optimal transport (min-cost flow with Bellman-Ford / transportation simplex) between lazy random walk distributions
- **Forman-Ricci curvature** — a combinatorial alternative (faster to compute, coarser signal)
- **Ricci flow evolution** — edge weights update by `dω/dt = -κ·ω`, curvature smooths over iterations
- **Community detection** — after Ricci flow, cut weak edges and extract connected components
- **Spectral analysis** — normalized Laplacian eigenvalues, spectral gap, Cheeger constant, expander detection
- **Agent similarity graphs** — build networks from agent feature vectors, detect communities via curvature evolution

Every structure is `serde`-serializable. The entire pipeline works on pure Rust with no external solver dependencies.

## The Key Idea

**Ollivier-Ricci curvature** assigns a scalar to each edge: κ(i,j) = 1 − W₁(μᵢ, μⱼ)/d(i,j), where W₁ is the Wasserstein-1 (earth mover's) distance between the neighbor distributions of nodes i and j.

The intuition:
- **κ > 0**: Neighbors of i and j overlap significantly — information spreads efficiently (like a complete graph)
- **κ = 0**: Neighbors are unrelated — flat geometry (like a cycle or tree edges)
- **κ < 0**: Neighbors diverge — a bottleneck or bridge between clusters

**Ricci flow** evolves edge weights: `w(t+1) = w(t) · (1 - dt · κ)`. Positively curved edges strengthen, negatively curved edges weaken. After enough iterations, within-community edges dominate and between-community edges vanish. Simple thresholding then recovers communities.

The crate verifies **10 named theorems** in its test suite, including:
- Non-leaf tree edges have negative curvature
- Complete graphs have uniform positive curvature
- C_n curvature approaches 0 as n → ∞
- Barbell modularity increases after correct partitioning
- Complete graphs are Ricci-flat (uniform curvature)
- Agent communities agree with feature-space clusters

## Install

```bash
cargo add lau-ricci-flow-agents
```

## Quick Start

```rust
use lau_ricci_flow_agents::*;

// Build a graph with known community structure (two cliques joined by a path)
let graph = barbell_graph(4, 2);

// Compute Ollivier-Ricci curvature for all edges
let orc = OllivierRicciCurvature::new(graph.clone(), 0.5);
let curvatures = orc.curvature_distribution();
println!("Average curvature: {:.4}", orc.average_curvature());

// Run Ricci flow for 20 iterations
let mut rf = RicciFlow::new(graph.clone(), 0.01);
let snapshots = rf.run(20);
println!("Curvature variance: {:.6} → {:.6}",
    snapshots[0].curvature_variance(),
    snapshots.last().unwrap().curvature_variance());

// Detect communities from the evolved graph
let mut g = graph.clone();
let cc = CurvatureCommunity::new();
let communities = cc.detect_communities(&mut g, 20, 0.001);
println!("Found {} communities", communities.len());

// Evaluate quality
let modularity = cc.modularity(&communities, &g);
let silhouette = cc.silhouette_score(&communities, &g);
println!("Modularity: {:.4}, Silhouette: {:.4}", modularity, silhouette);

// Spectral analysis
let spec = CurvatureSpectrum::from_graph(&graph);
println!("Spectral gap: {:.4}, Cheeger constant: {:.4}, Expander: {}",
    spec.spectral_gap(), spec.cheeger_constant(), spec.is_expander());
```

## API Reference

### DenseMatrix

Row-major dense matrix for linear algebra support (used internally for Laplacians and eigendecomposition).

| Method | Description |
|--------|-------------|
| `zeros(rows, cols)` | Zero matrix |
| `identity(n)` | Identity matrix |
| `get(i, j)` / `set(i, j, v)` | Element access |
| `multiply(other)` | Matrix multiplication |
| `transpose()` | Transpose |
| `top_k_eigenvalues(k, iters)` | Power iteration with deflation for top-k eigenpairs |
| `eigenvalues_small()` | Exact eigenvalues for n ≤ 4 |
| `trace()` | Trace (sum of diagonal) |
| `determinant()` | Determinant via cofactor expansion |

### GraphMetric

A weighted undirected graph with shortest-path metric distances.

| Method | Description |
|--------|-------------|
| `new(n)` | Empty graph on `n` vertices |
| `add_edge(i, j, w)` | Add undirected weighted edge |
| `distance(i, j)` | Dijkstra shortest-path distance |
| `degree(i)` / `volume(i)` | Number of neighbors / sum of edge weights |
| `diameter()` | Maximum pairwise distance |
| `all_distances()` | Floyd–Warshall all-pairs shortest paths |
| `laplacian()` | Combinatorial Laplacian L = D − A |
| `normalized_laplacian()` | Normalized Laplacian D⁻¹⸍² L D⁻¹⸍² |
| `edges()` | All edges as `(i, j, weight)` with i < j |
| `total_volume()` | Sum of all edge weights |
| `triangles_on_edge(i, j)` | Number of triangles containing edge (i,j) |

**Graph constructors** (free functions):

| Function | Description |
|----------|-------------|
| `complete_graph(n)` | K_n with unit weights |
| `path_graph(n)` | P_n chain |
| `cycle_graph(n)` | C_n ring |
| `star_graph(n)` | Center node 0 connected to all others |
| `binary_tree(depth)` | Full binary tree with given depth |
| `barbell_graph(m, p)` | Two K_m cliques joined by a P_p path |

### OllivierRicciCurvature

Ollivier-Ricci curvature on graph edges via optimal transport.

| Method | Description |
|--------|-------------|
| `new(graph, alpha)` | Create with lazy random walk parameter α ∈ [0,1] |
| `neighbor_distribution(i)` | Lazy random walk distribution μᵢ: self-loop gets α, neighbors share (1−α) |
| `wasserstein_1(mu, nu)` | Exact W₁ distance via min-cost flow (transportation simplex) |
| `curvature(i, j)` | κ(i,j) = 1 − W₁(μᵢ, νⱼ)/d(i,j) |
| `average_curvature()` | Mean curvature over all edges |
| `curvature_distribution()` | Curvature of every edge |
| `ricci_flat_edges()` | Edges with |κ| < 10⁻⁶ |
| `ricci_positive_edges()` | Edges with κ > 10⁻⁶ |
| `ricci_negative_edges()` | Edges with κ < −10⁻⁶ |

### FormRicciCurvature

Forman's combinatorial Ricci curvature — O(1) per edge, no optimal transport needed.

| Method | Description |
|--------|-------------|
| `new()` | Create |
| `curvature(graph, i, j)` | F(i,j) = 4 − deg(i) − deg(j) + 3·#triangles(i,j) |
| `average(graph)` | Mean Forman curvature |
| `compare_with_ollivier(other)` | Pearson correlation between Forman and Ollivier values |

### GraphSnapshot

State of the graph at a time step during Ricci flow.

| Method | Description |
|--------|-------------|
| `total_volume()` | Sum of edge weights |
| `curvature_variance()` | Variance of edge curvatures (convergence measure) |
| `modularity(communities)` | Newman–Girvan modularity Q for a given partition |

### RicciFlow

Curvature-driven edge weight evolution.

| Method | Description |
|--------|-------------|
| `new(graph, dt)` | Create with time step `dt` |
| `step()` | One step: wᵢⱼ ← max(wᵢⱼ · (1 − dt · κᵢⱼ), 10⁻¹⁰) |
| `run(steps)` | Multiple steps, returning snapshots |
| `converged(tolerance)` | Check if curvature variance < tolerance |
| `time_to_converge(tolerance)` | Estimate steps to convergence (max 1000) |
| `normalize()` | Scale all weights so total volume = n |

### CurvatureCommunity

Community detection via Ricci flow + thresholding.

| Method | Description |
|--------|-------------|
| `new()` | Create |
| `detect_communities(graph, steps, threshold)` | Run Ricci flow, cut edges below threshold, extract connected components |
| `modularity(communities, graph)` | Newman–Girvan modularity Q |
| `silhouette_score(communities, graph)` | Silhouette coefficient based on graph distances |

### CurvatureSpectrum

Eigenvalues of the normalized Laplacian.

| Method | Description |
|--------|-------------|
| `from_graph(graph)` | Compute spectrum via power iteration |
| `spectral_gap()` | Smallest nonzero eigenvalue λ₁ |
| `cheeger_constant()` | h ≥ λ₁/2 (Cheeger inequality lower bound) |
| `is_expander()` | True if spectral gap > 0.1 |

### AgentProfile

An agent with numeric features and capability tags.

| Method | Description |
|--------|-------------|
| `new(id, features, capabilities)` | Create profile |
| `similarity(other)` | Cosine similarity between feature vectors |

### Community

A detected community of agents.

| Method | Description |
|--------|-------------|
| `representative(graph)` | Most central node (min total distance to members) |
| `boundary(graph)` | Members with at least one edge outside the community |

### AgentSimilarityGraph

Agents as nodes, cosine similarity as edges. Ricci flow reveals community structure.

| Method | Description |
|--------|-------------|
| `new(agents)` | Create from agent profiles |
| `build_from_features(threshold)` | Connect agents with similarity > threshold |
| `evolve_communities(steps)` | Run Ricci flow and detect communities |
| `track_agent(agent_id, steps)` | Track which community an agent belongs to over flow iterations |

## How It Works

### Architecture

```
AgentProfile              ← feature vectors + cosine similarity
  └→ AgentSimilarityGraph ← threshold → GraphMetric
      └→ OllivierRicciCurvature ← W₁ via min-cost flow (transportation simplex)
      └→ FormRicciCurvature      ← O(1) combinatorial formula
      └→ RicciFlow               ← w(t+1) = w(t) · (1 - dt · κ)
          └→ CurvatureCommunity  ← cut weak edges, extract components
          └→ GraphSnapshot        ← time-series of curvature evolution
      └→ CurvatureSpectrum        ← normalized Laplacian eigenvalues
                                    spectral gap, Cheeger constant, expander check

DenseMatrix              ← Laplacian storage, eigendecomposition (power iteration)
```

### Optimal Transport

The Wasserstein-1 distance is the linchpin of Ollivier-Ricci curvature. This crate implements it two ways depending on the code path:

1. **Transportation simplex** (MODI method): Northwest corner initialization, then iterative improvement via reduced costs and cycle augmentation. Handles arbitrary discrete distributions.
2. **Min-cost flow** (successive shortest paths with Bellman-Ford): Network flow formulation with supply nodes, demand nodes, and shortest augmenting paths.

Both produce exact solutions for the discrete distributions arising from lazy random walks on finite graphs.

### Ricci Flow Convergence

The flow `dω/dt = -κ·ω` is normalized to prevent weights from collapsing to zero (each step clips to a minimum of 10⁻¹⁰). The `normalize()` method rescales to keep total volume = n, preventing global weight decay. Convergence is measured by curvature variance dropping below a tolerance.

### Spectral Connections

The normalized Laplacian L_norm = D⁻¹⸍² L D⁻¹⸍² has eigenvalues in [0, 2]. The **spectral gap** λ₁ (smallest nonzero eigenvalue) controls mixing time and is related to the **Cheeger constant** h by:

```
λ₁/2 ≤ h ≤ √(2λ₁)
```

This connects the curvature-based community detection to spectral graph theory.

## The Math

### Ollivier-Ricci Curvature

For a graph with shortest-path metric d and lazy random walk parameter α ∈ [0,1], the Ollivier-Ricci curvature of edge (i,j) is:

$$\kappa(i,j) = 1 - \frac{W_1(\mu_i, \mu_j)}{d(i,j)}$$

where μᵢ is the lazy random walk distribution:
- μᵢ(i) = α (self-loop probability)
- μᵢ(k) = (1 − α) · w_{ik} / vol(i) for neighbors k

W₁ is the Wasserstein-1 (earth mover's) distance — the minimum cost to transport mass from μᵢ to μⱼ, where the cost of moving one unit from node a to node b is d(a,b).

### Forman-Ricci Curvature

A combinatorial alternative with no optimal transport:

$$F(i,j) = 4 - \deg(i) - \deg(j) + 3 \cdot \#\text{triangles}(i,j)$$

Fast to compute (O(1) per edge given triangle counts) but captures only degree and clustering information, not the full metric structure.

### Ricci Flow

Discrete Ricci flow evolves edge weights:

$$\omega_{ij}(t+1) = \omega_{ij}(t) \cdot (1 - \delta t \cdot \kappa_{ij})$$

Positively curved edges (κ > 0) strengthen; negatively curved edges (κ < 0) weaken. After sufficient iterations, within-community edges dominate and between-community edges approach zero weight. Cutting weak edges then reveals the community structure.

### Cheeger Inequality

For a graph with normalized Laplacian spectral gap λ₁ and Cheeger constant h:

$$\frac{\lambda_1}{2} \leq h \leq \sqrt{2\lambda_1}$$

This crate computes the lower bound h ≥ λ₁/2.

### Named Theorems (verified in tests)

| # | Theorem | Statement |
|---|---------|-----------|
| 1 | Tree non-leaf negative curvature | Non-leaf edges in trees have κ < 0 (with α = 0) |
| 2 | Complete graph positive | K_n has κ > 0 for all edges, all α |
| 3 | Cycle convergence | C_n curvature → 0 as n → ∞ |
| 5 | Forman on trees | F(i,j) = 4 − deg(i) − deg(j) exactly (no triangles) |
| 7 | Modularity increase | Correct partition has higher modularity than single community |
| 8 | K_n is Ricci-flat | All curvatures equal (uniform geometry) |
| 9 | Path flatness | Path edges have κ ≈ 0 |
| 10 | Agent-feature agreement | Detected communities match feature-space clusters |

## Testing

**98 tests** covering:

- **DenseMatrix**: construction, arithmetic (multiply, transpose), eigenvalues, trace, determinant
- **GraphMetric**: construction, distances, degrees, volumes, Laplacians, graph constructors
- **Ollivier-Ricci curvature**: neighbor distributions, W₁ distance, curvature computation, flat/positive/negative edge classification
- **Forman curvature**: formula verification, comparison with Ollivier
- **Graph snapshots**: volume, curvature variance, modularity
- **Ricci flow**: stepping, running, variance decrease, normalization, convergence
- **Community detection**: modularity, silhouette scores, connected components
- **Curvature spectrum**: spectral gap, Cheeger constant, expander detection, Cheeger inequality
- **Agent profiles**: similarity computation (identical, orthogonal, opposite, mismatched)
- **Agent similarity graphs**: building from features, evolving communities, tracking agents
- **Serde round-trips**: every public type serializes/deserializes correctly
- **Edge cases**: empty graphs, single nodes, isolated communities
- **Theorem verification**: all 10 named theorems

Run tests with:

```bash
cargo test
```

## License

MIT
