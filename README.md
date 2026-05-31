# lau-ricci-flow-agents

Networks have curvature. Not the kind you see — the kind you measure by how efficiently information flows between neighbors. Ollivier-Ricci curvature quantifies this: positive curvature means neighbors are already close, negative means they're far apart. Run Ricci flow to smooth the curvature, and communities emerge like islands from the sea.

## The math in 60 seconds

**Ollivier-Ricci curvature** κ(i,j) = 1 - W₁(μᵢ, μⱼ) where W₁ is the Wasserstein-1 distance between the neighbor distributions of nodes i and j. Intuitively: κ > 0 means "neighbors overlap," κ < 0 means "neighbors diverge."

Key results:

- **Ricci flow:** evolve edge weights by dω/dt = -κ·ω — curvature smooths out
- **Community detection:** after Ricci flow, within-community edges strengthen, between-community edges weaken
- **Cheeger inequality:** h(G) ≤ √(2λ₁) where λ₁ is the spectral gap — curvature connects to cuts
- **Modularity + silhouette:** community quality metrics for validation
- **Spectral analysis:** normalized Laplacian eigenvalues encode cluster structure

References: Ollivier, *Ricci curvature of Markov chains on metric spaces* (2009); Ni et al., *Community Detection on Networks with Ricci Flow* (2019)

## Quick start

```rust
use lau_ricci_flow_agents::{GraphMetric, RicciFlow, CurvatureCommunity};

// Build a graph with known community structure
let graph = GraphMetric::stochastic_block_model(3, 20, 0.8, 0.1);

// Compute Ollivier-Ricci curvature for all edges
let curvatures = graph.ollivier_ricci_curvature();

// Run Ricci flow for 10 iterations
let evolved = RicciFlow::run(&graph, 10, 0.5);

// Detect communities
let communities = CurvatureCommunity::detect(&evolved);
let modularity = communities.modularity();

// Verify Cheeger inequality
let cheeger = evolved.cheeger_constant();
let spectral_gap = evolved.spectral_gap();
assert!(cheeger <= (2.0 * spectral_gap).sqrt() + 1e-6);
```

## Key types

| Type | What it is |
|------|-----------|
| `GraphMetric` | Weighted graph with shortest-path metric |
| `OllivierRicci` | Curvature κ(i,j) via optimal transport between neighborhoods |
| `FormanRicci` | Combinatorial curvature (faster to compute, coarser) |
| `RicciFlow` | Curvature-driven edge weight evolution |
| `CurvatureCommunity` | Community detection from evolved graph |
| `SpectralAnalysis` | Laplacian eigenvalues, spectral gap, Cheeger constant |

## Contributing

[Open an issue](https://github.com/SuperInstance/lau-ricci-flow-agents/issues) or PR. We'd love:

- Directed graph support (asymmetric curvature)
- Dynamic graphs (streaming curvature updates)
- GPU-accelerated optimal transport
