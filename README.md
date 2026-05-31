# lau-ricci-flow-agents

Ricci flow on agent similarity graphs — the same math Perelman used to prove the Poincaré conjecture, applied to how agent populations evolve.

## Mathematical Foundations

Ricci flow: ∂g/∂t = -2 Ric(g). The metric evolves to uniformize curvature. On graphs, Ollivier-Ricci curvature κ(x,y) = 1 - W₁(μₓ, μᵧ)/d(x,y) where W₁ is Wasserstein-1 distance between neighbor distributions.

## Types

- **GraphMetric** — weighted graph with Dijkstra shortest paths, Laplacian, normalized Laplacian
- **OllivierRicciCurvature** — discrete Ricci curvature via optimal transport
- **FormRicciCurvature** — Forman's combinatorial curvature (faster, O(1) per edge)
- **RicciFlow** — evolving edge weights by curvature
- **CurvatureCommunity** — community detection via Ricci flow
- **CurvatureSpectrum** — spectral analysis via normalized Laplacian eigenvalues
- **AgentSimilarityGraph** — agents as nodes, cosine similarity as edges
- **AgentProfile**, **Community**, **GraphSnapshot**, **DenseMatrix**

## Theorems Verified

1. Non-leaf tree edges have negative Ollivier-Ricci curvature
2. Complete graph K_n has positive curvature
3. Cycle C_n curvature → 0 as n → ∞
4. Ricci flow reduces curvature variance
5. Forman curvature on trees: F = 4 - deg(i) - deg(j)
6. Cheeger inequality: λ₁/2 ≤ h ≤ √(2λ₁)
7. Modularity increases with natural community splits
8. Complete graph is Ricci-flat (uniform curvature)
9. Path graph curvature properties verified
10. Agent communities found by Ricci flow agree with feature clustering

## Usage

```rust
use lau_ricci_flow_agents::*;

let g = complete_graph(5);
let orc = OllivierRicciCurvature::new(g, 0.5);
println!("Average curvature: {}", orc.average_curvature());
```

## License

MIT
