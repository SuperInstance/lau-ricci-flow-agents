        let dij = self.graph.distance(i, j);
        if dij == 0.0 || !dij.is_finite() {
            return 0.0;
        }
        let mu = self.neighbor_distribution(i);
        let nu = self.neighbor_distribution(j);
        let w1 = self.wasserstein_1(&mu, &nu);
        1.0 - w1 / dij
    }

    /// Mean curvature over all edges.
    pub fn average_curvature(&self) -> f64 {
        let edges = self.graph.edges();
        if edges.is_empty() {
            return 0.0;
        }
        let total: f64 = edges.iter().map(|&(i, j, _)| self.curvature(i, j)).sum();
        total / edges.len() as f64
    }

    /// All edge curvatures.
    pub fn curvature_distribution(&self) -> Vec<f64> {
        self.graph
            .edges()
            .iter()
            .map(|&(i, j, _)| self.curvature(i, j))
            .collect()
    }

    pub fn ricci_flat_edges(&self) -> Vec<(usize, usize)> {
        self.graph
            .edges()
            .iter()
            .filter(|&&(i, j, _)| {
                let k = self.curvature(i, j);
                (k).abs() < 1e-6
            })
            .map(|&(i, j, _)| (i, j))
            .collect()
    }

    pub fn ricci_positive_edges(&self) -> Vec<(usize, usize)> {
        self.graph
            .edges()
            .iter()
            .filter(|&&(i, j, _)| self.curvature(i, j) > 1e-6)
            .map(|&(i, j, _)| (i, j))
            .collect()
    }

    pub fn ricci_negative_edges(&self) -> Vec<(usize, usize)> {
        self.graph
            .edges()
            .iter()
            .filter(|&&(i, j, _)| self.curvature(i, j) < -1e-6)
            .map(|&(i, j, _)| (i, j))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// FormRicciCurvature
// ---------------------------------------------------------------------------

/// Forman's discrete Ricci curvature (combinatorial, faster than Ollivier).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormRicciCurvature;

impl FormRicciCurvature {
    pub fn new() -> Self {
        Self
    }

    /// Forman curvature for edge (i,j):
    /// F(i,j) = 4 - deg(i) - deg(j) + 3·#triangles(i,j)
    pub fn curvature(&self, graph: &GraphMetric, i: usize, j: usize) -> f64 {
        let triangles = graph.triangles_on_edge(i, j) as f64;
        4.0 - graph.degree(i) as f64 - graph.degree(j) as f64 + 3.0 * triangles
    }

    pub fn average(&self, graph: &GraphMetric) -> f64 {
        let edges = graph.edges();
        if edges.is_empty() {
            return 0.0;
        }
        let total: f64 = edges.iter().map(|&(i, j, _)| self.curvature(graph, i, j)).sum();
        total / edges.len() as f64
    }

    /// Correlation between Forman and Ollivier curvature values.
    pub fn compare_with_ollivier(&self, other: &OllivierRicciCurvature) -> f64 {
        let edges = other.graph.edges();
        if edges.len() < 2 {
            return 0.0;
        }
        let forman: Vec<f64> = edges
            .iter()
            .map(|&(i, j, _)| self.curvature(&other.graph, i, j))
            .collect();
        let ollivier: Vec<f64> = edges
            .iter()
            .map(|&(i, j, _)| other.curvature(i, j))
            .collect();

        let n = forman.len() as f64;
        let mf = forman.iter().sum::<f64>() / n;
        let mo = ollivier.iter().sum::<f64>() / n;

        let cov: f64 = forman
            .iter()
            .zip(&ollivier)
            .map(|(f, o)| (f - mf) * (o - mo))
            .sum::<f64>()
            / n;

        let sf = (forman.iter().map(|f| (f - mf).powi(2)).sum::<f64>() / n).sqrt();
        let so = (ollivier.iter().map(|o| (o - mo).powi(2)).sum::<f64>() / n).sqrt();

        if sf < 1e-12 || so < 1e-12 {
            return 0.0;
        }
        cov / (sf * so)
    }
}

impl Default for FormRicciCurvature {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GraphSnapshot
// ---------------------------------------------------------------------------

/// State of the graph at a time step during Ricci flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub time: usize,
    pub metric: GraphMetric,
    pub curvatures: Vec<f64>,
}

impl GraphSnapshot {
    pub fn total_volume(&self) -> f64 {
        self.metric.total_volume()
    }

    pub fn curvature_variance(&self) -> f64 {
        if self.curvatures.is_empty() {
            return 0.0;
        }
        let n = self.curvatures.len() as f64;
        let mean = self.curvatures.iter().sum::<f64>() / n;
        let var = self.curvatures.iter().map(|k| (k - mean).powi(2)).sum::<f64>() / n;
        var
    }

    /// Newman-Girvan modularity for a given partition.
    pub fn modularity(&self, communities: &[Vec<usize>]) -> f64 {
        let m = self.metric.total_volume();
        if m == 0.0 {
            return 0.0;
        }
        let mut q = 0.0;
        for community in communities {
            for &i in community {
                for &j in community {
                    let a_ij = self
                        .metric
                        .adjacency[i]
                        .iter()
                        .find(|&&(v, _)| v == j)
                        .map(|&(_, w)| w)
                        .unwrap_or(0.0);
                    let ki = self.metric.volume(i);
                    let kj = self.metric.volume(j);
                    q += a_ij - ki * kj / (2.0 * m);
                }
            }
        }
        q / (2.0 * m)
    }
}

// ---------------------------------------------------------------------------
// RicciFlow
// ---------------------------------------------------------------------------

/// Ricci flow: evolves edge weights by curvature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RicciFlow {
    pub graph: GraphMetric,
    pub dt: f64,
}

impl RicciFlow {
    pub fn new(graph: GraphMetric, dt: f64) -> Self {
        Self { graph, dt }
    }

    /// One step: w_ij *= (1 - dt * κ_ij).
    pub fn step(&mut self) {
        let orc = OllivierRicciCurvature::new(self.graph.clone(), 0.5);
        let edges: Vec<(usize, usize)> = self
            .graph
            .edges()
            .iter()
            .map(|&(i, j, _)| (i, j))
            .collect();

        let mut updates = Vec::new();
        for (i, j) in &edges {
            let kappa = orc.curvature(*i, *j);
            let w = self
                .graph
                .adjacency[*i]
                .iter()
                .find(|&&(v, _)| v == *j)
                .map(|&(_, w)| w)
                .unwrap_or(0.0);
            let new_w = (w * (1.0 - self.dt * kappa)).max(1e-10);
            updates.push((*i, *j, new_w));
        }

        for (i, j, w) in &updates {
            for entry in &mut self.graph.adjacency[*i] {
                if entry.0 == *j {
                    entry.1 = *w;
                }
            }
            for entry in &mut self.graph.adjacency[*j] {
                if entry.0 == *i {
                    entry.1 = *w;
                }
            }
        }
    }

    /// Run multiple steps, recording snapshots.
    pub fn run(&mut self, steps: usize) -> Vec<GraphSnapshot> {
        let mut snapshots = Vec::new();
        for t in 0..steps {
            let orc = OllivierRicciCurvature::new(self.graph.clone(), 0.5);
            let curvatures = orc.curvature_distribution();
            snapshots.push(GraphSnapshot {
                time: t,
                metric: self.graph.clone(),
                curvatures,
            });
            self.step();
        }
        snapshots
    }

    /// Check if curvature variance is below tolerance.
    pub fn converged(&self, tolerance: f64) -> bool {
        let orc = OllivierRicciCurvature::new(self.graph.clone(), 0.5);
        let dist = orc.curvature_distribution();
        if dist.is_empty() {
            return true;
        }
        let n = dist.len() as f64;
        let mean = dist.iter().sum::<f64>() / n;
        let var = dist.iter().map(|k| (k - mean).powi(2)).sum::<f64>() / n;
        var < tolerance
    }

    /// Estimate steps to convergence.
    pub fn time_to_converge(&self, tolerance: f64) -> usize {
        let mut rf = self.clone();
        let max_steps = 1000;
        for s in 0..max_steps {
            if rf.converged(tolerance) {
                return s;
            }
            rf.step();
        }
        max_steps
    }

    /// Normalize to keep total volume constant.
    pub fn normalize(&mut self) {
        let vol = self.graph.total_volume();
        if vol <= 0.0 {
            return;
        }
        let target = self.graph.n as f64; // target volume = n
        let scale = target / vol;
        for adj in &mut self.graph.adjacency {
            for entry in adj {
                entry.1 *= scale;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CurvatureCommunity
// ---------------------------------------------------------------------------

/// Community detection via Ricci flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureCommunity;

impl CurvatureCommunity {
    pub fn new() -> Self {
        Self
    }

    /// Detect communities: run Ricci flow, cut edges below weight threshold.
    pub fn detect_communities(
        &self,
        graph: &mut GraphMetric,
        steps: usize,
        threshold: f64,
    ) -> Vec<Vec<usize>> {
        let mut rf = RicciFlow::new(graph.clone(), 0.01);
        rf.run(steps);

        // Build a modified graph with edges below threshold removed
        let mut visited = vec![false; graph.n];
        let mut communities = Vec::new();

        // Use the evolved graph from RicciFlow
        let evolved = &rf.graph;

        for start in 0..graph.n {
            if visited[start] {
                continue;
            }
            let mut community = Vec::new();
            let mut stack = vec![start];
            while let Some(node) = stack.pop() {
                if visited[node] {
                    continue;
                }
                visited[node] = true;
                community.push(node);
                for &(neighbor, w) in &evolved.adjacency[node] {
                    if !visited[neighbor] && w >= threshold {
                        stack.push(neighbor);
                    }
                }
            }
            communities.push(community);
        }
        communities
    }

    /// Newman-Girvan modularity.
    pub fn modularity(&self, communities: &[Vec<usize>], graph: &GraphMetric) -> f64 {
        let snap = GraphSnapshot {
            time: 0,
            metric: graph.clone(),
            curvatures: vec![],
        };
        snap.modularity(communities)
    }

    /// Silhouette score for communities based on graph distances.
    pub fn silhouette_score(&self, communities: &[Vec<usize>], graph: &GraphMetric) -> f64 {
        if communities.len() < 2 {
            return 0.0;
        }
        let dist = graph.all_distances();
        let mut node_community = vec![0usize; graph.n];
        for (ci, community) in communities.iter().enumerate() {
            for &node in community {
                node_community[node] = ci;
            }
        }

        let mut scores = Vec::new();
        for community in communities {
            if community.len() < 2 {
                continue;
            }
            for &i in community {
                // Mean distance to own community
                let a: f64 = community
                    .iter()
                    .filter(|&&j| j != i)
                    .map(|&j| dist[i][j])
                    .sum::<f64>()
                    / (community.len() - 1).max(1) as f64;

                // Min mean distance to other communities
                let mut min_b = f64::INFINITY;
                for (ci, other) in communities.iter().enumerate() {
                    if ci == node_community[i] || other.is_empty() {
                        continue;
                    }
                    let b: f64 = other.iter().map(|&j| dist[i][j]).sum::<f64>() / other.len() as f64;
                    min_b = min_b.min(b);
                }

                if min_b.is_infinite() {
                    continue;
                }

                let denom = a.max(min_b);
                if denom == 0.0 {
                    scores.push(0.0);
                } else {
                    scores.push((min_b - a) / denom);
                }
            }
        }
        if scores.is_empty() {
            return 0.0;
        }
        scores.iter().sum::<f64>() / scores.len() as f64
    }
}

impl Default for CurvatureCommunity {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CurvatureSpectrum
// ---------------------------------------------------------------------------

/// Eigenvalues of the curvature-related operator (normalized Laplacian).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureSpectrum {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<Vec<f64>>,
}

impl CurvatureSpectrum {
    pub fn from_graph(graph: &GraphMetric) -> Self {
        let nl = graph.normalized_laplacian();
        let results = nl.top_k_eigenvalues(graph.n, 200);
        let mut eigenvalues: Vec<f64> = results.iter().map(|(l, _)| *l).collect();
        let eigenvectors: Vec<Vec<f64>> = results.into_iter().map(|(_, v)| v).collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self {
            eigenvalues,
            eigenvectors,
        }
    }

    /// Spectral gap: smallest nonzero eigenvalue.
    pub fn spectral_gap(&self) -> f64 {
        self.eigenvalues
            .iter()
            .find(|&&l| l > 1e-6)
            .copied()
            .unwrap_or(0.0)
    }

    /// Cheeger constant estimate: h ≥ λ₁/2.
    pub fn cheeger_constant(&self) -> f64 {
        let lambda1 = self.spectral_gap();
        lambda1 / 2.0
    }

    /// Whether graph is an expander (spectral gap bounded away from 0).
    pub fn is_expander(&self) -> bool {
        self.spectral_gap() > 0.1
    }
}

// ---------------------------------------------------------------------------
// AgentProfile
// ---------------------------------------------------------------------------

/// Profile of an agent with numeric features and capability tags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    pub id: String,
    pub features: Vec<f64>,
    pub capabilities: Vec<String>,
}

impl AgentProfile {
    pub fn new(id: impl Into<String>, features: Vec<f64>, capabilities: Vec<String>) -> Self {
        Self {
            id: id.into(),
            features,
            capabilities,
        }
    }

    /// Cosine similarity between two agent feature vectors.
    pub fn similarity(&self, other: &AgentProfile) -> f64 {
        if self.features.len() != other.features.len() {
            return 0.0;
        }
        let dot: f64 = self
            .features
            .iter()
            .zip(&other.features)
            .map(|(a, b)| a * b)
            .sum();
        let na = self.features.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = other.features.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 {
            return 0.0;
        }
        dot / (na * nb)
    }
}

// ---------------------------------------------------------------------------
// Community
// ---------------------------------------------------------------------------

/// A community of agents detected by Ricci flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    pub members: Vec<usize>,
    pub cohesion: f64,
    pub curvature: f64,
}

impl Community {
    /// Most central node (minimum sum of distances to other members).
    pub fn representative(&self, graph: &GraphMetric) -> usize {
        if self.members.len() == 1 {
            return self.members[0];
        }
        let mut best = self.members[0];
        let mut best_sum = f64::INFINITY;
        for &i in &self.members {
            let sum: f64 = self
                .members
                .iter()
                .map(|&j| graph.distance(i, j))
                .sum();
            if sum < best_sum {
                best_sum = sum;
                best = i;
            }
        }
        best
    }

    /// Nodes with at least one edge to outside the community.
    pub fn boundary(&self, graph: &GraphMetric) -> Vec<usize> {
        let member_set: Vec<bool> = {
            let mut s = vec![false; graph.n];
            for &m in &self.members {
                if m < graph.n {
                    s[m] = true;
                }
            }
            s
        };
        self.members
            .iter()
            .filter(|&&i| {
                graph.adjacency[i]
                    .iter()
                    .any(|&(j, _)| j < member_set.len() && !member_set[j])
            })
            .copied()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// AgentSimilarityGraph
// ---------------------------------------------------------------------------

/// Agents as nodes, similarity as edges. Ricci flow reveals community structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSimilarityGraph {
    pub agents: Vec<AgentProfile>,
    pub graph: GraphMetric,
}

impl AgentSimilarityGraph {
    pub fn new(agents: Vec<AgentProfile>) -> Self {
        let n = agents.len();
        Self {
            agents,
            graph: GraphMetric::new(n),
        }
    }

    /// Build graph from agent features: connect agents with similarity > threshold.
    pub fn build_from_features(&mut self, threshold: f64) {
        let n = self.agents.len();
        self.graph = GraphMetric::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = self.agents[i].similarity(&self.agents[j]);
                if sim > threshold {
                    self.graph.add_edge(i, j, sim);
                }
            }
        }
    }

    /// Evolve communities using Ricci flow.
    pub fn evolve_communities(&mut self, steps: usize) -> Vec<Community> {
        let cc = CurvatureCommunity::new();
        let communities = cc.detect_communities(&mut self.graph, steps, 1e-6);

        communities
            .into_iter()
            .map(|members| {
                let orc = OllivierRicciCurvature::new(self.graph.clone(), 0.5);
                // Average curvature of intra-community edges
                let mut curv_sum = 0.0;
                let mut count = 0;
                for (idx_i, &i) in members.iter().enumerate() {
                    for &j in &members[idx_i + 1..] {
                        if self.graph.adjacency[i].iter().any(|&(v, _)| v == j) {
                            curv_sum += orc.curvature(i, j);
                            count += 1;
                        }
                    }
                }
                Community {
                    members,
                    cohesion: if count > 0 { curv_sum / count as f64 } else { 0.0 },
                    curvature: if count > 0 { curv_sum / count as f64 } else { 0.0 },
                }
            })
            .collect()
    }

    /// Track which community an agent belongs to over flow steps.
    pub fn track_agent(&self, agent_id: usize, steps: usize) -> Vec<usize> {
        let mut graph_copy = self.graph.clone();
        let cc = CurvatureCommunity::new();
        let mut assignments = Vec::new();

        for t in 0..steps {
            let mut g = graph_copy.clone();
            let communities = cc.detect_communities(&mut g, 1, 1e-6);
            let mut found = 0;
            for (ci, community) in communities.iter().enumerate() {
                if community.contains(&agent_id) {
                    found = ci;
                    break;
                }
            }
            assignments.push(found);

            // Evolve one step
            let mut rf = RicciFlow::new(graph_copy.clone(), 0.01);
            rf.step();
            graph_copy = rf.graph;
            let _ = t;
        }
        assignments
    }
}

// ---------------------------------------------------------------------------
// Graph constructors for testing
// ---------------------------------------------------------------------------

/// Build a complete graph K_n with unit weights.
pub fn complete_graph(n: usize) -> GraphMetric {
    let mut g = GraphMetric::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            g.add_edge(i, j, 1.0);
        }
    }
    g
}

/// Build a path graph P_n.
pub fn path_graph(n: usize) -> GraphMetric {
    let mut g = GraphMetric::new(n);
    for i in 0..n.saturating_sub(1) {
        g.add_edge(i, i + 1, 1.0);
    }
    g
}

/// Build a cycle graph C_n.
pub fn cycle_graph(n: usize) -> GraphMetric {
    let mut g = GraphMetric::new(n);
    for i in 0..n {
        g.add_edge(i, (i + 1) % n, 1.0);
    }
    g
}

/// Build a star graph: center node 0 connected to all others.
pub fn star_graph(n: usize) -> GraphMetric {
    let mut g = GraphMetric::new(n);
    for i in 1..n {
        g.add_edge(0, i, 1.0);
    }
    g
}

/// Build a tree: binary tree with given depth.
pub fn binary_tree(depth: usize) -> GraphMetric {
    let n = (1 << (depth + 1)) - 1;
    let mut g = GraphMetric::new(n);
    for i in 0..n {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        if left < n {
            g.add_edge(i, left, 1.0);
        }
        if right < n {
            g.add_edge(i, right, 1.0);
        }
    }
    g
}

/// Build a barbell graph: two cliques of size m joined by a path of length p.
pub fn barbell_graph(m: usize, p: usize) -> GraphMetric {
    let n = 2 * m + p;
    let mut g = GraphMetric::new(n);
    // Left clique
    for i in 0..m {
        for j in (i + 1)..m {
            g.add_edge(i, j, 1.0);
        }
    }
    // Right clique
    for i in m + p..n {
        for j in (i + 1)..n {
            g.add_edge(i, j, 1.0);
        }
    }
    // Path
    for i in 0..p.saturating_sub(1) {
        g.add_edge(m + i, m + i + 1, 1.0);
    }
    // Connect path to cliques
    if m > 0 {
        g.add_edge(m - 1, m, 1.0);
        if p > 0 {
            g.add_edge(m + p - 1, m + p, 1.0);
        }
    }
    g
}
