//! Ricci flow on agent similarity graphs.
//!
//! Implements Ollivier-Ricci and Forman discrete curvature, Ricci flow evolution,
//! community detection, and spectral analysis on weighted graphs.

#![allow(clippy::needless_range_loop)]

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// DenseMatrix
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl DenseMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![vec![0.0; cols]; rows] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n { m.data[i][i] = 1.0; }
        m
    }

    pub fn get(&self, i: usize, j: usize) -> f64 { self.data[i][j] }
    pub fn set(&mut self, i: usize, j: usize, v: f64) { self.data[i][j] = v; }

    pub fn multiply(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.cols, other.rows);
        let mut r = DenseMatrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut s = 0.0;
                for k in 0..self.cols { s += self.data[i][k] * other.data[k][j]; }
                r.data[i][j] = s;
            }
        }
        r
    }

    pub fn transpose(&self) -> DenseMatrix {
        let mut r = DenseMatrix::zeros(self.cols, self.rows);
        for i in 0..self.rows { for j in 0..self.cols { r.data[j][i] = self.data[i][j]; } }
        r
    }

    pub fn top_k_eigenvalues(&self, k: usize, iterations: usize) -> Vec<(f64, Vec<f64>)> {
        let n = self.rows;
        let mut results = Vec::new();
        let mut deflated = self.clone();
        for _ in 0..k.min(n) {
            let mut v: Vec<f64> = (0..n).map(|i| (i as f64 * 1.618 + 0.5).sin()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            for x in &mut v { *x /= norm; }
            for _ in 0..iterations {
                let mut nv = vec![0.0; n];
                for i in 0..n { for j in 0..n { nv[i] += deflated.data[i][j] * v[j]; } }
                let norm = nv.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
                for x in &mut nv { *x /= norm; }
                v = nv;
            }
            let mut lambda = 0.0;
            for i in 0..n { for j in 0..n { lambda += v[i] * deflated.data[i][j] * v[j]; } }
            for i in 0..n { for j in 0..n { deflated.data[i][j] -= lambda * v[i] * v[j]; } }
            results.push((lambda, v));
        }
        results
    }

    pub fn eigenvalues_small(&self) -> Vec<f64> {
        let n = self.rows;
        match n {
            0 => vec![],
            1 => vec![self.data[0][0]],
            2 => {
                let (a, b, c, d) = (self.data[0][0], self.data[0][1], self.data[1][0], self.data[1][1]);
                let tr = a + d;
                let disc = (tr * tr - 4.0 * (a * d - b * c)).max(0.0).sqrt();
                vec![(tr + disc) / 2.0, (tr - disc) / 2.0]
            }
            _ => {
                let mut eigs: Vec<f64> = self.top_k_eigenvalues(n, 200).into_iter().map(|(l, _)| l).collect();
                eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());
                eigs
            }
        }
    }

    pub fn trace(&self) -> f64 { (0..self.rows.min(self.cols)).map(|i| self.data[i][i]).sum() }

    pub fn determinant(&self) -> f64 {
        let n = self.rows;
        assert_eq!(n, self.cols);
        match n {
            0 => 1.0,
            1 => self.data[0][0],
            2 => self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0],
            _ => {
                let mut det = 0.0;
                for j in 0..n {
                    let minor = {
                        let mut m = DenseMatrix::zeros(n - 1, n - 1);
                        let mut ri = 0;
                        for i in 0..n {
                            if i == 0 { continue; }
                            let mut ci = 0;
                            for jj in 0..n {
                                if jj == j { continue; }
                                m.data[ri][ci] = self.data[i][jj];
                                ci += 1;
                            }
                            ri += 1;
                        }
                        m
                    };
                    det += if j % 2 == 0 { 1.0 } else { -1.0 } * self.data[0][j] * minor.determinant();
                }
                det
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GraphMetric
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetric {
    pub n: usize,
    pub adjacency: Vec<Vec<(usize, f64)>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OrdF(u64);
impl OrdF { fn from_f64(f: f64) -> Self { OrdF(f.to_bits()) } }
impl PartialOrd for OrdF { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }
impl Ord for OrdF { fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.0.cmp(&other.0) } }

impl GraphMetric {
    pub fn new(n: usize) -> Self { Self { n, adjacency: vec![vec![]; n] } }

    pub fn add_edge(&mut self, i: usize, j: usize, w: f64) {
        self.adjacency[i].push((j, w));
        self.adjacency[j].push((i, w));
    }

    pub fn distance(&self, i: usize, j: usize) -> f64 {
        if i == j { return 0.0; }
        let mut dist = vec![f64::INFINITY; self.n];
        dist[i] = 0.0;
        let mut heap = BinaryHeap::<std::cmp::Reverse<(OrdF, usize)>>::new();
        heap.push(std::cmp::Reverse((OrdF::from_f64(0.0), i)));
        while let Some(std::cmp::Reverse((d, u))) = heap.pop() {
            if u == j { return f64::from_bits(d.0); }
            let du = f64::from_bits(d.0);
            if du > dist[u] { continue; }
            for &(v, w) in &self.adjacency[u] {
                let nd = du + w;
                if nd < dist[v] { dist[v] = nd; heap.push(std::cmp::Reverse((OrdF::from_f64(nd), v))); }
            }
        }
        dist[j]
    }

    pub fn degree(&self, i: usize) -> usize { self.adjacency[i].len() }
    pub fn volume(&self, i: usize) -> f64 { self.adjacency[i].iter().map(|&(_, w)| w).sum() }

    pub fn diameter(&self) -> f64 {
        let mut max_d: f64 = 0.0;
        for i in 0..self.n { for j in (i+1)..self.n { let d = self.distance(i, j); if d.is_finite() { max_d = max_d.max(d); } } }
        max_d
    }

    pub fn all_distances(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut dist = vec![vec![f64::INFINITY; n]; n];
        for i in 0..n { dist[i][i] = 0.0; for &(j, w) in &self.adjacency[i] { dist[i][j] = w; } }
        for k in 0..n { for i in 0..n { for j in 0..n { let d = dist[i][k] + dist[k][j]; if d < dist[i][j] { dist[i][j] = d; } } } }
        dist
    }

    pub fn laplacian(&self) -> DenseMatrix {
        let n = self.n;
        let mut l = DenseMatrix::zeros(n, n);
        for i in 0..n {
            l.set(i, i, self.volume(i));
            for &(j, w) in &self.adjacency[i] { l.set(i, j, l.get(i, j) - w); }
        }
        l
    }

    pub fn normalized_laplacian(&self) -> DenseMatrix {
        let n = self.n;
        let lap = self.laplacian();
        let dsi: Vec<f64> = (0..n).map(|i| { let d = self.volume(i); if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 } }).collect();
        let mut nl = DenseMatrix::zeros(n, n);
        for i in 0..n { for j in 0..n { nl.set(i, j, lap.get(i, j) * dsi[i] * dsi[j]); } }
        nl
    }

    pub fn edges(&self) -> Vec<(usize, usize, f64)> {
        let mut e = Vec::new();
        for i in 0..self.n { for &(j, w) in &self.adjacency[i] { if i < j { e.push((i, j, w)); } } }
        e
    }

    pub fn total_volume(&self) -> f64 { self.edges().iter().map(|&(_, _, w)| w).sum() }

    pub fn triangles_on_edge(&self, i: usize, j: usize) -> usize {
        let ni: Vec<usize> = self.adjacency[i].iter().map(|&(v, _)| v).collect();
        let nj: Vec<usize> = self.adjacency[j].iter().map(|&(v, _)| v).collect();
        ni.iter().filter(|x| nj.contains(x) && **x != i && **x != j).count()
    }
}

// ---------------------------------------------------------------------------
// Optimal Transport
// ---------------------------------------------------------------------------

/// Solve the discrete optimal transport problem using min-cost flow with Bellman-Ford.
fn optimal_transport(supply: &[f64], demand: &[f64], cost: &[Vec<f64>]) -> f64 {
    let ns = supply.len();
    let nd = demand.len();
    if ns == 0 || nd == 0 { return 0.0; }

    // Nodes: 0 = source, 1..ns = supply, ns+1..ns+nd+1 = demand, ns+nd+1 = sink
    let source = 0;
    let sink = ns + nd + 1;
    let n_nodes = ns + nd + 2;

    // Edge list: (from, to, capacity, cost, flow)
    let mut edges: Vec<(usize, usize, f64, f64, f64)> = Vec::new();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];

    let add_edge = |u: usize, v: usize, cap: f64, c: f64, edges: &mut Vec<_>, adj: &mut Vec<Vec<usize>>| {
        let idx = edges.len();
        edges.push((u, v, cap, c, 0.0));
        adj[u].push(idx);
        adj[v].push(idx + 1);
        edges.push((v, u, 0.0, -c, 0.0)); // reverse edge
    };

    // Source to supply nodes
    for i in 0..ns {
        add_edge(source, 1 + i, supply[i], 0.0, &mut edges, &mut adj);
    }
    // Supply to demand nodes
    for i in 0..ns {
        for j in 0..nd {
            add_edge(1 + i, 1 + ns + j, supply[i].min(demand[j]), cost[i][j], &mut edges, &mut adj);
        }
    }
    // Demand to sink
    for j in 0..nd {
        add_edge(1 + ns + j, sink, demand[j], 0.0, &mut edges, &mut adj);
    }

    let mut total_cost = 0.0;

    loop {
        // Bellman-Ford to find shortest path from source to sink
        let mut dist = vec![f64::INFINITY; n_nodes];
        let mut parent_edge = vec![None; n_nodes];
        dist[source] = 0.0;

        for _ in 0..n_nodes - 1 {
            let mut updated = false;
            for (idx, &(u, v, _cap, c, flow)) in edges.iter().enumerate() {
                let residual = if idx % 2 == 0 { _cap - flow } else { flow };
                if residual > 1e-15 && dist[u].is_finite() {
                    let nd = dist[u] + c;
                    if nd < dist[v] - 1e-15 {
                        dist[v] = nd;
                        parent_edge[v] = Some(idx);
                        updated = true;
                    }
                }
            }
            if !updated { break; }
        }

        if dist[sink].is_infinite() { break; } // no more augmenting paths

        // Find bottleneck (minimum residual capacity along path)
        let mut bottleneck = f64::INFINITY;
        let mut v = sink;
        while v != source {
            let idx = parent_edge[v].unwrap();
            let (u, _, cap, _, flow) = edges[idx];
            let residual = if idx % 2 == 0 { cap - flow } else { flow };
            bottleneck = bottleneck.min(residual);
            v = u;
        }

        // Augment flow
        v = sink;
        while v != source {
            let idx = parent_edge[v].unwrap();
            let (u, _, _cap, c, _) = edges[idx];
            if idx % 2 == 0 {
                edges[idx].4 += bottleneck;
                edges[idx + 1].4 += bottleneck; // reverse
            } else {
                edges[idx].4 -= bottleneck;
                edges[idx - 1].4 -= bottleneck; // forward
            }
            total_cost += bottleneck * c;
            v = u;
        }
    }

    total_cost
}

// ---------------------------------------------------------------------------
// OllivierRicciCurvature
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllivierRicciCurvature {
    pub graph: GraphMetric,
    pub alpha: f64,
}

impl OllivierRicciCurvature {
    pub fn new(graph: GraphMetric, alpha: f64) -> Self { Self { graph, alpha } }

    pub fn neighbor_distribution(&self, i: usize) -> Vec<(usize, f64)> {
        let deg = self.graph.degree(i);
        if deg == 0 { return vec![]; }
        let vol = self.graph.volume(i);
        if vol == 0.0 { return vec![]; }
        let mut dist = vec![(i, self.alpha)];
        for &(j, w) in &self.graph.adjacency[i] { dist.push((j, (1.0 - self.alpha) * w / vol)); }
        dist
    }

    pub fn wasserstein_1(&self, mu: &[(usize, f64)], nu: &[(usize, f64)]) -> f64 {
        if mu.is_empty() || nu.is_empty() { return 0.0; }
        // Filter out zero-mass entries to avoid degeneracy
        let mu: Vec<(usize, f64)> = mu.iter().filter(|&&(_, m)| m > 1e-15).cloned().collect();
        let nu: Vec<(usize, f64)> = nu.iter().filter(|&&(_, m)| m > 1e-15).cloned().collect();
        if mu.is_empty() || nu.is_empty() { return 0.0; }
        let ns = mu.len();
        let nd = nu.len();
        let mut cost = vec![vec![0.0f64; nd]; ns];
        for i in 0..ns { for j in 0..nd { cost[i][j] = self.graph.distance(mu[i].0, nu[j].0); } }
        let supply: Vec<f64> = mu.iter().map(|&(_, m)| m).collect();
        let demand: Vec<f64> = nu.iter().map(|&(_, m)| m).collect();
        optimal_transport(&supply, &demand, &cost)
    }

    pub fn curvature(&self, i: usize, j: usize) -> f64 {
        let dij = self.graph.distance(i, j);
        if dij == 0.0 || !dij.is_finite() { return 0.0; }
        let mu = self.neighbor_distribution(i);
        let nu = self.neighbor_distribution(j);
        let w1 = self.wasserstein_1(&mu, &nu);
        1.0 - w1 / dij
    }

    pub fn average_curvature(&self) -> f64 {
        let e = self.graph.edges();
        if e.is_empty() { return 0.0; }
        e.iter().map(|&(i, j, _)| self.curvature(i, j)).sum::<f64>() / e.len() as f64
    }

    pub fn curvature_distribution(&self) -> Vec<f64> {
        self.graph.edges().iter().map(|&(i, j, _)| self.curvature(i, j)).collect()
    }

    pub fn ricci_flat_edges(&self) -> Vec<(usize, usize)> {
        self.graph.edges().iter().filter(|&&(i, j, _)| self.curvature(i, j).abs() < 1e-6).map(|&(i, j, _)| (i, j)).collect()
    }
    pub fn ricci_positive_edges(&self) -> Vec<(usize, usize)> {
        self.graph.edges().iter().filter(|&&(i, j, _)| self.curvature(i, j) > 1e-6).map(|&(i, j, _)| (i, j)).collect()
    }
    pub fn ricci_negative_edges(&self) -> Vec<(usize, usize)> {
        self.graph.edges().iter().filter(|&&(i, j, _)| self.curvature(i, j) < -1e-6).map(|&(i, j, _)| (i, j)).collect()
    }
}

// ---------------------------------------------------------------------------
// FormRicciCurvature
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormRicciCurvature;

impl FormRicciCurvature {
    pub fn new() -> Self { Self }
    pub fn curvature(&self, graph: &GraphMetric, i: usize, j: usize) -> f64 {
        4.0 - graph.degree(i) as f64 - graph.degree(j) as f64 + 3.0 * graph.triangles_on_edge(i, j) as f64
    }
    pub fn average(&self, graph: &GraphMetric) -> f64 {
        let e = graph.edges();
        if e.is_empty() { return 0.0; }
        e.iter().map(|&(i, j, _)| self.curvature(graph, i, j)).sum::<f64>() / e.len() as f64
    }
    pub fn compare_with_ollivier(&self, other: &OllivierRicciCurvature) -> f64 {
        let edges = other.graph.edges();
        if edges.len() < 2 { return 0.0; }
        let fv: Vec<f64> = edges.iter().map(|&(i, j, _)| self.curvature(&other.graph, i, j)).collect();
        let ov: Vec<f64> = edges.iter().map(|&(i, j, _)| other.curvature(i, j)).collect();
        let n = fv.len() as f64;
        let mf = fv.iter().sum::<f64>() / n;
        let mo = ov.iter().sum::<f64>() / n;
        let cov = fv.iter().zip(&ov).map(|(f, o)| (f - mf) * (o - mo)).sum::<f64>() / n;
        let sf = (fv.iter().map(|f| (f - mf).powi(2)).sum::<f64>() / n).sqrt();
        let so = (ov.iter().map(|o| (o - mo).powi(2)).sum::<f64>() / n).sqrt();
        if sf < 1e-12 || so < 1e-12 { return 0.0; }
        cov / (sf * so)
    }
}
impl Default for FormRicciCurvature { fn default() -> Self { Self::new() } }

// ---------------------------------------------------------------------------
// GraphSnapshot
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub time: usize,
    pub metric: GraphMetric,
    pub curvatures: Vec<f64>,
}

impl GraphSnapshot {
    pub fn total_volume(&self) -> f64 { self.metric.total_volume() }

    pub fn curvature_variance(&self) -> f64 {
        if self.curvatures.is_empty() { return 0.0; }
        let n = self.curvatures.len() as f64;
        let m = self.curvatures.iter().sum::<f64>() / n;
        self.curvatures.iter().map(|k| (k - m).powi(2)).sum::<f64>() / n
    }

    pub fn modularity(&self, communities: &[Vec<usize>]) -> f64 {
        let m = self.metric.total_volume();
        if m == 0.0 { return 0.0; }
        let mut q = 0.0;
        for c in communities {
            for &i in c {
                for &j in c {
                    let a = self.metric.adjacency[i].iter().find(|&&(v, _)| v == j).map(|&(_, w)| w).unwrap_or(0.0);
                    q += a - self.metric.volume(i) * self.metric.volume(j) / (2.0 * m);
                }
            }
        }
        q / (2.0 * m)
    }
}

// ---------------------------------------------------------------------------
// RicciFlow
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RicciFlow {
    pub graph: GraphMetric,
    pub dt: f64,
}

impl RicciFlow {
    pub fn new(graph: GraphMetric, dt: f64) -> Self { Self { graph, dt } }

    pub fn step(&mut self) {
        let orc = OllivierRicciCurvature::new(self.graph.clone(), 0.5);
        let edges: Vec<(usize, usize)> = self.graph.edges().iter().map(|&(i, j, _)| (i, j)).collect();
        let mut updates = Vec::new();
        for (i, j) in &edges {
            let k = orc.curvature(*i, *j);
            let w = self.graph.adjacency[*i].iter().find(|&&(v, _)| v == *j).map(|&(_, w)| w).unwrap_or(0.0);
            updates.push((*i, *j, (w * (1.0 - self.dt * k)).max(1e-10)));
        }
        for (i, j, w) in &updates {
            for e in &mut self.graph.adjacency[*i] { if e.0 == *j { e.1 = *w; } }
            for e in &mut self.graph.adjacency[*j] { if e.0 == *i { e.1 = *w; } }
        }
    }

    pub fn run(&mut self, steps: usize) -> Vec<GraphSnapshot> {
        let mut snaps = Vec::new();
        for t in 0..steps {
            let orc = OllivierRicciCurvature::new(self.graph.clone(), 0.5);
            snaps.push(GraphSnapshot { time: t, metric: self.graph.clone(), curvatures: orc.curvature_distribution() });
            self.step();
        }
        snaps
    }

    pub fn converged(&self, tolerance: f64) -> bool {
        let orc = OllivierRicciCurvature::new(self.graph.clone(), 0.5);
        let d = orc.curvature_distribution();
        if d.is_empty() { return true; }
        let n = d.len() as f64;
        let m = d.iter().sum::<f64>() / n;
        d.iter().map(|k| (k - m).powi(2)).sum::<f64>() / n < tolerance
    }

    pub fn time_to_converge(&self, tolerance: f64) -> usize {
        let mut rf = self.clone();
        for s in 0..1000 { if rf.converged(tolerance) { return s; } rf.step(); }
        1000
    }

    pub fn normalize(&mut self) {
        let vol = self.graph.total_volume();
        if vol <= 0.0 { return; }
        let scale = self.graph.n as f64 / vol;
        for adj in &mut self.graph.adjacency { for e in adj { e.1 *= scale; } }
    }
}

// ---------------------------------------------------------------------------
// CurvatureCommunity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureCommunity;

impl CurvatureCommunity {
    pub fn new() -> Self { Self }

    pub fn detect_communities(&self, graph: &mut GraphMetric, steps: usize, threshold: f64) -> Vec<Vec<usize>> {
        let mut rf = RicciFlow::new(graph.clone(), 0.01);
        rf.run(steps);
        let evolved = &rf.graph;
        let mut visited = vec![false; graph.n];
        let mut communities = Vec::new();
        for start in 0..graph.n {
            if visited[start] { continue; }
            let mut community = Vec::new();
            let mut stack = vec![start];
            while let Some(node) = stack.pop() {
                if visited[node] { continue; }
                visited[node] = true;
                community.push(node);
                for &(nb, w) in &evolved.adjacency[node] { if !visited[nb] && w >= threshold { stack.push(nb); } }
            }
            communities.push(community);
        }
        communities
    }

    pub fn modularity(&self, communities: &[Vec<usize>], graph: &GraphMetric) -> f64 {
        GraphSnapshot { time: 0, metric: graph.clone(), curvatures: vec![] }.modularity(communities)
    }

    pub fn silhouette_score(&self, communities: &[Vec<usize>], graph: &GraphMetric) -> f64 {
        if communities.len() < 2 { return 0.0; }
        let dist = graph.all_distances();
        let mut nc = vec![0usize; graph.n];
        for (ci, c) in communities.iter().enumerate() { for &node in c { nc[node] = ci; } }
        let mut scores = Vec::new();
        for c in communities {
            if c.len() < 2 { continue; }
            for &i in c {
                let a: f64 = c.iter().filter(|&&j| j != i).map(|&j| dist[i][j]).sum::<f64>() / (c.len() - 1).max(1) as f64;
                let mut min_b = f64::INFINITY;
                for (ci, o) in communities.iter().enumerate() {
                    if ci == nc[i] || o.is_empty() { continue; }
                    min_b = min_b.min(o.iter().map(|&j| dist[i][j]).sum::<f64>() / o.len() as f64);
                }
                if min_b.is_infinite() { continue; }
                let denom = a.max(min_b);
                scores.push(if denom == 0.0 { 0.0 } else { (min_b - a) / denom });
            }
        }
        if scores.is_empty() { 0.0 } else { scores.iter().sum::<f64>() / scores.len() as f64 }
    }
}
impl Default for CurvatureCommunity { fn default() -> Self { Self::new() } }

// ---------------------------------------------------------------------------
// CurvatureSpectrum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureSpectrum {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<Vec<f64>>,
}

impl CurvatureSpectrum {
    pub fn from_graph(graph: &GraphMetric) -> Self {
        let nl = graph.normalized_laplacian();
        let results = nl.top_k_eigenvalues(graph.n, 200);
        let mut eigs: Vec<f64> = results.iter().map(|(l, _)| *l).collect();
        let evecs: Vec<Vec<f64>> = results.into_iter().map(|(_, v)| v).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self { eigenvalues: eigs, eigenvectors: evecs }
    }

    pub fn spectral_gap(&self) -> f64 { self.eigenvalues.iter().find(|&&l| l > 1e-6).copied().unwrap_or(0.0) }
    pub fn cheeger_constant(&self) -> f64 { self.spectral_gap() / 2.0 }
    pub fn is_expander(&self) -> bool { self.spectral_gap() > 0.1 }
}

// ---------------------------------------------------------------------------
// AgentProfile
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    pub id: String,
    pub features: Vec<f64>,
    pub capabilities: Vec<String>,
}

impl AgentProfile {
    pub fn new(id: impl Into<String>, features: Vec<f64>, capabilities: Vec<String>) -> Self {
        Self { id: id.into(), features, capabilities }
    }

    pub fn similarity(&self, other: &AgentProfile) -> f64 {
        if self.features.len() != other.features.len() { return 0.0; }
        let dot: f64 = self.features.iter().zip(&other.features).map(|(a, b)| a * b).sum();
        let na = self.features.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = other.features.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 { 0.0 } else { dot / (na * nb) }
    }
}

// ---------------------------------------------------------------------------
// Community
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    pub members: Vec<usize>,
    pub cohesion: f64,
    pub curvature: f64,
}

impl Community {
    pub fn representative(&self, graph: &GraphMetric) -> usize {
        if self.members.len() == 1 { return self.members[0]; }
        *self.members.iter().min_by(|&&a, &&b| {
            let sa: f64 = self.members.iter().map(|&j| graph.distance(a, j)).sum();
            let sb: f64 = self.members.iter().map(|&j| graph.distance(b, j)).sum();
            sa.partial_cmp(&sb).unwrap()
        }).unwrap()
    }

    pub fn boundary(&self, graph: &GraphMetric) -> Vec<usize> {
        let ms: Vec<bool> = {
            let mut s = vec![false; graph.n];
            for &m in &self.members { if m < graph.n { s[m] = true; } }
            s
        };
        self.members.iter().filter(|&&i| graph.adjacency[i].iter().any(|&(j, _)| j < ms.len() && !ms[j])).copied().collect()
    }
}

// ---------------------------------------------------------------------------
// AgentSimilarityGraph
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSimilarityGraph {
    pub agents: Vec<AgentProfile>,
    pub graph: GraphMetric,
}

impl AgentSimilarityGraph {
    pub fn new(agents: Vec<AgentProfile>) -> Self { let n = agents.len(); Self { agents, graph: GraphMetric::new(n) } }

    pub fn build_from_features(&mut self, threshold: f64) {
        let n = self.agents.len();
        self.graph = GraphMetric::new(n);
        for i in 0..n { for j in (i+1)..n { let s = self.agents[i].similarity(&self.agents[j]); if s > threshold { self.graph.add_edge(i, j, s); } } }
    }

    pub fn evolve_communities(&mut self, steps: usize) -> Vec<Community> {
        let cc = CurvatureCommunity::new();
        cc.detect_communities(&mut self.graph, steps, 1e-6).into_iter().map(|members| {
            let orc = OllivierRicciCurvature::new(self.graph.clone(), 0.5);
            let (mut cs, mut cnt) = (0.0, 0);
            for (ii, &i) in members.iter().enumerate() { for &j in &members[ii+1..] {
                if self.graph.adjacency[i].iter().any(|&(v, _)| v == j) { cs += orc.curvature(i, j); cnt += 1; }
            }}
            Community { members, cohesion: if cnt > 0 { cs / cnt as f64 } else { 0.0 }, curvature: if cnt > 0 { cs / cnt as f64 } else { 0.0 } }
        }).collect()
    }

    pub fn track_agent(&self, agent_id: usize, steps: usize) -> Vec<usize> {
        let mut gc = self.graph.clone();
        let cc = CurvatureCommunity::new();
        let mut assignments = Vec::new();
        for _ in 0..steps {
            let mut g = gc.clone();
            let comms = cc.detect_communities(&mut g, 1, 1e-6);
            let mut found = 0;
            for (ci, c) in comms.iter().enumerate() { if c.contains(&agent_id) { found = ci; break; } }
            assignments.push(found);
            let mut rf = RicciFlow::new(gc.clone(), 0.01);
            rf.step();
            gc = rf.graph;
        }
        assignments
    }
}

// ---------------------------------------------------------------------------
// Graph constructors
// ---------------------------------------------------------------------------

pub fn complete_graph(n: usize) -> GraphMetric {
    let mut g = GraphMetric::new(n);
    for i in 0..n { for j in (i+1)..n { g.add_edge(i, j, 1.0); } }
    g
}

pub fn path_graph(n: usize) -> GraphMetric {
    let mut g = GraphMetric::new(n);
    for i in 0..n.saturating_sub(1) { g.add_edge(i, i + 1, 1.0); }
    g
}

pub fn cycle_graph(n: usize) -> GraphMetric {
    let mut g = GraphMetric::new(n);
    for i in 0..n { g.add_edge(i, (i + 1) % n, 1.0); }
    g
}

pub fn star_graph(n: usize) -> GraphMetric {
    let mut g = GraphMetric::new(n);
    for i in 1..n { g.add_edge(0, i, 1.0); }
    g
}

pub fn binary_tree(depth: usize) -> GraphMetric {
    let n = (1 << (depth + 1)) - 1;
    let mut g = GraphMetric::new(n);
    for i in 0..n {
        if 2 * i + 1 < n { g.add_edge(i, 2 * i + 1, 1.0); }
        if 2 * i + 2 < n { g.add_edge(i, 2 * i + 2, 1.0); }
    }
    g
}

pub fn barbell_graph(m: usize, p: usize) -> GraphMetric {
    let n = 2 * m + p;
    let mut g = GraphMetric::new(n);
    for i in 0..m { for j in (i+1)..m { g.add_edge(i, j, 1.0); } }
    for i in m+p..n { for j in (i+1)..n { g.add_edge(i, j, 1.0); } }
    for i in 0..p.saturating_sub(1) { g.add_edge(m + i, m + i + 1, 1.0); }
    if m > 0 {
        g.add_edge(m - 1, m, 1.0);
        if p > 0 { g.add_edge(m + p - 1, m + p, 1.0); }
    }
    g
}
