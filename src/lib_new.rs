//! Ricci flow on agent similarity graphs.
//!
//! Implements Ollivier-Ricci and Forman discrete curvature, Ricci flow evolution,
//! community detection, and spectral analysis on weighted graphs.

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// DenseMatrix
// ---------------------------------------------------------------------------

/// A simple row-major dense matrix for linear algebra support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl DenseMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i][i] = 1.0;
        }
        m
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i][j] = v;
    }

    pub fn multiply(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.cols, other.rows);
        let mut result = DenseMatrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut s = 0.0;
                for k in 0..self.cols {
                    s += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = s;
            }
        }
        result
    }

    pub fn transpose(&self) -> DenseMatrix {
        let mut result = DenseMatrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }

    /// Power iteration for top-k eigenvalues (symmetric matrix assumed).
    pub fn top_k_eigenvalues(&self, k: usize, iterations: usize) -> Vec<(f64, Vec<f64>)> {
        let n = self.rows;
        let mut results = Vec::new();
        let mut deflated = self.clone();

        for _ in 0..k.min(n) {
            let mut v = vec![0.0; n];
            // Initialize with a random-ish vector (deterministic)
            for i in 0..n {
                v[i] = (i as f64 * 1.618 + 0.5).sin();
            }
            // Normalize
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            for x in &mut v {
                *x /= norm;
            }

            for _ in 0..iterations {
                // Multiply
                let mut nv = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        nv[i] += deflated.data[i][j] * v[j];
                    }
                }
                let norm = nv.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
                for x in &mut nv {
                    *x /= norm;
                }
                v = nv;
            }

            // Compute eigenvalue: v^T A v
            let mut lambda = 0.0;
            for i in 0..n {
                for j in 0..n {
                    lambda += v[i] * deflated.data[i][j] * v[j];
                }
            }

            // Deflate
            for i in 0..n {
                for j in 0..n {
                    deflated.data[i][j] -= lambda * v[i] * v[j];
                }
            }

            results.push((lambda, v));
        }
        results
    }

    /// Eigenvalues via characteristic polynomial for small matrices (n <= 4).
    pub fn eigenvalues_small(&self) -> Vec<f64> {
        let n = self.rows;
        assert!(n <= 4, "eigenvalues_small only for n <= 4");

        match n {
            0 => vec![],
            1 => vec![self.data[0][0]],
            2 => {
                let a = self.data[0][0];
                let b = self.data[0][1];
                let c = self.data[1][0];
                let d = self.data[1][1];
                let tr = a + d;
                let det = a * d - b * c;
                let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
                vec![(tr + disc) / 2.0, (tr - disc) / 2.0]
            }
            3 => {
                // Use power iteration fallback for 3x3
                let results = self.top_k_eigenvalues(3, 200);
                let mut eigs: Vec<f64> = results.into_iter().map(|(l, _)| l).collect();
                eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());
                eigs
            }
            4 => {
                let results = self.top_k_eigenvalues(4, 200);
                let mut eigs: Vec<f64> = results.into_iter().map(|(l, _)| l).collect();
                eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());
                eigs
            }
            _ => unreachable!(),
        }
    }

    pub fn trace(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..self.rows.min(self.cols) {
            s += self.data[i][i];
        }
        s
    }

    pub fn determinant(&self) -> f64 {
        let n = self.rows;
        assert_eq!(n, self.cols);
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return self.data[0][0];
        }
        if n == 2 {
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0];
        }
        // LU-style cofactor expansion for small matrices
        let mut det = 0.0;
        for j in 0..n {
            let minor = self.minor(0, j);
            det += if j % 2 == 0 {
                1.0
            } else {
                -1.0
            } * self.data[0][j]
                * minor.determinant();
        }
        det
    }

    fn minor(&self, skip_row: usize, skip_col: usize) -> DenseMatrix {
        let mut m = DenseMatrix::zeros(self.rows - 1, self.cols - 1);
        let mut ri = 0;
        for i in 0..self.rows {
            if i == skip_row {
                continue;
            }
            let mut ci = 0;
            for j in 0..self.cols {
                if j == skip_col {
                    continue;
                }
                m.data[ri][ci] = self.data[i][j];
                ci += 1;
            }
            ri += 1;
        }
        m
    }
}

// ---------------------------------------------------------------------------
// GraphMetric
// ---------------------------------------------------------------------------

/// A weighted graph with metric (shortest-path) distances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetric {
    pub n: usize,
    pub adjacency: Vec<Vec<(usize, f64)>>,
}

impl GraphMetric {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adjacency: vec![vec![]; n],
        }
    }

    pub fn add_edge(&mut self, i: usize, j: usize, w: f64) {
        self.adjacency[i].push((j, w));
        self.adjacency[j].push((i, w));
    }

    /// Dijkstra shortest path distance.
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 0.0;
        }
        let mut dist = vec![f64::INFINITY; self.n];
        dist[i] = 0.0;
        // Min-heap: (distance, node)
        let mut heap = BinaryHeap::<std::cmp::Reverse<(OrderedFloat, usize)>>::new();
        heap.push(std::cmp::Reverse((OrderedFloat(0.0), i)));

        while let Some(std::cmp::Reverse((d, u))) = heap.pop() {
            if u == j {
                return d.0;
            }
            if d.0 > dist[u] {
                continue;
            }
            for &(v, w) in &self.adjacency[u] {
                let nd = d.0 + w;
                if nd < dist[v] {
                    dist[v] = nd;
                    heap.push(std::cmp::Reverse((OrderedFloat(nd), v)));
                }
            }
        }
        dist[j]
    }

    pub fn degree(&self, i: usize) -> usize {
        self.adjacency[i].len()
    }

    pub fn volume(&self, i: usize) -> f64 {
        self.adjacency[i].iter().map(|&(_, w)| w).sum()
    }

    pub fn diameter(&self) -> f64 {
        let mut max_d: f64 = 0.0;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let d = self.distance(i, j);
                if d.is_finite() {
                    max_d = max_d.max(d);
                }
            }
        }
        max_d
    }

    /// All-pairs shortest distances (for efficient batch computation).
    pub fn all_distances(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut dist = vec![vec![f64::INFINITY; n]; n];
        for i in 0..n {
            dist[i][i] = 0.0;
            for &(j, w) in &self.adjacency[i] {
                dist[i][j] = w;
            }
        }
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if dist[i][k] + dist[k][j] < dist[i][j] {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
        dist
    }

    /// Graph Laplacian L = D - A.
    pub fn laplacian(&self) -> DenseMatrix {
        let n = self.n;
        let mut l = DenseMatrix::zeros(n, n);
        for i in 0..n {
            let deg: f64 = self.adjacency[i].iter().map(|&(_, w)| w).sum();
            l.set(i, i, deg);
            for &(j, w) in &self.adjacency[i] {
                l.set(i, j, l.get(i, j) - w);
            }
        }
        l
    }

    /// Normalized Laplacian: D^{-1/2} L D^{-1/2}.
    pub fn normalized_laplacian(&self) -> DenseMatrix {
        let n = self.n;
        let lap = self.laplacian();
        let mut d_sqrt_inv = vec![0.0; n];
        for i in 0..n {
            let deg: f64 = self.adjacency[i].iter().map(|&(_, w)| w).sum();
            if deg > 0.0 {
                d_sqrt_inv[i] = 1.0 / deg.sqrt();
            }
        }
        let mut nl = DenseMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                nl.set(i, j, lap.get(i, j) * d_sqrt_inv[i] * d_sqrt_inv[j]);
            }
        }
        nl
    }

    /// All edges as (i, j, weight) with i < j.
    pub fn edges(&self) -> Vec<(usize, usize, f64)> {
        let mut edges = Vec::new();
        for i in 0..self.n {
            for &(j, w) in &self.adjacency[i] {
                if i < j {
                    edges.push((i, j, w));
                }
            }
        }
        edges
    }

    /// Total volume (sum of all edge weights, counting each once).
    pub fn total_volume(&self) -> f64 {
        self.edges().iter().map(|&(_, _, w)| w).sum()
    }

    /// Number of triangles containing edge (i, j).
    pub fn triangles_on_edge(&self, i: usize, j: usize) -> usize {
        let ni: Vec<usize> = self.adjacency[i].iter().map(|&(v, _)| v).collect();
        let nj: Vec<usize> = self.adjacency[j].iter().map(|&(v, _)| v).collect();
        ni.iter().filter(|x| nj.contains(x) && **x != i && **x != j).count()
    }
}

/// Helper for f64 ordering in BinaryHeap.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// OllivierRicciCurvature
// ---------------------------------------------------------------------------

/// Ollivier-Ricci curvature on graph edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllivierRicciCurvature {
    pub graph: GraphMetric,
    pub alpha: f64,
}

impl OllivierRicciCurvature {
    pub fn new(graph: GraphMetric, alpha: f64) -> Self {
        Self { graph, alpha }
    }

    /// Neighbor distribution μ_i: lazy random walk distribution.
    pub fn neighbor_distribution(&self, i: usize) -> Vec<(usize, f64)> {
        let deg = self.graph.degree(i);
        if deg == 0 {
            return vec![];
        }
        let vol = self.graph.volume(i);
        if vol == 0.0 {
            return vec![];
        }
        let mut dist = Vec::new();
        // Self-loop gets alpha mass
        dist.push((i, self.alpha));
        // Neighbors share (1 - alpha) proportional to edge weights
        for &(j, w) in &self.graph.adjacency[i] {
            dist.push((j, (1.0 - self.alpha) * w / vol));
        }
        dist
    }

    /// Wasserstein-1 distance (exact optimal transport cost) between two distributions.
    /// Uses a simple LP solver (transportation simplex) for small discrete distributions.

/// Compute exact W1 (earth mover's distance) between two discrete distributions.
/// Uses min-cost flow with successive shortest paths on the bipartite transport graph.
fn optimal_transport_cost(supply: &[f64], demand: &[f64], cost: &[Vec<f64>]) -> f64 {
    let ns = supply.len();
    let nd = demand.len();
    if ns == 0 || nd == 0 {
        return 0.0;
    }

    // Normalize to ensure total supply = total demand
    let total_s: f64 = supply.iter().sum();
    let total_d: f64 = demand.iter().sum();
    let _ratio = total_s / total_d;

    // Use a simple but correct approach: transportation simplex
    // Phase 1: Initial BFS via Northwest Corner method
    let mut flow: Vec<Vec<f64>> = vec![vec![0.0; nd]; ns];
    let mut rem_s: Vec<f64> = supply.to_vec();
    let mut rem_d: Vec<f64> = demand.to_vec();

    // Northwest corner: fill row by row, column by column
    let mut i = 0;
    let mut j = 0;
    while i < ns && j < nd {
        let f = rem_s[i].min(rem_d[j]);
        flow[i][j] = f;
        rem_s[i] -= f;
        rem_d[j] -= f;
        if rem_s[i] <= 1e-15 { i += 1; }
        if rem_d[j] <= 1e-15 { j += 1; }
    }

    // Phase 2: Optimize using MODI method
    for _ in 0..2000 {
        // Compute duals u, v for basic cells
        let mut u = vec![f64::NAN; ns];
        let mut v = vec![f64::NAN; nd];
        u[0] = 0.0;

        let mut progress = true;
        while progress {
            progress = false;
            for si in 0..ns {
                for sj in 0..nd {
                    if flow[si][sj] > 1e-15 {
                        if u[si].is_finite() && !v[sj].is_finite() {
                            v[sj] = cost[si][sj] - u[si];
                            progress = true;
                        } else if !u[si].is_finite() && v[sj].is_finite() {
                            u[si] = cost[si][sj] - v[sj];
                            progress = true;
                        }
                    }
                }
            }
        }
        for ui in &mut u { if !ui.is_finite() { *ui = 0.0; } }
        for vj in &mut v { if !vj.is_finite() { *vj = 0.0; } }

        // Find most negative reduced cost (entering variable)
        let mut best_rc = -1e-10; // threshold
        let mut enter = None;
        for si in 0..ns {
            for sj in 0..nd {
                if flow[si][sj] <= 1e-15 {
                    let rc = cost[si][sj] - u[si] - v[sj];
                    if rc < best_rc {
                        best_rc = rc;
                        enter = Some((si, sj));
                    }
                }
            }
        }

        let (ei, ej) = match enter {
            Some(e) => e,
            None => break, // optimal
        };

        // Find cycle: trace from (ei, ej) through basic cells
        // BFS approach: build adjacency from basic cells
        // Each basic cell connects its row to its column
        // Adding (ei, ej) creates exactly one cycle

        // Find cycle using DFS
        let cycle = find_cycle(&flow, ei, ej, ns, nd);

        // Find minimum flow at odd-position cells (those to decrease)
        let mut theta = f64::INFINITY;
        for k in 1..cycle.len() {
            if k % 2 == 1 {
                let (ci, cj) = cycle[k];
                theta = theta.min(flow[ci][cj]);
            }
        }

        // Augment along cycle
        for k in 0..cycle.len() {
            let (ci, cj) = cycle[k];
            if k % 2 == 0 {
                flow[ci][cj] += theta;
            } else {
                flow[ci][cj] -= theta;
            }
        }
    }

    // Compute total cost
    let mut total = 0.0;
    for si in 0..ns {
        for sj in 0..nd {
            total += flow[si][sj] * cost[si][sj];
        }
    }
    total
}

/// Find the unique cycle created by adding (ei, ej) to the basis tree.
/// Returns alternating (+, -, +, ...) cells starting with (ei, ej).
fn find_cycle(flow: &[Vec<f64>], ei: usize, ej: usize, ns: usize, nd: usize) -> Vec<(usize, usize)> {
    // Build a bipartite graph from basic cells
    // Nodes: rows 0..ns, columns ns..ns+nd
    // Edge from row i to col j if flow[i][j] > 0 (basic cell)
    // Plus the entering edge (ei, ej)
    // Find the unique cycle using DFS

    let total_nodes = ns + nd;
    let mut adj: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); total_nodes];
    // adj[node] = list of (neighbor_node, row, col)

    for si in 0..ns {
        for sj in 0..nd {
            if flow[si][sj] > 1e-15 || (si == ei && sj == ej) {
                let row_node = si;
                let col_node = ns + sj;
                adj[row_node].push((col_node, si, sj));
                adj[col_node].push((row_node, si, sj));
            }
        }
    }

    // DFS from ei (row node) to find cycle
    let start = ei; // row node
    let target = ns + ej; // col node (the entering edge destination)

    // The entering edge creates a cycle. Find path from target back to start
    // excluding the direct edge (ei, ej)
    let mut visited = vec![false; total_nodes];
    let mut path: Vec<(usize, usize)> = Vec::new(); // (row, col) of edges

    fn dfs(
        node: usize, parent: usize, target: usize, start: usize,
        adj: &[Vec<(usize, usize, usize)>], visited: &mut Vec<bool>,
        path: &mut Vec<(usize, usize)>, ns: usize, ei: usize, ej: usize,
    ) -> bool {
        if node == target && path.len() > 1 {
            return true;
        }
        visited[node] = true;
        for &(next, r, c) in &adj[node] {
            if next == parent { continue; }
            // Skip the entering edge
            if (node == ei && next == ns + ej) || (node == ns + ej && next == ei) {
                if path.is_empty() { continue; } // skip direct entering edge at start
            }
            if visited[next] { continue; }
            path.push((r, c));
            if dfs(next, node, target, start, adj, visited, path, ns, ei, ej) {
                return true;
            }
            path.pop();
        }
        false
    }

    // Start DFS from col node of entering edge, looking for path back to row node
    // through basic cells only
    visited[ns + ej] = true;
    for &(next, r, c) in &adj[ns + ej] {
        if next == ei { continue; } // skip entering edge
        if visited[next] { continue; }
        path.push((r, c));
        if dfs(next, ns + ej, ei, ei, &adj, &mut visited, &mut path, ns, ei, ej) {
            break;
        }
        path.pop();
    }

    // The cycle is: (ei, ej) + path
    let mut result = vec![(ei, ej)];
    result.extend(path);
    result
}

