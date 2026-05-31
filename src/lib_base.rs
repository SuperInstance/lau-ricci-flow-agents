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
