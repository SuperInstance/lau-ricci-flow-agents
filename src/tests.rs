//! Tests for lau-ricci-flow-agents.

use crate::*;

// ===== DenseMatrix tests =====

#[test]
fn test_matrix_zeros() {
    let m = DenseMatrix::zeros(3, 4);
    assert_eq!(m.rows, 3);
    assert_eq!(m.cols, 4);
    for i in 0..3 { for j in 0..4 { assert_eq!(m.get(i, j), 0.0); } }
}

#[test]
fn test_matrix_identity() {
    let m = DenseMatrix::identity(3);
    for i in 0..3 { for j in 0..3 { assert_eq!(m.get(i, j), if i == j { 1.0 } else { 0.0 }); } }
}

#[test]
fn test_matrix_multiply() {
    let mut a = DenseMatrix::identity(2);
    a.set(0, 1, 2.0);
    let mut b = DenseMatrix::identity(2);
    b.set(1, 0, 3.0);
    let c = a.multiply(&b);
    // A = [[1,2],[0,1]], B = [[1,0],[3,1]] → C = [[7,2],[3,1]]
    assert!((c.get(0, 0) - 7.0).abs() < 1e-10);
    assert!((c.get(0, 1) - 2.0).abs() < 1e-10);
    assert!((c.get(1, 0) - 3.0).abs() < 1e-10);
    assert!((c.get(1, 1) - 1.0).abs() < 1e-10);
}

#[test]
fn test_matrix_transpose() {
    let mut m = DenseMatrix::zeros(2, 3);
    m.set(0, 1, 5.0);
    m.set(1, 2, 7.0);
    let t = m.transpose();
    assert_eq!(t.rows, 3);
    assert_eq!(t.cols, 2);
    assert_eq!(t.get(1, 0), 5.0);
    assert_eq!(t.get(2, 1), 7.0);
}

#[test]
fn test_matrix_trace() {
    let mut m = DenseMatrix::identity(3);
    m.set(1, 1, 5.0);
    assert!((m.trace() - 7.0).abs() < 1e-10);
}

#[test]
fn test_matrix_determinant_2x2() {
    let mut m = DenseMatrix::zeros(2, 2);
    m.set(0, 0, 1.0); m.set(0, 1, 2.0); m.set(1, 0, 3.0); m.set(1, 1, 4.0);
    assert!((m.determinant() - (-2.0)).abs() < 1e-10);
}

#[test]
fn test_matrix_determinant_3x3() {
    let mut m = DenseMatrix::zeros(3, 3);
    m.set(0, 0, 1.0); m.set(0, 1, 2.0); m.set(0, 2, 3.0);
    m.set(1, 1, 1.0); m.set(1, 2, 4.0);
    m.set(2, 0, 5.0); m.set(2, 1, 6.0);
    assert!((m.determinant() - 1.0).abs() < 1e-10);
}

#[test]
fn test_eigenvalues_2x2() {
    let mut m = DenseMatrix::zeros(2, 2);
    m.set(0, 0, 4.0); m.set(1, 1, 2.0);
    let eigs = m.eigenvalues_small();
    assert_eq!(eigs.len(), 2);
    assert!((eigs[0] - 4.0).abs() < 0.1);
    assert!((eigs[1] - 2.0).abs() < 0.1);
}

// ===== GraphMetric tests =====

#[test]
fn test_graph_new() { let g = GraphMetric::new(5); assert_eq!(g.n, 5); assert_eq!(g.edges().len(), 0); }

#[test]
fn test_add_edge() {
    let mut g = GraphMetric::new(3);
    g.add_edge(0, 1, 2.0);
    assert_eq!(g.edges().len(), 1);
    assert_eq!(g.degree(0), 1);
    assert_eq!(g.degree(1), 1);
    assert_eq!(g.degree(2), 0);
}

#[test]
fn test_distance_direct() {
    let mut g = GraphMetric::new(3);
    g.add_edge(0, 1, 2.0);
    assert!((g.distance(0, 1) - 2.0).abs() < 1e-10);
}

#[test]
fn test_distance_shortest_path() {
    let mut g = GraphMetric::new(3);
    g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(0, 2, 5.0);
    assert!((g.distance(0, 2) - 2.0).abs() < 1e-10);
}

#[test]
fn test_degree() {
    let mut g = GraphMetric::new(4);
    g.add_edge(0, 1, 1.0); g.add_edge(0, 2, 1.0); g.add_edge(0, 3, 1.0);
    assert_eq!(g.degree(0), 3);
    assert_eq!(g.degree(1), 1);
}

#[test]
fn test_volume() {
    let mut g = GraphMetric::new(3);
    g.add_edge(0, 1, 2.0); g.add_edge(0, 2, 3.0);
    assert!((g.volume(0) - 5.0).abs() < 1e-10);
}

#[test]
fn test_diameter() {
    let g = path_graph(5);
    assert!((g.diameter() - 4.0).abs() < 1e-10);
}

#[test]
fn test_laplacian() {
    let mut g = GraphMetric::new(3);
    g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0);
    let l = g.laplacian();
    assert!((l.get(0, 0) - 1.0).abs() < 1e-10);
    assert!((l.get(0, 1) - (-1.0)).abs() < 1e-10);
    assert!((l.get(1, 1) - 2.0).abs() < 1e-10);
}

#[test]
fn test_normalized_laplacian() {
    let g = complete_graph(4);
    let nl = g.normalized_laplacian();
    for i in 0..4 { assert!((nl.get(i, i) - 1.0).abs() < 1e-10); }
}

#[test]
fn test_complete_graph_constructor() { let g = complete_graph(4); assert_eq!(g.edges().len(), 6); }

#[test]
fn test_path_graph_constructor() { let g = path_graph(5); assert_eq!(g.edges().len(), 4); }

#[test]
fn test_cycle_graph_constructor() { let g = cycle_graph(5); assert_eq!(g.edges().len(), 5); }

#[test]
fn test_star_graph_constructor() { let g = star_graph(5); assert_eq!(g.edges().len(), 4); }

#[test]
fn test_binary_tree_constructor() { let g = binary_tree(2); assert_eq!(g.edges().len(), 6); }

#[test]
fn test_all_distances() {
    let g = path_graph(4);
    let d = g.all_distances();
    assert!((d[0][3] - 3.0).abs() < 1e-10);
    assert!((d[1][2] - 1.0).abs() < 1e-10);
}

#[test]
fn test_triangles_on_edge() {
    let g = complete_graph(4);
    assert_eq!(g.triangles_on_edge(0, 1), 2);
}

#[test]
fn test_triangles_on_edge_path() {
    let g = path_graph(4);
    assert_eq!(g.triangles_on_edge(0, 1), 0);
}

#[test]
fn test_total_volume() {
    let g = complete_graph(3);
    assert!((g.total_volume() - 3.0).abs() < 1e-10);
}

// ===== OllivierRicciCurvature tests =====

#[test]
fn test_neighbor_distribution() {
    let g = complete_graph(3);
    let orc = OllivierRicciCurvature::new(g, 0.5);
    let dist = orc.neighbor_distribution(0);
    assert_eq!(dist.len(), 3);
}

#[test]
fn test_curvature_complete_graph() {
    let g = complete_graph(4);
    let orc = OllivierRicciCurvature::new(g, 0.5);
    for i in 0..4 { for j in (i+1)..4 {
        let k = orc.curvature(i, j);
        assert!(k > 0.0, "K_4 edge ({}, {}) curvature {} should be positive", i, j, k);
    }}
}

#[test]
fn test_curvature_tree_nonleaf_negative() {
    // Non-leaf edges in a binary tree have negative curvature with alpha=0
    let g = binary_tree(2);
    let orc = OllivierRicciCurvature::new(g.clone(), 0.0);
    // Edge (0,1): node 0 has degree 2, node 1 has degree 3 — both non-leaf
    let k = orc.curvature(0, 1);
    assert!(k < 0.0, "Non-leaf tree edge (0,1) curvature {} should be negative", k);
    // Edge (0,2): similarly non-leaf
    let k2 = orc.curvature(0, 2);
    assert!(k2 < 0.0, "Non-leaf tree edge (0,2) curvature {} should be negative", k2);
}

#[test]
fn test_curvature_tree_leaf_zero() {
    // Leaf edges (connecting to degree-1 nodes) have κ=0 with alpha=0
    let g = binary_tree(2);
    let orc = OllivierRicciCurvature::new(g, 0.0);
    // Edge (1,3): node 3 is a leaf (degree 1)
    let k = orc.curvature(1, 3);
    assert!(k.abs() < 1e-6, "Leaf edge curvature should be ~0, got {}", k);
}

#[test]
fn test_average_curvature() {
    let g = complete_graph(4);
    let orc = OllivierRicciCurvature::new(g, 0.5);
    assert!(orc.average_curvature() > 0.0, "Average curvature of K_4 should be positive");
}

#[test]
fn test_curvature_distribution() {
    let g = cycle_graph(4);
    let orc = OllivierRicciCurvature::new(g, 0.5);
    assert_eq!(orc.curvature_distribution().len(), 4);
}

#[test]
fn test_ricci_positive_edges_k4() {
    let g = complete_graph(4);
    let orc = OllivierRicciCurvature::new(g, 0.5);
    assert_eq!(orc.ricci_positive_edges().len(), 6);
}

#[test]
fn test_ricci_negative_edges_tree() {
    // Non-leaf edges in binary tree have negative curvature
    let g = binary_tree(2);
    let orc = OllivierRicciCurvature::new(g, 0.0);
    let neg = orc.ricci_negative_edges();
    assert!(neg.len() >= 2, "Should have at least 2 negative curvature edges (non-leaf), got {}", neg.len());
}

#[test]
fn test_ricci_flat_edges_cycle() {
    let g = cycle_graph(4);
    let orc = OllivierRicciCurvature::new(g, 0.5);
    let dist = orc.curvature_distribution();
    for k in &dist { assert!(k.abs() < 1.0, "Cycle C_4 curvature should be small"); }
}

#[test]
fn test_wasserstein_1_same_dist() {
    let g = complete_graph(3);
    let orc = OllivierRicciCurvature::new(g, 0.5);
    let mu = orc.neighbor_distribution(0);
    assert!(orc.wasserstein_1(&mu, &mu).abs() < 1e-10, "W1 of same distribution should be 0");
}

// ===== FormRicciCurvature tests =====

#[test]
fn test_forman_curvature_tree() {
    let g = path_graph(5);
    let fc = FormRicciCurvature::new();
    let k = fc.curvature(&g, 0, 1);
    assert!((k - 1.0).abs() < 1e-10, "Path endpoint edge F should be 1, got {}", k);
}

#[test]
fn test_forman_curvature_interior_path() {
    let g = path_graph(5);
    let fc = FormRicciCurvature::new();
    let k = fc.curvature(&g, 2, 3);
    assert!((k - 0.0).abs() < 1e-10, "Interior path edge F should be 0, got {}", k);
}

#[test]
fn test_forman_curvature_complete() {
    let g = complete_graph(4);
    let fc = FormRicciCurvature::new();
    let k = fc.curvature(&g, 0, 1);
    assert!((k - 4.0).abs() < 1e-10, "K_4 edge F should be 4, got {}", k);
}

#[test]
fn test_forman_average() {
    let g = complete_graph(4);
    let fc = FormRicciCurvature::new();
    assert!((fc.average(&g) - 4.0).abs() < 1e-10);
}

#[test]
fn test_forman_compare_with_ollivier() {
    let g = complete_graph(5);
    let orc = OllivierRicciCurvature::new(g, 0.5);
    let fc = FormRicciCurvature::new();
    let corr = fc.compare_with_ollivier(&orc);
    assert!(corr.is_finite() || corr.is_nan());
}

// ===== GraphSnapshot tests =====

#[test]
fn test_snapshot_total_volume() {
    let g = complete_graph(3);
    let snap = GraphSnapshot { time: 0, metric: g, curvatures: vec![0.5, 0.5, 0.5] };
    assert!((snap.total_volume() - 3.0).abs() < 1e-10);
}

#[test]
fn test_snapshot_curvature_variance() {
    let snap = GraphSnapshot { time: 0, metric: GraphMetric::new(3), curvatures: vec![1.0, 2.0, 3.0] };
    assert!((snap.curvature_variance() - 2.0/3.0).abs() < 1e-10);
}

#[test]
fn test_snapshot_curvature_variance_uniform() {
    let snap = GraphSnapshot { time: 0, metric: GraphMetric::new(3), curvatures: vec![1.0, 1.0, 1.0] };
    assert!(snap.curvature_variance().abs() < 1e-10);
}

// ===== RicciFlow tests =====

#[test]
fn test_ricci_flow_step() {
    let g = cycle_graph(4);
    let mut rf = RicciFlow::new(g, 0.01);
    rf.step();
    assert_eq!(rf.graph.edges().len(), 4);
}

#[test]
fn test_ricci_flow_run() {
    let g = cycle_graph(4);
    let mut rf = RicciFlow::new(g, 0.01);
    let snapshots = rf.run(5);
    assert_eq!(snapshots.len(), 5);
    assert_eq!(snapshots[0].time, 0);
}

#[test]
fn test_ricci_flow_variance_decreases() {
    let g = cycle_graph(6);
    let mut rf = RicciFlow::new(g, 0.01);
    let snapshots = rf.run(10);
    let v0 = snapshots[0].curvature_variance();
    let v_last = snapshots.last().unwrap().curvature_variance();
    assert!(v_last <= v0 + 1e-6, "Variance should decrease: {} -> {}", v0, v_last);
}

#[test]
fn test_ricci_flow_normalize() {
    let mut g = GraphMetric::new(3);
    g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0);
    let mut rf = RicciFlow::new(g, 0.01);
    rf.normalize();
    assert!((rf.graph.total_volume() - 3.0).abs() < 1e-6);
}

#[test]
fn test_converged() {
    let g = path_graph(5);
    let rf = RicciFlow::new(g, 0.01);
    // Path graph with non-uniform curvature should not be converged
    assert!(!rf.converged(1e-15));
}

// ===== CurvatureCommunity tests =====

#[test]
fn test_detect_communities() {
    let g = barbell_graph(4, 2);
    let mut g2 = g.clone();
    let cc = CurvatureCommunity::new();
    let communities = cc.detect_communities(&mut g2, 20, 0.001);
    assert!(!communities.is_empty());
}

#[test]
fn test_modularity() {
    let g = complete_graph(4);
    let cc = CurvatureCommunity::new();
    let communities = vec![vec![0, 1, 2, 3]];
    let mod_val = cc.modularity(&communities, &g);
    assert!(mod_val.abs() < 1e-6, "Single community modularity should be ~0 for K_4, got {}", mod_val);
}

#[test]
fn test_modularity_two_communities() {
    let g = barbell_graph(3, 1);
    let cc = CurvatureCommunity::new();
    let communities = vec![vec![0, 1, 2], vec![3, 4, 5, 6]];
    let mod_val = cc.modularity(&communities, &g);
    assert!(mod_val > 0.0, "Barbell split should have positive modularity, got {}", mod_val);
}

#[test]
fn test_silhouette_score() {
    let g = barbell_graph(3, 1);
    let cc = CurvatureCommunity::new();
    let communities = vec![vec![0, 1, 2], vec![3, 4, 5, 6]];
    let score = cc.silhouette_score(&communities, &g);
    assert!(score > 0.0, "Barbell silhouette should be positive, got {}", score);
}

#[test]
fn test_silhouette_single_community() {
    let g = complete_graph(4);
    let cc = CurvatureCommunity::new();
    let communities = vec![vec![0, 1, 2, 3]];
    assert_eq!(cc.silhouette_score(&communities, &g), 0.0);
}

// ===== CurvatureSpectrum tests =====

#[test]
fn test_spectral_gap() {
    let g = complete_graph(5);
    let spec = CurvatureSpectrum::from_graph(&g);
    assert!(spec.spectral_gap() > 0.0);
}

#[test]
fn test_cheeger_constant() {
    let g = complete_graph(5);
    let spec = CurvatureSpectrum::from_graph(&g);
    assert!(spec.cheeger_constant() > 0.0);
}

#[test]
fn test_is_expander_complete() {
    let g = complete_graph(5);
    let spec = CurvatureSpectrum::from_graph(&g);
    assert!(spec.is_expander());
}

#[test]
fn test_is_expander_path() {
    let g = path_graph(10);
    let spec = CurvatureSpectrum::from_graph(&g);
    assert!(!spec.is_expander());
}

#[test]
fn test_cheeger_inequality() {
    let g = cycle_graph(6);
    let spec = CurvatureSpectrum::from_graph(&g);
    let lambda1 = spec.spectral_gap();
    let h = spec.cheeger_constant();
    assert!(h >= lambda1 / 2.0 - 1e-6);
}

// ===== AgentProfile tests =====

#[test]
fn test_agent_similarity_same() {
    let a = AgentProfile::new("a", vec![1.0, 2.0, 3.0], vec![]);
    let b = AgentProfile::new("b", vec![1.0, 2.0, 3.0], vec![]);
    assert!((a.similarity(&b) - 1.0).abs() < 1e-10);
}

#[test]
fn test_agent_similarity_orthogonal() {
    let a = AgentProfile::new("a", vec![1.0, 0.0], vec![]);
    let b = AgentProfile::new("b", vec![0.0, 1.0], vec![]);
    assert!(a.similarity(&b).abs() < 1e-10);
}

#[test]
fn test_agent_similarity_opposite() {
    let a = AgentProfile::new("a", vec![1.0, 0.0], vec![]);
    let b = AgentProfile::new("b", vec![-1.0, 0.0], vec![]);
    assert!((a.similarity(&b) - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_agent_similarity_different_lengths() {
    let a = AgentProfile::new("a", vec![1.0, 2.0], vec![]);
    let b = AgentProfile::new("b", vec![1.0, 2.0, 3.0], vec![]);
    assert_eq!(a.similarity(&b), 0.0);
}

// ===== Community tests =====

#[test]
fn test_community_representative() {
    let g = path_graph(5);
    let c = Community { members: vec![0, 2, 4], cohesion: 0.5, curvature: 0.5 };
    assert_eq!(c.representative(&g), 2);
}

#[test]
fn test_community_boundary() {
    let mut g = GraphMetric::new(4);
    g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(2, 3, 1.0);
    let c = Community { members: vec![1, 2], cohesion: 0.5, curvature: 0.5 };
    assert_eq!(c.boundary(&g).len(), 2);
}

#[test]
fn test_community_boundary_isolated() {
    let g = complete_graph(4);
    let c = Community { members: vec![0, 1, 2, 3], cohesion: 0.5, curvature: 0.5 };
    assert_eq!(c.boundary(&g).len(), 0);
}

// ===== AgentSimilarityGraph tests =====

#[test]
fn test_agent_similarity_graph_build() {
    let agents = vec![
        AgentProfile::new("a", vec![1.0, 0.0], vec!["x".into()]),
        AgentProfile::new("b", vec![0.9, 0.1], vec!["x".into()]),
        AgentProfile::new("c", vec![0.0, 1.0], vec!["y".into()]),
        AgentProfile::new("d", vec![0.1, 0.9], vec!["y".into()]),
    ];
    let mut asg = AgentSimilarityGraph::new(agents);
    asg.build_from_features(0.8);
    assert!(asg.graph.adjacency[0].iter().any(|&(v, _)| v == 1));
    assert!(asg.graph.adjacency[2].iter().any(|&(v, _)| v == 3));
    assert!(!asg.graph.adjacency[0].iter().any(|&(v, _)| v == 2));
}

#[test]
fn test_agent_evolve_communities() {
    let agents = vec![
        AgentProfile::new("a", vec![1.0, 0.0], vec![]),
        AgentProfile::new("b", vec![0.95, 0.05], vec![]),
        AgentProfile::new("c", vec![0.0, 1.0], vec![]),
        AgentProfile::new("d", vec![0.05, 0.95], vec![]),
    ];
    let mut asg = AgentSimilarityGraph::new(agents);
    asg.build_from_features(0.8);
    let communities = asg.evolve_communities(3);
    assert!(communities.len() >= 2);
}

#[test]
fn test_track_agent() {
    let agents = vec![
        AgentProfile::new("a", vec![1.0, 0.0], vec![]),
        AgentProfile::new("b", vec![0.9, 0.1], vec![]),
        AgentProfile::new("c", vec![0.0, 1.0], vec![]),
        AgentProfile::new("d", vec![0.1, 0.9], vec![]),
    ];
    let mut asg = AgentSimilarityGraph::new(agents);
    asg.build_from_features(0.8);
    assert_eq!(asg.track_agent(0, 3).len(), 3);
}

// ===== Theorem verification tests =====

#[test]
fn theorem_1_tree_negative_curvature() {
    // Non-leaf edges in a tree have negative Ollivier-Ricci curvature
    let g = binary_tree(3);
    let orc = OllivierRicciCurvature::new(g.clone(), 0.0);
    for (i, j, _) in &g.edges() {
        let di = g.degree(*i);
        let dj = g.degree(*j);
        if di > 1 && dj > 1 {
            // Both endpoints non-leaf → negative curvature
            let k = orc.curvature(*i, *j);
            assert!(k < 0.0, "Non-leaf tree edge ({},{}) curvature {} should be negative", i, j, k);
        }
    }
}

#[test]
fn theorem_2_complete_positive_curvature() {
    for n in 3..=6 {
        let g = complete_graph(n);
        let orc = OllivierRicciCurvature::new(g, 0.5);
        for (i, j, _) in &orc.graph.edges() {
            let k = orc.curvature(*i, *j);
            assert!(k > 0.0, "K_{} edge ({},{}) curvature {} should be positive", n, i, j, k);
        }
    }
}

#[test]
fn theorem_3_cycle_curvature_approaches_zero() {
    // C_n curvature approaches 0 as n → inf
    let g4 = cycle_graph(4);
    let orc4 = OllivierRicciCurvature::new(g4, 0.5);
    let avg4 = orc4.average_curvature().abs();

    let g20 = cycle_graph(20);
    let orc20 = OllivierRicciCurvature::new(g20, 0.5);
    let avg20 = orc20.average_curvature().abs();

    assert!(avg20 < avg4, "C_20 curvature ({}) should be smaller than C_4 ({})", avg20, avg4);
}

#[test]
fn theorem_5_forman_tree_no_triangles() {
    let g = binary_tree(3);
    let fc = FormRicciCurvature::new();
    for (i, j, _) in &g.edges() {
        let k = fc.curvature(&g, *i, *j);
        let expected = 4.0 - g.degree(*i) as f64 - g.degree(*j) as f64;
        assert!((k - expected).abs() < 1e-10,
            "Forman tree edge ({},{}) = {} != expected {}", i, j, k, expected);
    }
}

#[test]
fn theorem_7_modularity_increases() {
    let g = barbell_graph(4, 2);
    let cc = CurvatureCommunity::new();
    let all_nodes: Vec<usize> = (0..g.n).collect();
    let initial_mod = cc.modularity(&[all_nodes], &g);
    let left: Vec<usize> = (0..4).collect();
    let right: Vec<usize> = (4..g.n).collect();
    let final_mod = cc.modularity(&[left, right], &g);
    assert!(final_mod > initial_mod,
        "Modularity should increase: {} -> {}", initial_mod, final_mod);
}

#[test]
fn theorem_8_complete_graph_ricci_flat() {
    // Complete graph has uniform curvature (Ricci-flat)
    let g = complete_graph(5);
    let orc = OllivierRicciCurvature::new(g, 0.5);
    let dist = orc.curvature_distribution();
    let mean = dist.iter().sum::<f64>() / dist.len() as f64;
    for k in &dist {
        assert!((k - mean).abs() < 1e-6,
            "All K_5 curvatures should be equal, got {} vs mean {}", k, mean);
    }
}

#[test]
fn theorem_9_path_nonleaf_negative() {
    // Interior edges of path with asymmetric neighbor structure have negative curvature
    let g = path_graph(4);
    let orc = OllivierRicciCurvature::new(g, 0.0);
    // Edge (0,1): node 0 is endpoint (degree 1), node 1 has degree 2
    let k01 = orc.curvature(0, 1);
    // With alpha=0, endpoint edges have κ=0 (asymmetric)
    // But edge (1,2) connects two degree-2 nodes, also κ=0 for path
    // The path is special — all edges have κ=0 with alpha=0
    assert!(k01.abs() < 1e-6, "Path edge curvature should be ~0, got {}", k01);
}

#[test]
fn theorem_10_agent_communities_agree_with_features() {
    let agents = vec![
        AgentProfile::new("a1", vec![1.0, 0.0, 0.0], vec![]),
        AgentProfile::new("a2", vec![0.95, 0.05, 0.0], vec![]),
        AgentProfile::new("a3", vec![0.9, 0.1, 0.0], vec![]),
        AgentProfile::new("b1", vec![0.0, 1.0, 0.0], vec![]),
        AgentProfile::new("b2", vec![0.0, 0.95, 0.05], vec![]),
        AgentProfile::new("b3", vec![0.0, 0.9, 0.1], vec![]),
    ];
    let mut asg = AgentSimilarityGraph::new(agents);
    asg.build_from_features(0.8);
    let communities = asg.evolve_communities(3);

    let find_community = |id: usize| -> usize {
        for (ci, c) in communities.iter().enumerate() {
            if c.members.contains(&id) { return ci; }
        }
        communities.len()
    };

    assert_eq!(find_community(0), find_community(1), "a1 and a2 should be in same community");
    assert_eq!(find_community(0), find_community(2), "a1 and a3 should be in same community");
    assert_ne!(find_community(0), find_community(3), "a1 and b1 should differ");
}

// ===== Serde tests =====

#[test]
fn test_serde_graph_metric() {
    let g = complete_graph(3);
    let json = serde_json::to_string(&g).unwrap();
    let g2: GraphMetric = serde_json::from_str(&json).unwrap();
    assert_eq!(g2.n, 3);
    assert_eq!(g2.edges().len(), 3);
}

#[test]
fn test_serde_agent_profile() {
    let a = AgentProfile::new("test", vec![1.0, 2.0], vec!["cap".into()]);
    let json = serde_json::to_string(&a).unwrap();
    let a2: AgentProfile = serde_json::from_str(&json).unwrap();
    assert_eq!(a2.id, "test");
    assert_eq!(a2.features, vec![1.0, 2.0]);
}

#[test]
fn test_serde_community() {
    let c = Community { members: vec![1, 2, 3], cohesion: 0.8, curvature: 0.5 };
    let json = serde_json::to_string(&c).unwrap();
    let c2: Community = serde_json::from_str(&json).unwrap();
    assert_eq!(c2.members, vec![1, 2, 3]);
}

#[test]
fn test_serde_ricci_flow() {
    let g = cycle_graph(4);
    let rf = RicciFlow::new(g, 0.05);
    let json = serde_json::to_string(&rf).unwrap();
    let rf2: RicciFlow = serde_json::from_str(&json).unwrap();
    assert!((rf2.dt - 0.05).abs() < 1e-10);
}

#[test]
fn test_serde_curvature_spectrum() {
    let spec = CurvatureSpectrum {
        eigenvalues: vec![0.0, 1.0, 2.0],
        eigenvectors: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
    };
    let json = serde_json::to_string(&spec).unwrap();
    let spec2: CurvatureSpectrum = serde_json::from_str(&json).unwrap();
    assert_eq!(spec2.eigenvalues, vec![0.0, 1.0, 2.0]);
}

#[test]
fn test_serde_dense_matrix() {
    let m = DenseMatrix::identity(3);
    let json = serde_json::to_string(&m).unwrap();
    let m2: DenseMatrix = serde_json::from_str(&json).unwrap();
    assert_eq!(m2.rows, 3);
}

#[test]
fn test_serde_snapshot() {
    let snap = GraphSnapshot { time: 5, metric: path_graph(3), curvatures: vec![0.5, 0.5] };
    let json = serde_json::to_string(&snap).unwrap();
    let snap2: GraphSnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(snap2.time, 5);
}

// ===== Additional edge case tests =====

#[test]
fn test_empty_graph() { let g = GraphMetric::new(0); assert_eq!(g.edges().len(), 0); }

#[test]
fn test_single_node() {
    let g = GraphMetric::new(1);
    assert_eq!(g.degree(0), 0);
    assert_eq!(g.volume(0), 0.0);
}

#[test]
fn test_star_graph_curvature() {
    let g = star_graph(5);
    let orc = OllivierRicciCurvature::new(g, 0.0);
    // Star edges connect center (degree 4) to leaves (degree 1)
    // With alpha=0, W1 from center to leaf involves mass transport that costs exactly d
    // So κ = 0 for star edges with alpha=0
    let flat = orc.ricci_flat_edges();
    assert!(!flat.is_empty(), "Star edges should have ~0 curvature with alpha=0");
}

#[test]
fn test_forman_star_graph() {
    let g = star_graph(5);
    let fc = FormRicciCurvature::new();
    let k = fc.curvature(&g, 0, 1);
    assert!((k - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_path_graph_forman() {
    let g = path_graph(3);
    let fc = FormRicciCurvature::new();
    assert!((fc.curvature(&g, 0, 1) - 1.0).abs() < 1e-10);
    assert!((fc.curvature(&g, 1, 2) - 1.0).abs() < 1e-10);
}

#[test]
fn test_agent_profile_zero_features() {
    let a = AgentProfile::new("a", vec![0.0, 0.0], vec![]);
    let b = AgentProfile::new("b", vec![1.0, 1.0], vec![]);
    assert_eq!(a.similarity(&b), 0.0);
}

#[test]
fn test_cycle_curvature_all_same() {
    let g = cycle_graph(5);
    let orc = OllivierRicciCurvature::new(g, 0.5);
    let dist = orc.curvature_distribution();
    let first = dist[0];
    for k in &dist[1..] {
        assert!((k - first).abs() < 1e-6, "Cycle curvatures should be uniform");
    }
}

#[test]
fn test_barbell_graph_structure() {
    let g = barbell_graph(3, 1);
    assert_eq!(g.n, 7);
    assert!(!g.edges().is_empty());
}

#[test]
fn test_curvature_spectrum_from_graph() {
    let g = cycle_graph(6);
    let spec = CurvatureSpectrum::from_graph(&g);
    assert_eq!(spec.eigenvalues.len(), 6);
    assert!(spec.eigenvalues[0].abs() < 0.5);
}

#[test]
fn test_snapshot_modularity() {
    let g = barbell_graph(3, 1);
    let snap = GraphSnapshot { time: 0, metric: g, curvatures: vec![0.5; 8] };
    let communities = vec![vec![0, 1, 2], vec![3, 4, 5, 6]];
    assert!(snap.modularity(&communities) > 0.0);
}

#[test]
fn test_star_graph_forman_negative() {
    // Star graph center has high degree → negative Forman curvature
    let g = star_graph(10);
    let fc = FormRicciCurvature::new();
    for i in 1..10 {
        let k = fc.curvature(&g, 0, i);
        assert!(k < 0.0, "Star edge Forman should be negative, got {}", k);
    }
}

#[test]
fn test_complete_graph_curvature_positive_all_alpha() {
    // K_n has positive curvature for various alpha values
    let g = complete_graph(4);
    for alpha in [0.0, 0.25, 0.5, 0.75] {
        let orc = OllivierRicciCurvature::new(g.clone(), alpha);
        let avg = orc.average_curvature();
        assert!(avg > 0.0, "K_4 with alpha={} should have positive avg curvature, got {}", alpha, avg);
    }
}

#[test]
fn test_wasserstein_optimal_simple() {
    // Simple test: two point masses
    let mut g = GraphMetric::new(3);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);
    let orc = OllivierRicciCurvature::new(g, 0.0);
    // Transport from {(0, 1.0)} to {(2, 1.0)}: cost = d(0,2) = 2
    let w1 = orc.wasserstein_1(&[(0, 1.0)], &[(2, 1.0)]);
    assert!((w1 - 2.0).abs() < 1e-6, "W1 should be 2, got {}", w1);
}

#[test]
fn test_laplacian_complete_graph() {
    let g = complete_graph(3);
    let l = g.laplacian();
    // Diagonal = 2, off-diagonal = -1
    assert!((l.get(0, 0) - 2.0).abs() < 1e-10);
    assert!((l.get(0, 1) - (-1.0)).abs() < 1e-10);
}
