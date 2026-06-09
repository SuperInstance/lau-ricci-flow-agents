//! Tutorial: Ricci Flow on Agent Graphs — Geometry Meets Network Science
//!
//! Run with: cargo run --example tutorial

use lau_ricci_flow_agents::*;

fn main() {
    println!("=== Lesson 1: Building Graphs ===\n");
    {
        // Graph constructors for common topologies
        let complete = complete_graph(4);
        let path = path_graph(5);
        let cycle = cycle_graph(5);
        let star = star_graph(4);

        println!("Complete K4: {} edges", complete.edges().len());
        println!("Path P5:     {} edges", path.edges().len());
        println!("Cycle C5:    {} edges", cycle.edges().len());
        println!("Star S4:     {} edges", star.edges().len());
        println!("Path diameter: {}", path.diameter());
        println!();
    }

    println!("=== Lesson 2: Graph Distances & Metrics ===\n");
    {
        let mut g = GraphMetric::new(5);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(2, 3, 1.0);
        g.add_edge(3, 4, 1.0);

        println!("Weighted path graph 0-1-2-3-4");
        println!("d(0,2) = {}", g.distance(0, 2)); // 1+2=3
        println!("d(0,4) = {}", g.distance(0, 4)); // 1+2+1+1=5
        println!("vol(2) = {}", g.volume(2));       // 2+1=3
        println!("degree(0) = {}", g.degree(0));     // 1
        println!();
    }

    println!("=== Lesson 3: Ollivier-Ricci Curvature ===\n");
    {
        // Ollivier-Ricci curvature: κ(i,j) = 1 - W1(μᵢ, μⱼ)/d(i,j)
        // Positive = edges in dense regions (triangles), Negative = bridges/bottlenecks

        let cycle = cycle_graph(6);
        let orc = OllivierRicciCurvature::new(cycle.clone(), 0.5);

        println!("Cycle C6 — all edges should have same curvature:");
        for &(i, j, _) in &cycle.edges() {
            println!("  κ({},{}) = {:.4}", i, j, orc.curvature(i, j));
        }
        println!("Average: {:.4}", orc.average_curvature());

        let star = star_graph(5);
        let orc_star = OllivierRicciCurvature::new(star.clone(), 0.5);
        println!("\nStar S5 — center edges:");
        for &(i, j, _) in &star.edges() {
            println!("  κ({},{}) = {:.4}", i, j, orc_star.curvature(i, j));
        }
        println!();
    }

    println!("=== Lesson 4: Forman-Ricci Curvature ===\n");
    {
        // Forman: F(i,j) = 4 - deg(i) - deg(j) + 3·triangles(i,j)
        let form = FormRicciCurvature::new();
        let cycle = cycle_graph(6);
        let complete = complete_graph(4);

        println!("Forman curvature on C6:");
        for &(i, j, _) in &cycle.edges() {
            println!("  F({},{}) = {:.1}", i, j, form.curvature(&cycle, i, j));
        }
        println!("\nForman curvature on K4 (lots of triangles):");
        for &(i, j, _) in &complete.edges() {
            println!("  F({},{}) = {:.1}", i, j, form.curvature(&complete, i, j));
        }

        // Correlation between Ollivier and Forman
        let orc = OllivierRicciCurvature::new(cycle.clone(), 0.5);
        println!("\nOllivier-Forman correlation: {:.4}", form.compare_with_ollivier(&orc));
        println!();
    }

    println!("=== Lesson 5: Ricci Flow Evolution ===\n");
    {
        // Ricci flow: dw/dt = -κ·w (edges shrink where curvature is negative)
        let mut g = barbell_graph(3, 2); // two cliques + bridge
        println!("Barbell graph (3+2+3): {} edges", g.edges().len());

        let mut rf = RicciFlow::new(g, 0.01);
        let snapshots = rf.run(10);

        println!("Ricci flow evolution (10 steps):");
        for snap in &snapshots[0..3] {
            let avg = snap.curvatures.iter().sum::<f64>() / snap.curvatures.len().max(1) as f64;
            println!("  t={}: avg κ = {:.4}, variance = {:.6}",
                snap.time, avg, snap.curvature_variance());
        }
        println!();
    }

    println!("=== Lesson 6: Curvature-Based Community Detection ===\n");
    {
        // After Ricci flow, bridge edges shrink → communities separate
        let mut g = barbell_graph(4, 3);
        let cc = CurvatureCommunity::new();
        let communities = cc.detect_communities(&mut g, 5, 1e-6);

        println!("Barbell (4+3+4) communities:");
        for (i, c) in communities.iter().enumerate() {
            println!("  Community {}: {:?}", i, c);
        }
        let modularity = cc.modularity(&communities, &g);
        println!("Modularity: {:.4}", modularity);

        let sil = cc.silhouette_score(&communities, &g);
        println!("Silhouette score: {:.4}", sil);
        println!();
    }

    println!("=== Lesson 7: Spectral Analysis ===\n");
    {
        let cycle = cycle_graph(6);
        let spec = CurvatureSpectrum::from_graph(&cycle);

        println!("Cycle C6 spectrum:");
        println!("  Eigenvalues: {:?}", spec.eigenvalues.iter()
            .map(|v| format!("{:.3}", v)).collect::<Vec<_>>());
        println!("  Spectral gap: {:.4}", spec.spectral_gap());
        println!("  Cheeger constant: {:.4}", spec.cheeger_constant());
        println!("  Is expander? {}", spec.is_expander());

        let complete = complete_graph(8);
        let spec_k = CurvatureSpectrum::from_graph(&complete);
        println!("\nComplete K8:");
        println!("  Spectral gap: {:.4}", spec_k.spectral_gap());
        println!("  Is expander? {}", spec_k.is_expander());
        println!();
    }

    println!("=== Lesson 8: Agent Similarity Graphs ===\n");
    {
        // Build a graph from agent feature vectors
        let agents = vec![
            AgentProfile::new("alpha", vec![1.0, 0.0, 0.0], vec!["math".into()]),
            AgentProfile::new("beta",  vec![0.9, 0.1, 0.0], vec!["math".into()]),
            AgentProfile::new("gamma", vec![0.0, 1.0, 0.0], vec!["music".into()]),
            AgentProfile::new("delta", vec![0.0, 0.9, 0.1], vec!["music".into()]),
            AgentProfile::new("omega", vec![0.5, 0.5, 0.5], vec!["general".into()]),
        ];

        println!("Agent similarities:");
        for i in 0..agents.len() {
            for j in (i+1)..agents.len() {
                println!("  {}-{}: {:.3}",
                    agents[i].id, agents[j].id,
                    agents[i].similarity(&agents[j]));
            }
        }

        let mut sg = AgentSimilarityGraph::new(agents);
        sg.build_from_features(0.7); // threshold for edge creation
        println!("\nSimilarity graph: {} edges (threshold 0.7)", sg.graph.edges().len());

        let communities = sg.evolve_communities(3);
        println!("Communities:");
        for c in &communities {
            println!("  Members {:?}, cohesion: {:.4}", c.members, c.cohesion);
        }
        println!();
    }

    println!("Tutorial complete!");
    println!("Key insight: Ricci flow smooths curvature on graphs,");
    println!("causing bridge edges to shrink and communities to separate.");
}
