//! End-to-end: use `walk` to generate a hard-negative pool for `subsume`.
//!
//! This example threads together:
//! - `walk`: personalized PageRank (PPR) to find “nearby” nodes in a graph
//! - `subsume-core`: negative sampling from an explicit candidate pool
//! - `subsume-ndarray`: concrete box implementation
//!
//! Motivation: in metric learning terms, PPR-near nodes are often “harder negatives” than uniform
//! negatives because they share context/topology with the anchor.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use subsume_core::Box as CoreBox;
use subsume_core::dataset::Triple;
use subsume_core::trainer::{generate_negative_samples_from_pool_with_rng, NegativeSamplingStrategy};
use subsume_ndarray::NdarrayBox;
use walk::{personalized_pagerank, Graph, PageRankConfig};

use ndarray::Array1;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
struct Adj {
    adj: Vec<Vec<usize>>,
}

impl Adj {
    fn sbm_two_block(n: usize, p_in: f64, p_out: f64, seed: u64) -> Self {
        assert!(n >= 4);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut adj = vec![Vec::new(); n];
        let half = n / 2;
        for i in 0..n {
            for j in (i + 1)..n {
                let same = (i < half) == (j < half);
                let p = if same { p_in } else { p_out };
                if rng.random::<f64>() < p {
                    adj[i].push(j);
                    adj[j].push(i);
                }
            }
        }
        for nbrs in &mut adj {
            nbrs.sort_unstable();
            nbrs.dedup();
        }
        Self { adj }
    }

    fn from_undirected_edgelist(path: &Path) -> Result<Self, String> {
        let txt = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut max_node = 0usize;

        for (line_no, line) in txt.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            let a = it
                .next()
                .ok_or_else(|| format!("line {}: missing src", line_no + 1))?;
            let b = it
                .next()
                .ok_or_else(|| format!("line {}: missing dst", line_no + 1))?;
            let u: usize = a
                .parse()
                .map_err(|e| format!("line {}: bad src '{a}': {e}", line_no + 1))?;
            let v: usize = b
                .parse()
                .map_err(|e| format!("line {}: bad dst '{b}': {e}", line_no + 1))?;
            max_node = max_node.max(u).max(v);
            edges.push((u, v));
        }

        let n = max_node + 1;
        if n == 0 {
            return Err("edgelist produced empty graph".to_string());
        }

        let mut adj = vec![Vec::new(); n];
        for (u, v) in edges {
            if u == v {
                continue;
            }
            adj[u].push(v);
            adj[v].push(u);
        }
        for nbrs in &mut adj {
            nbrs.sort_unstable();
            nbrs.dedup();
        }
        Ok(Self { adj })
    }
}

impl walk::Graph for Adj {
    fn node_count(&self) -> usize {
        self.adj.len()
    }
    fn neighbors(&self, node: usize) -> Vec<usize> {
        self.adj[node].clone()
    }
    fn out_degree(&self, node: usize) -> usize {
        self.adj[node].len()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1) Build a graph.
    //
    // If you have a real graph edge list, point to it:
    // SUBSUME_EDGELIST=/path/to/edges.txt cargo run -p subsume-ndarray --example walk_hard_negatives
    //
    // Otherwise we fall back to a seeded SBM graph.
    let g = if let Ok(path) = std::env::var("SUBSUME_EDGELIST") {
        Adj::from_undirected_edgelist(Path::new(&path)).map_err(|e| format!("SUBSUME_EDGELIST: {e}"))?
    } else {
        Adj::sbm_two_block(200, 0.08, 0.005, 123)
    };
    let n = g.node_count();

    // 2) Assign entity ids: e000, e001, ...
    let ids: Vec<String> = (0..n).map(|i| format!("e{i:03}")).collect();

    // 3) Create boxes with a weak community structure:
    //    block A boxes are centered near -1, block B near +1.
    let d = 8usize;
    let mut rng = rand::rng();
    let mut boxes: HashMap<String, NdarrayBox> = HashMap::new();
    for i in 0..n {
        let block_sign = if i < n / 2 { -1.0f32 } else { 1.0f32 };
        let center: Vec<f32> = (0..d)
            .map(|_| block_sign + rng.random_range(-0.2..0.2))
            .collect();
        let size: Vec<f32> = (0..d).map(|_| rng.random_range(0.4..0.7)).collect();
        let min = Array1::from_iter(center.iter().zip(size.iter()).map(|(c, s)| c - s / 2.0));
        let max = Array1::from_iter(center.iter().zip(size.iter()).map(|(c, s)| c + s / 2.0));
        boxes.insert(ids[i].clone(), NdarrayBox::new(min, max, 1.0)?);
    }

    // 4) Pick one “positive triple”: head in A, tail is a neighbor in A.
    let head_i = 10usize;
    let tail_i = *g.adj[head_i]
        .iter()
        .find(|&&j| j < n / 2)
        .unwrap_or(&(head_i + 1));

    let triple = Triple {
        head: ids[head_i].clone(),
        relation: "r".to_string(),
        tail: ids[tail_i].clone(),
    };

    // 5) Compute a PPR vector from head to get a “hard candidate pool”.
    let mut personalization = vec![0.0f64; n];
    personalization[head_i] = 1.0;
    let ppr = personalized_pagerank(
        &g,
        PageRankConfig {
            damping: 0.85,
            max_iterations: 50,
            tolerance: 1e-9,
        },
        &personalization,
    );

    // Take top-K PPR nodes (excluding head and the true tail).
    let k_pool = 50usize;
    let mut ranked: Vec<usize> = (0..n).collect();
    ranked.sort_by(|&a, &b| ppr[b].total_cmp(&ppr[a]).then_with(|| a.cmp(&b)));
    let hard_pool_ids: Vec<String> = ranked
        .into_iter()
        .filter(|&j| j != head_i && j != tail_i)
        .take(k_pool)
        .map(|j| ids[j].clone())
        .collect();

    // 6) Compare “hard negatives” vs uniform negatives by their containment score.
    let head_box = boxes.get(&triple.head).unwrap();
    let tail_score = head_box.containment_prob(boxes.get(&triple.tail).unwrap(), 1.0)?;

    // Sample 100 negatives from the hard pool.
    let mut rng_h = rand::rngs::StdRng::seed_from_u64(7);
    let hard_negs = generate_negative_samples_from_pool_with_rng(
        &triple,
        &hard_pool_ids,
        &NegativeSamplingStrategy::CorruptTail,
        100,
        &mut rng_h,
    );

    // Sample 100 negatives uniformly from all entity ids (excluding positives by construction).
    let mut all_ids = ids.clone();
    all_ids.shuffle(&mut rng);
    let mut rng_u = rand::rngs::StdRng::seed_from_u64(9);
    let uniform_negs = generate_negative_samples_from_pool_with_rng(
        &triple,
        &all_ids,
        &NegativeSamplingStrategy::CorruptTail,
        100,
        &mut rng_u,
    );

    let score_of = |t: &Triple| -> Result<f32, Box<dyn std::error::Error>> {
        let b = boxes.get(&t.tail).ok_or("missing tail box")?;
        Ok(head_box.containment_prob(b, 1.0)?)
    };

    let hard_scores: Vec<f32> = hard_negs.iter().map(score_of).collect::<Result<_, _>>()?;
    let uni_scores: Vec<f32> = uniform_negs.iter().map(score_of).collect::<Result<_, _>>()?;

    let mean = |xs: &[f32]| xs.iter().sum::<f32>() / (xs.len().max(1) as f32);

    println!("positive tail_score: {tail_score:.6}");
    println!("hard neg mean score: {:.6}", mean(&hard_scores));
    println!("uni  neg mean score: {:.6}", mean(&uni_scores));
    println!(
        "interpretation: hard negatives should usually score higher (closer to positive), \
so they apply more pressure during training."
    );

    Ok(())
}

