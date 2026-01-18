use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

use subsume_core::dataset::Triple;
use subsume_core::trainer::{generate_negative_samples_from_pool_with_rng, NegativeSamplingStrategy};
use subsume_core::Box as CoreBox;
use subsume_ndarray::NdarrayBox;
use walk::{personalized_pagerank, Graph, PageRankConfig};

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
}

impl Graph for Adj {
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

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / (xs.len().max(1) as f32)
}

#[test]
fn walk_ppr_candidates_are_harder_than_uniform_negatives_smoke() -> Result<(), Box<dyn std::error::Error>> {
    // Deterministic toy setting:
    // - SBM graph with 2 blocks
    // - Boxes clustered by block so within-block containment is generally higher
    let g = Adj::sbm_two_block(200, 0.08, 0.005, 123);
    let n = g.node_count();

    let ids: Vec<String> = (0..n).map(|i| format!("e{i:03}")).collect();

    // Boxes (deterministic RNG): block A near -1, block B near +1.
    let d = 8usize;
    let mut rng = rand::rngs::StdRng::seed_from_u64(999);
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

    // Pick a positive (head in A; tail is an A-neighbor if available).
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

    // PPR candidate pool (top-K excluding head/tail).
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

    let k_pool = 50usize;
    let mut ranked: Vec<usize> = (0..n).collect();
    ranked.sort_by(|&a, &b| ppr[b].total_cmp(&ppr[a]).then_with(|| a.cmp(&b)));
    let hard_pool_ids: Vec<String> = ranked
        .into_iter()
        .filter(|&j| j != head_i && j != tail_i)
        .take(k_pool)
        .map(|j| ids[j].clone())
        .collect();

    // Sample negatives (deterministic).
    let mut rng_h = rand::rngs::StdRng::seed_from_u64(7);
    let hard_negs = generate_negative_samples_from_pool_with_rng(
        &triple,
        &hard_pool_ids,
        &NegativeSamplingStrategy::CorruptTail,
        100,
        &mut rng_h,
    );

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

    let head_box = boxes.get(&triple.head).ok_or("missing head box")?;
    let score_of = |t: &Triple| -> Result<f32, Box<dyn std::error::Error>> {
        let b = boxes.get(&t.tail).ok_or("missing tail box")?;
        Ok(head_box.containment_prob(b, 1.0)?)
    };

    let hard_scores: Vec<f32> = hard_negs.iter().map(score_of).collect::<Result<_, _>>()?;
    let uni_scores: Vec<f32> = uniform_negs.iter().map(score_of).collect::<Result<_, _>>()?;

    // The core qualitative claim:
    // “walk-near nodes are harder negatives” ⇒ higher containment score on average.
    let hard_mean = mean(&hard_scores);
    let uni_mean = mean(&uni_scores);

    assert!(
        hard_mean > uni_mean,
        "expected hard negatives to score higher on average (hard_mean={hard_mean}, uni_mean={uni_mean})"
    );

    Ok(())
}

