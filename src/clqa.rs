//! Conjunctive least-common-ancestor queries over faithful EL++ box embeddings.
//!
//! Given faithful box embeddings (each concept is a box, containment =
//! subsumption), the conjunctive query "find X such that A ⊑ X AND B ⊑ X" has
//! as its tightest certain answer the least common ancestor (LCA) of A and B.
//! This module answers it geometrically, with the readout that empirically
//! recovers the LCA on real ontologies (GALEN): **containment-gated proximity
//! to the join box**.
//!
//! # Why not plain containment
//!
//! The obvious readout, "rank X by the degree that X contains both A and B", is
//! degenerate. Containment saturates: any box large enough to hold both scores
//! ~1, so the ranking is decided by a size penalty rather than by the data, and
//! a single blown-up box wins every query. Measured on GALEN this readout gives
//! a top-1 LCA accuracy near the containment baseline while a single degenerate
//! concept wins 100% of queries.
//!
//! # The join-gated readout
//!
//! The LCA is the *tightest concept box that contains the join* of A and B
//! (`join` = the smallest enclosing box, the `BoxClqa::join` method). Two half-right
//! signals combine into the working one:
//!
//! - **Proximity** to the join is non-saturating and discriminates among
//!   containers, but on its own it also rewards close boxes that do *not*
//!   contain the join (siblings, cousins).
//! - **Containment** of the join is geometrically correct (the LCA must contain
//!   it) but saturates, so it cannot pick the *tightest* container.
//!
//! The `BoxClqa::score_lca` method multiplies a containment gate `exp(-‖join sticks out
//! of X‖)` by a proximity term `exp(-‖Box(X) − join‖₁ / τ)`. The gate
//! suppresses non-containers; proximity then breaks the tie among containers
//! toward the tightest, which is the LCA. On GALEN (dim 200) this lifts top-1
//! LCA accuracy from 0.42 (proximity alone) to 0.60, and beats a plain-KGE
//! (TransE) point baseline on the same conjunctive queries.
//!
//! The embeddings are consumed as flat `centers` / `offsets` slices in the
//! layout produced by the box trainer: concept `c` occupies indices
//! `c*dim .. (c+1)*dim` in each slice.

/// Construction problems [`BoxClqa::new`] rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClqaError {
    /// `dim` is zero.
    ZeroDim,
    /// `centers` and `offsets` differ in length, or the length is not a
    /// multiple of `dim`.
    DimensionMismatch,
}

impl std::fmt::Display for ClqaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDim => write!(f, "dim must be positive"),
            Self::DimensionMismatch => {
                write!(
                    f,
                    "centers and offsets must share a length that is a multiple of dim"
                )
            }
        }
    }
}

impl std::error::Error for ClqaError {}

/// A read-only view over trained faithful box embeddings for answering
/// conjunctive LCA queries. Borrows the flat `centers` / `offsets` slices; does
/// not copy.
#[derive(Debug, Clone, Copy)]
pub struct BoxClqa<'a> {
    centers: &'a [f32],
    offsets: &'a [f32],
    dim: usize,
    n: usize,
}

impl<'a> BoxClqa<'a> {
    /// Build a query view over `n = centers.len() / dim` concept boxes.
    ///
    /// # Errors
    ///
    /// [`ClqaError::ZeroDim`] if `dim == 0`; [`ClqaError::DimensionMismatch`]
    /// if the slices differ in length or the length is not a multiple of `dim`.
    pub fn new(centers: &'a [f32], offsets: &'a [f32], dim: usize) -> Result<Self, ClqaError> {
        if dim == 0 {
            return Err(ClqaError::ZeroDim);
        }
        if centers.len() != offsets.len() || !centers.len().is_multiple_of(dim) {
            return Err(ClqaError::DimensionMismatch);
        }
        Ok(Self {
            centers,
            offsets,
            dim,
            n: centers.len() / dim,
        })
    }

    /// Number of concept boxes.
    pub fn num_concepts(&self) -> usize {
        self.n
    }

    /// The join (least upper bound) box of concepts `a` and `b`: the smallest
    /// axis-aligned box enclosing both. Returned as `(center, offset)`, each of
    /// length `dim`.
    pub fn join(&self, a: usize, b: usize) -> (Vec<f32>, Vec<f32>) {
        let (ao, bo) = (a * self.dim, b * self.dim);
        let mut jc = vec![0f32; self.dim];
        let mut jo = vec![0f32; self.dim];
        for i in 0..self.dim {
            let lo = (self.centers[ao + i] - self.offsets[ao + i])
                .min(self.centers[bo + i] - self.offsets[bo + i]);
            let hi = (self.centers[ao + i] + self.offsets[ao + i])
                .max(self.centers[bo + i] + self.offsets[bo + i]);
            jc[i] = (lo + hi) / 2.0;
            jo[i] = (hi - lo) / 2.0;
        }
        (jc, jo)
    }

    /// Containment-gated proximity of concept `x` to a join box `(jc, jo)`.
    ///
    /// `gate = exp(-‖join sticking out of Box(x)‖₂)` is ~1 when `x` contains the
    /// join and decays as the join protrudes; `proximity = exp(-‖Box(x) −
    /// join‖₁ / τ)` is highest for the box closest to the join. Their product is
    /// maximized by the tightest container, the LCA. `tau` scales the L1
    /// distance; too small collapses toward the saturating pure-containment
    /// behaviour, too large washes out the proximity tie-break.
    pub fn score_lca(&self, jc: &[f32], jo: &[f32], x: usize, tau: f32) -> f32 {
        let xo = x * self.dim;
        let mut incl = 0f32; // squared L2 of the join protruding from Box(x)
        let mut prox = 0f32; // L1 distance of Box(x) to the join box
        for i in 0..self.dim {
            let v = ((jc[i] - self.centers[xo + i]).abs() + jo[i] - self.offsets[xo + i]).max(0.0);
            incl += v * v;
            prox += (self.centers[xo + i] - jc[i]).abs() + (self.offsets[xo + i] - jo[i]).abs();
        }
        let gate = (-incl.sqrt()).exp();
        let proximity = (-prox / tau).exp();
        gate * proximity
    }

    /// Rank every concept as the LCA of `(a, b)`, best first, excluding `a` and
    /// `b` themselves. Each entry is `(concept, score)`; scores are the
    /// [`score_lca`](Self::score_lca) product in `[0, 1]`.
    pub fn rank_lca(&self, a: usize, b: usize, tau: f32) -> Vec<(usize, f32)> {
        self.rank_lca_candidates(a, b, (0..self.n).filter(|&x| x != a && x != b), tau)
    }

    /// Rank a caller-supplied candidate set as the LCA of `(a, b)`, best first.
    ///
    /// This is the scoring half of a two-stage retrieval pipeline: a symbolic
    /// or approximate generator supplies candidate concept ids, and the box
    /// readout scores only those ids. Invalid candidate ids panic in the same
    /// way as [`score_lca`](Self::score_lca).
    pub fn rank_lca_candidates<I>(
        &self,
        a: usize,
        b: usize,
        candidates: I,
        tau: f32,
    ) -> Vec<(usize, f32)>
    where
        I: IntoIterator<Item = usize>,
    {
        let (jc, jo) = self.join(a, b);
        let mut scored: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|x| (x, self.score_lca(&jc, &jo, x, tau)))
            .collect();
        scored.sort_by(|p, r| r.1.partial_cmp(&p.1).unwrap().then_with(|| p.0.cmp(&r.0)));
        scored
    }
}

/// Construction problems [`DirectFrontier`] rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DirectFrontierError {
    /// An edge or query referenced a concept outside `0..num_concepts`.
    InvalidConcept {
        /// The invalid concept id.
        concept: usize,
        /// Number of concepts known to the frontier graph.
        num_concepts: usize,
    },
}

impl std::fmt::Display for DirectFrontierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConcept {
                concept,
                num_concepts,
            } => write!(f, "concept id {concept} is outside 0..{num_concepts}"),
        }
    }
}

impl std::error::Error for DirectFrontierError {}

/// A candidate returned by [`DirectFrontier::pool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrontierCandidate {
    /// Candidate concept id.
    pub concept: usize,
    /// The larger of the upward direct-edge distances from the two query
    /// concepts.
    pub frontier_depth: usize,
    /// Sum of the two upward direct-edge distances. Used as the secondary
    /// ordering key within the same frontier.
    pub path_len: usize,
}

/// A two-anchor DCA/LCA query for candidate-pool evaluation.
///
/// `targets` is the set of acceptable deepest common ancestors for `(a, b)`.
/// Multiple targets are valid in ontologies with multiple inheritance. The
/// candidate-pool metrics treat a query as retrieved when any target appears in
/// the pool.
#[derive(Debug, Clone, Copy)]
pub struct DirectFrontierQuery<'a> {
    /// First query concept.
    pub a: usize,
    /// Second query concept.
    pub b: usize,
    /// Acceptable target concepts, usually the deepest common ancestors.
    pub targets: &'a [usize],
}

impl<'a> DirectFrontierQuery<'a> {
    /// Build a direct-frontier evaluation query.
    pub fn new(a: usize, b: usize, targets: &'a [usize]) -> Self {
        Self { a, b, targets }
    }

    fn is_target(&self, concept: usize) -> bool {
        self.targets.contains(&concept)
    }
}

/// Aggregate retrieval metrics for a ranked DCA/LCA candidate pool.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidatePoolMetrics {
    /// Fraction of queries whose pool contains at least one target.
    pub recall: f64,
    /// Mean number of candidates in the pool.
    pub mean_pool: f32,
    /// 90th percentile pool size.
    pub p90_pool: usize,
    /// Mean reciprocal rank of the first target in the pool ordering.
    ///
    /// Queries whose target is absent contribute `0`.
    pub mrr: f64,
    /// Fraction of queries whose first target is ranked at most 10.
    pub hits10: f64,
    /// Fraction of queries whose first candidate is a target.
    pub top1_target: f64,
}

impl CandidatePoolMetrics {
    /// Zeroed metrics for an empty query set.
    pub fn empty() -> Self {
        Self {
            recall: 0.0,
            mean_pool: 0.0,
            p90_pool: 0,
            mrr: 0.0,
            hits10: 0.0,
            top1_target: 0.0,
        }
    }

    /// Compute metrics from already-ranked candidate pools.
    ///
    /// `ranked_pools[i]` is the ranked candidate concept ids for `queries[i]`.
    /// The first target in a pool determines MRR and Hits@10. Missing targets
    /// count as retrieval misses and contribute zero to ranking metrics.
    ///
    /// # Panics
    ///
    /// Panics if `ranked_pools.len() != queries.len()`.
    pub fn from_ranked_pools<P>(queries: &[DirectFrontierQuery<'_>], ranked_pools: &[P]) -> Self
    where
        P: AsRef<[usize]>,
    {
        assert_eq!(
            queries.len(),
            ranked_pools.len(),
            "queries and ranked_pools must align"
        );
        if queries.is_empty() {
            return Self::empty();
        }

        let mut hits = 0usize;
        let mut reciprocal_rank_sum = 0.0f64;
        let mut hits10 = 0usize;
        let mut top1 = 0usize;
        let mut pool_sizes = Vec::with_capacity(queries.len());

        for (query, pool) in queries.iter().zip(ranked_pools) {
            let pool = pool.as_ref();
            pool_sizes.push(pool.len());
            if let Some(rank0) = pool.iter().position(|&concept| query.is_target(concept)) {
                hits += 1;
                let rank = rank0 + 1;
                reciprocal_rank_sum += 1.0 / rank as f64;
                hits10 += usize::from(rank <= 10);
            }
            top1 += usize::from(
                pool.first()
                    .is_some_and(|&concept| query.is_target(concept)),
            );
        }

        let n = queries.len() as f64;
        Self {
            recall: hits as f64 / n,
            mean_pool: pool_sizes.iter().sum::<usize>() as f32 / pool_sizes.len() as f32,
            p90_pool: quantile_usize(&pool_sizes, 0.90),
            mrr: reciprocal_rank_sum / n,
            hits10: hits10 as f64 / n,
            top1_target: top1 as f64 / n,
        }
    }
}

/// Candidate generator for conjunctive DCA/LCA queries using only direct
/// `SubClassOf` edges.
///
/// For query concepts `(a, b)`, the generator runs upward BFS from both
/// concepts over direct superclass edges, finds the first common frontier, and
/// optionally includes ancestors within `extra_hops` of that frontier. It does
/// not use transitive closure to generate candidates. Closure or labels are
/// only needed by callers that want to evaluate retrieval risk.
///
/// ```
/// use subsume::clqa::DirectFrontier;
///
/// // 0 root, 1 parent, 2/3 leaves.
/// let frontier = DirectFrontier::from_edges(4, [(1, 0), (2, 1), (3, 1)])?;
/// let pool = frontier.pool(2, 3, 0)?;
/// assert_eq!(pool[0].concept, 1);
/// assert_eq!(pool[0].frontier_depth, 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectFrontier {
    parents: Vec<Vec<usize>>,
}

impl DirectFrontier {
    /// Build a direct-frontier generator from `(sub, sup)` direct subclass
    /// edges.
    ///
    /// # Errors
    ///
    /// Returns [`DirectFrontierError::InvalidConcept`] if any edge endpoint is
    /// outside `0..num_concepts`.
    pub fn from_edges<I>(num_concepts: usize, edges: I) -> Result<Self, DirectFrontierError>
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        let mut parents = vec![Vec::new(); num_concepts];
        for (sub, sup) in edges {
            Self::check_concept(sub, num_concepts)?;
            Self::check_concept(sup, num_concepts)?;
            parents[sub].push(sup);
        }
        for ps in &mut parents {
            ps.sort_unstable();
            ps.dedup();
        }
        Ok(Self { parents })
    }

    /// Build from the direct [`Axiom::SubClassOf`](crate::el_training::Axiom::SubClassOf)
    /// edges in an [`Ontology`](crate::el_training::Ontology).
    ///
    /// # Errors
    ///
    /// Returns [`DirectFrontierError::InvalidConcept`] if the ontology's public
    /// fields contain an out-of-range `SubClassOf` endpoint.
    #[cfg(feature = "rand")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
    pub fn from_ontology(
        ontology: &crate::el_training::Ontology,
    ) -> Result<Self, DirectFrontierError> {
        let num_concepts = ontology.concept_names.len();
        let edges = ontology.axioms.iter().filter_map(|axiom| match axiom {
            crate::el_training::Axiom::SubClassOf { sub, sup } => Some((*sub, *sup)),
            _ => None,
        });
        Self::from_edges(num_concepts, edges)
    }

    /// Number of concepts in the frontier graph.
    pub fn num_concepts(&self) -> usize {
        self.parents.len()
    }

    /// Direct superclasses of `concept`.
    ///
    /// # Errors
    ///
    /// Returns [`DirectFrontierError::InvalidConcept`] if `concept` is outside
    /// `0..num_concepts`.
    pub fn parents(&self, concept: usize) -> Result<&[usize], DirectFrontierError> {
        Self::check_concept(concept, self.num_concepts())?;
        Ok(&self.parents[concept])
    }

    /// Generate direct-frontier candidates for `(a, b)`.
    ///
    /// Candidates are ordered by `(frontier_depth, path_len, concept)`.
    /// Increasing `extra_hops` can only add candidates; it does not reorder the
    /// candidates that were already present at smaller values.
    ///
    /// # Errors
    ///
    /// Returns [`DirectFrontierError::InvalidConcept`] if `a` or `b` is outside
    /// `0..num_concepts`.
    pub fn pool(
        &self,
        a: usize,
        b: usize,
        extra_hops: usize,
    ) -> Result<Vec<FrontierCandidate>, DirectFrontierError> {
        let n = self.num_concepts();
        Self::check_concept(a, n)?;
        Self::check_concept(b, n)?;
        let dist_a = self.upward_distances(a);
        let dist_b = self.upward_distances(b);
        let min_frontier = (0..n)
            .filter(|&x| x != a && x != b)
            .filter_map(|x| Some(dist_a[x]?.max(dist_b[x]?)))
            .min();
        let Some(min_frontier) = min_frontier else {
            return Ok(Vec::new());
        };
        let mut pool: Vec<FrontierCandidate> = (0..n)
            .filter(|&x| x != a && x != b)
            .filter_map(|x| {
                let da = dist_a[x]?;
                let db = dist_b[x]?;
                let frontier_depth = da.max(db);
                (frontier_depth <= min_frontier + extra_hops).then_some(FrontierCandidate {
                    concept: x,
                    frontier_depth,
                    path_len: da + db,
                })
            })
            .collect();
        pool.sort_by_key(|c| (c.frontier_depth, c.path_len, c.concept));
        Ok(pool)
    }

    /// Evaluate the retrieval quality of [`pool`](Self::pool) over DCA/LCA
    /// queries.
    ///
    /// This measures only the candidate generator and its deterministic BFS
    /// ordering. It does not score candidates with embeddings. A target missing
    /// from the pool is counted as a retrieval miss and contributes zero to MRR,
    /// Hits@10, and top-1 target accuracy.
    ///
    /// # Errors
    ///
    /// Returns [`DirectFrontierError::InvalidConcept`] if any query concept is
    /// outside `0..num_concepts`.
    pub fn pool_metrics(
        &self,
        queries: &[DirectFrontierQuery<'_>],
        extra_hops: usize,
    ) -> Result<CandidatePoolMetrics, DirectFrontierError> {
        let pools: Vec<Vec<usize>> = queries
            .iter()
            .map(|query| {
                self.pool(query.a, query.b, extra_hops).map(|pool| {
                    pool.into_iter()
                        .map(|candidate| candidate.concept)
                        .collect()
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(CandidatePoolMetrics::from_ranked_pools(queries, &pools))
    }

    fn check_concept(concept: usize, num_concepts: usize) -> Result<(), DirectFrontierError> {
        if concept < num_concepts {
            Ok(())
        } else {
            Err(DirectFrontierError::InvalidConcept {
                concept,
                num_concepts,
            })
        }
    }

    fn upward_distances(&self, start: usize) -> Vec<Option<usize>> {
        let mut dist = vec![None; self.parents.len()];
        let mut queue = std::collections::VecDeque::new();
        dist[start] = Some(0);
        queue.push_back(start);
        while let Some(cur) = queue.pop_front() {
            let next_dist = dist[cur].unwrap() + 1;
            for &parent in &self.parents[cur] {
                if dist[parent].is_none() {
                    dist[parent] = Some(next_dist);
                    queue.push_back(parent);
                }
            }
        }
        dist
    }
}

fn quantile_usize(values: &[usize], q: f32) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) as f32 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 2-D hierarchy: root(0) ⊒ parent(1) ⊒ {leafA(2), leafB(3)}, plus a
    // disjoint sibling(4) far away. Layout is flat: concept c at c*dim..(c+1)*dim.
    //         center            offset
    // 0 root  (0, 0)            (10, 10)   huge
    // 1 par   (0, 0)            ( 2,  2)   tight common ancestor
    // 2 leafA (-1, 0)           (0.5, 0.5) inside par
    // 3 leafB (1, 0)            (0.5, 0.5) inside par
    // 4 sib   (5, 5)            (1, 1)     disjoint, does not contain the join
    fn fixture() -> (Vec<f32>, Vec<f32>, usize) {
        let centers = vec![
            0.0, 0.0, // root
            0.0, 0.0, // par
            -1.0, 0.0, // leafA
            1.0, 0.0, // leafB
            5.0, 5.0, // sib
        ];
        let offsets = vec![
            10.0, 10.0, // root
            2.0, 2.0, // par
            0.5, 0.5, // leafA
            0.5, 0.5, // leafB
            1.0, 1.0, // sib
        ];
        (centers, offsets, 2)
    }

    #[test]
    fn join_is_smallest_enclosing_box() {
        let (c, o, dim) = fixture();
        let q = BoxClqa::new(&c, &o, dim).unwrap();
        // join(leafA, leafB): x spans [-1.5, 1.5], y spans [-0.5, 0.5].
        let (jc, jo) = q.join(2, 3);
        assert!(
            (jc[0] - 0.0).abs() < 1e-6 && (jc[1] - 0.0).abs() < 1e-6,
            "{jc:?}"
        );
        assert!((jo[0] - 1.5).abs() < 1e-6, "x half-width: {jo:?}");
        assert!((jo[1] - 0.5).abs() < 1e-6, "y half-width: {jo:?}");
    }

    #[test]
    fn lca_is_the_tightest_container_not_the_root() {
        let (c, o, dim) = fixture();
        let q = BoxClqa::new(&c, &o, dim).unwrap();
        let ranked = q.rank_lca(2, 3, 1.0);
        // The parent (1) is the LCA: it contains the join and is tighter than
        // the root (0). Both contain the join, so a saturating containment
        // readout would tie or prefer the root; gated proximity must pick par.
        assert_eq!(ranked[0].0, 1, "ranking: {ranked:?}");
        // The disjoint sibling (4) does not contain the join and must rank below
        // both containers.
        let par_pos = ranked.iter().position(|&(x, _)| x == 1).unwrap();
        let sib_pos = ranked.iter().position(|&(x, _)| x == 4).unwrap();
        assert!(
            par_pos < sib_pos,
            "container must outrank non-container: {ranked:?}"
        );
    }

    #[test]
    fn rank_lca_candidates_scores_only_supplied_pool() {
        let (c, o, dim) = fixture();
        let q = BoxClqa::new(&c, &o, dim).unwrap();
        let ranked = q.rank_lca_candidates(2, 3, [4, 0, 1], 1.0);
        assert_eq!(
            ranked.iter().map(|&(x, _)| x).collect::<Vec<_>>(),
            [1, 0, 4]
        );
    }

    #[test]
    fn gate_penalizes_non_containers() {
        let (c, o, dim) = fixture();
        let q = BoxClqa::new(&c, &o, dim).unwrap();
        let (jc, jo) = q.join(2, 3);
        // par contains the join (gate ~1); sib does not (gate < 1). Even before
        // proximity, the container must score strictly higher.
        let s_par = q.score_lca(&jc, &jo, 1, 1.0);
        let s_sib = q.score_lca(&jc, &jo, 4, 1.0);
        assert!(s_par > s_sib, "par {s_par} should beat sib {s_sib}");
        assert!((0.0..=1.0).contains(&s_par) && (0.0..=1.0).contains(&s_sib));
    }

    #[test]
    fn rejects_bad_dimensions() {
        assert_eq!(
            BoxClqa::new(&[1.0], &[1.0], 0).unwrap_err(),
            ClqaError::ZeroDim
        );
        assert_eq!(
            BoxClqa::new(&[1.0, 2.0], &[1.0], 2).unwrap_err(),
            ClqaError::DimensionMismatch
        );
        assert_eq!(
            BoxClqa::new(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 2).unwrap_err(),
            ClqaError::DimensionMismatch
        );
    }

    #[test]
    fn direct_frontier_returns_first_common_frontier() {
        // 0 root, 1 parent, 2/3 leaves.
        let frontier = DirectFrontier::from_edges(4, [(1, 0), (2, 1), (3, 1)]).unwrap();
        let pool0 = frontier.pool(2, 3, 0).unwrap();
        assert_eq!(
            pool0,
            vec![FrontierCandidate {
                concept: 1,
                frontier_depth: 1,
                path_len: 2,
            }]
        );
        let pool1 = frontier.pool(2, 3, 1).unwrap();
        assert_eq!(
            pool1,
            vec![
                FrontierCandidate {
                    concept: 1,
                    frontier_depth: 1,
                    path_len: 2,
                },
                FrontierCandidate {
                    concept: 0,
                    frontier_depth: 2,
                    path_len: 4,
                },
            ]
        );
    }

    #[test]
    fn direct_frontier_keeps_plural_deepest_ancestors() {
        // 4 and 5 share two direct parents (1 and 2), both under root 0.
        let frontier =
            DirectFrontier::from_edges(6, [(1, 0), (2, 0), (4, 1), (4, 2), (5, 1), (5, 2)])
                .unwrap();
        let pool = frontier.pool(4, 5, 0).unwrap();
        assert_eq!(
            pool,
            vec![
                FrontierCandidate {
                    concept: 1,
                    frontier_depth: 1,
                    path_len: 2,
                },
                FrontierCandidate {
                    concept: 2,
                    frontier_depth: 1,
                    path_len: 2,
                },
            ]
        );
    }

    #[test]
    fn direct_frontier_extra_hops_is_monotone() {
        let frontier = DirectFrontier::from_edges(5, [(1, 0), (2, 1), (3, 1), (4, 0)]).unwrap();
        let pool0 = frontier.pool(2, 3, 0).unwrap();
        let pool2 = frontier.pool(2, 3, 2).unwrap();
        assert!(pool2.starts_with(&pool0));
        assert!(pool2.len() >= pool0.len());
    }

    #[test]
    fn direct_frontier_handles_cycles() {
        let frontier = DirectFrontier::from_edges(4, [(0, 1), (1, 0), (2, 0), (3, 1)]).unwrap();
        let pool = frontier.pool(2, 3, 10).unwrap();
        assert!(pool.iter().any(|c| c.concept == 0));
        assert!(pool.iter().any(|c| c.concept == 1));
    }

    #[test]
    fn direct_frontier_rejects_invalid_concepts() {
        assert_eq!(
            DirectFrontier::from_edges(2, [(0, 2)]).unwrap_err(),
            DirectFrontierError::InvalidConcept {
                concept: 2,
                num_concepts: 2,
            }
        );
        let frontier = DirectFrontier::from_edges(2, [(0, 1)]).unwrap();
        assert_eq!(
            frontier.pool(0, 2, 0).unwrap_err(),
            DirectFrontierError::InvalidConcept {
                concept: 2,
                num_concepts: 2,
            }
        );
    }

    #[test]
    fn direct_frontier_pool_metrics_count_plural_targets() {
        let frontier =
            DirectFrontier::from_edges(6, [(1, 0), (2, 0), (4, 1), (4, 2), (5, 1), (5, 2)])
                .unwrap();
        let targets = [1, 2];
        let queries = [DirectFrontierQuery::new(4, 5, &targets)];
        let metrics = frontier.pool_metrics(&queries, 0).unwrap();
        assert_eq!(metrics.recall, 1.0);
        assert_eq!(metrics.mean_pool, 2.0);
        assert_eq!(metrics.p90_pool, 2);
        assert_eq!(metrics.mrr, 1.0);
        assert_eq!(metrics.hits10, 1.0);
        assert_eq!(metrics.top1_target, 1.0);
    }

    #[test]
    fn direct_frontier_pool_metrics_missing_target_is_retrieval_loss() {
        let frontier = DirectFrontier::from_edges(4, [(1, 0), (2, 1), (3, 1)]).unwrap();
        let targets = [99];
        let queries = [DirectFrontierQuery::new(2, 3, &targets)];
        let metrics = frontier.pool_metrics(&queries, 1).unwrap();
        assert_eq!(metrics.recall, 0.0);
        assert_eq!(metrics.mean_pool, 2.0);
        assert_eq!(metrics.p90_pool, 2);
        assert_eq!(metrics.mrr, 0.0);
        assert_eq!(metrics.hits10, 0.0);
        assert_eq!(metrics.top1_target, 0.0);
    }

    #[test]
    fn candidate_pool_metrics_respect_external_ranking() {
        let targets0 = [3];
        let targets1 = [4];
        let queries = [
            DirectFrontierQuery::new(0, 1, &targets0),
            DirectFrontierQuery::new(1, 2, &targets1),
        ];
        let ranked_pools = [vec![2, 3], vec![4, 3, 2]];
        let metrics = CandidatePoolMetrics::from_ranked_pools(&queries, &ranked_pools);
        assert_eq!(metrics.recall, 1.0);
        assert_eq!(metrics.mean_pool, 2.5);
        assert_eq!(metrics.p90_pool, 3);
        assert_eq!(metrics.mrr, 0.75);
        assert_eq!(metrics.hits10, 1.0);
        assert_eq!(metrics.top1_target, 0.5);
    }

    #[test]
    fn direct_frontier_pool_metrics_empty_query_set_is_zeroed() {
        let frontier = DirectFrontier::from_edges(2, [(0, 1)]).unwrap();
        let metrics = frontier.pool_metrics(&[], 0).unwrap();
        assert_eq!(
            metrics,
            CandidatePoolMetrics {
                recall: 0.0,
                mean_pool: 0.0,
                p90_pool: 0,
                mrr: 0.0,
                hits10: 0.0,
                top1_target: 0.0,
            }
        );
    }
}
