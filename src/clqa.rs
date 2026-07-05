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
//! (`join` = the smallest enclosing box, [`BoxClqa::join`]). Two half-right
//! signals combine into the working one:
//!
//! - **Proximity** to the join is non-saturating and discriminates among
//!   containers, but on its own it also rewards close boxes that do *not*
//!   contain the join (siblings, cousins).
//! - **Containment** of the join is geometrically correct (the LCA must contain
//!   it) but saturates, so it cannot pick the *tightest* container.
//!
//! [`BoxClqa::score_lca`] multiplies a containment gate `exp(-‖join sticks out
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
        let (jc, jo) = self.join(a, b);
        let mut scored: Vec<(usize, f32)> = (0..self.n)
            .filter(|&x| x != a && x != b)
            .map(|x| (x, self.score_lca(&jc, &jo, x, tau)))
            .collect();
        scored.sort_by(|p, r| r.1.partial_cmp(&p.1).unwrap());
        scored
    }
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
}
