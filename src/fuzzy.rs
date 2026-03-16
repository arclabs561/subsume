//! Fuzzy set-theoretic operators: t-norms, t-conorms, and negation.
//!
//! These operators extend crisp set operations (intersection, union, complement)
//! to fuzzy membership values in \[0, 1\]. They are the foundation of fuzzy
//! query answering over knowledge graphs.
//!
//! # T-norms (fuzzy intersection)
//!
//! A t-norm `T: [0,1]^2 -> [0,1]` must satisfy:
//! - Commutativity: `T(a, b) = T(b, a)`
//! - Associativity: `T(a, T(b, c)) = T(T(a, b), c)`
//! - Monotonicity: if `a <= c` then `T(a, b) <= T(c, b)`
//! - Identity: `T(a, 1) = a`
//! - Annihilator: `T(a, 0) = 0`
//!
//! # T-conorms (fuzzy union)
//!
//! Each t-norm has a dual t-conorm via De Morgan's law:
//! `S(a, b) = 1 - T(1-a, 1-b)`.
//!
//! # References
//!
//! - Chen et al. (AAAI 2022), "Fuzzy Logic Based Logical Query Answering on
//!   Knowledge Graphs" (FuzzQE)
//!
//! # Examples
//!
//! ```rust
//! use subsume::fuzzy::{TNorm, TConorm, fuzzy_negation};
//!
//! // Fuzzy intersection: how "aquatic" AND "mammal" is a dolphin?
//! let aquatic = 0.9;
//! let mammal = 0.95;
//!
//! let min = TNorm::Min.apply(aquatic, mammal);       // 0.9
//! let prod = TNorm::Product.apply(aquatic, mammal);   // 0.855
//! let luk = TNorm::Lukasiewicz.apply(aquatic, mammal); // 0.85
//! assert!(min >= prod && prod >= luk); // Min >= Product >= Lukasiewicz
//!
//! // De Morgan duality: neg(T(a,b)) = S(neg(a), neg(b))
//! let t = TNorm::Product;
//! let s = t.dual(); // TConorm::Probabilistic
//! let lhs = fuzzy_negation(t.apply(0.7, 0.4));
//! let rhs = s.apply(fuzzy_negation(0.7), fuzzy_negation(0.4));
//! assert!((lhs - rhs).abs() < 1e-6);
//! ```

/// Standard fuzzy negation: `1 - a`.
#[inline]
#[must_use]
pub fn fuzzy_negation(a: f32) -> f32 {
    1.0 - a
}

/// A triangular norm (t-norm) for fuzzy intersection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TNorm {
    /// Godel t-norm: `min(a, b)`.
    Min,
    /// Product t-norm: `a * b`.
    Product,
    /// Lukasiewicz t-norm: `max(a + b - 1, 0)`.
    Lukasiewicz,
}

impl TNorm {
    /// Apply this t-norm to two fuzzy values.
    #[inline]
    #[must_use]
    pub fn apply(&self, a: f32, b: f32) -> f32 {
        match self {
            TNorm::Min => a.min(b),
            TNorm::Product => a * b,
            TNorm::Lukasiewicz => (a + b - 1.0).max(0.0),
        }
    }

    /// Get the dual t-conorm (via De Morgan).
    #[inline]
    #[must_use]
    pub fn dual(&self) -> TConorm {
        match self {
            TNorm::Min => TConorm::Max,
            TNorm::Product => TConorm::Probabilistic,
            TNorm::Lukasiewicz => TConorm::Lukasiewicz,
        }
    }
}

/// A triangular conorm (t-conorm) for fuzzy union.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TConorm {
    /// Godel t-conorm: `max(a, b)`.
    Max,
    /// Probabilistic t-conorm: `a + b - a*b`.
    Probabilistic,
    /// Lukasiewicz t-conorm: `min(a + b, 1)`.
    Lukasiewicz,
}

impl TConorm {
    /// Apply this t-conorm to two fuzzy values.
    #[inline]
    #[must_use]
    pub fn apply(&self, a: f32, b: f32) -> f32 {
        match self {
            TConorm::Max => a.max(b),
            TConorm::Probabilistic => a + b - a * b,
            TConorm::Lukasiewicz => (a + b).min(1.0),
        }
    }

    /// Get the dual t-norm (via De Morgan).
    #[inline]
    #[must_use]
    pub fn dual(&self) -> TNorm {
        match self {
            TConorm::Max => TNorm::Min,
            TConorm::Probabilistic => TNorm::Product,
            TConorm::Lukasiewicz => TNorm::Lukasiewicz,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ---- Unit tests ----

    #[test]
    fn tnorm_min_basic() {
        assert_eq!(TNorm::Min.apply(0.3, 0.7), 0.3);
        assert_eq!(TNorm::Min.apply(0.5, 0.5), 0.5);
        assert_eq!(TNorm::Min.apply(1.0, 0.4), 0.4);
        assert_eq!(TNorm::Min.apply(0.0, 0.9), 0.0);
    }

    #[test]
    fn tnorm_product_basic() {
        assert!((TNorm::Product.apply(0.5, 0.5) - 0.25).abs() < 1e-7);
        assert_eq!(TNorm::Product.apply(1.0, 0.7), 0.7);
        assert_eq!(TNorm::Product.apply(0.0, 0.5), 0.0);
    }

    #[test]
    fn tnorm_lukasiewicz_basic() {
        assert_eq!(TNorm::Lukasiewicz.apply(0.3, 0.5), 0.0); // 0.3+0.5-1 = -0.2 -> 0
        assert!((TNorm::Lukasiewicz.apply(0.8, 0.9) - 0.7).abs() < 1e-7);
        assert_eq!(TNorm::Lukasiewicz.apply(1.0, 1.0), 1.0);
        assert_eq!(TNorm::Lukasiewicz.apply(0.0, 1.0), 0.0);
    }

    #[test]
    fn tconorm_max_basic() {
        assert_eq!(TConorm::Max.apply(0.3, 0.7), 0.7);
        assert_eq!(TConorm::Max.apply(0.0, 0.0), 0.0);
        assert_eq!(TConorm::Max.apply(1.0, 0.5), 1.0);
    }

    #[test]
    fn tconorm_probabilistic_basic() {
        // 0.3 + 0.7 - 0.21 = 0.79
        assert!((TConorm::Probabilistic.apply(0.3, 0.7) - 0.79).abs() < 1e-6);
        assert_eq!(TConorm::Probabilistic.apply(0.0, 0.5), 0.5);
        assert_eq!(TConorm::Probabilistic.apply(1.0, 0.5), 1.0);
    }

    #[test]
    fn tconorm_lukasiewicz_basic() {
        assert!((TConorm::Lukasiewicz.apply(0.3, 0.5) - 0.8).abs() < 1e-7);
        assert_eq!(TConorm::Lukasiewicz.apply(0.6, 0.7), 1.0); // 1.3 clamped
        assert_eq!(TConorm::Lukasiewicz.apply(0.0, 0.0), 0.0);
    }

    #[test]
    fn fuzzy_negation_basic() {
        assert_eq!(fuzzy_negation(0.0), 1.0);
        assert_eq!(fuzzy_negation(1.0), 0.0);
        assert!((fuzzy_negation(0.3) - 0.7).abs() < 1e-7);
    }

    #[test]
    fn dual_roundtrip() {
        assert_eq!(TNorm::Min.dual().dual(), TNorm::Min);
        assert_eq!(TNorm::Product.dual().dual(), TNorm::Product);
        assert_eq!(TNorm::Lukasiewicz.dual().dual(), TNorm::Lukasiewicz);
    }

    // ---- Property tests ----

    fn arb_unit() -> impl Strategy<Value = f32> {
        0.0f32..=1.0
    }

    proptest! {
        // -- Commutativity --

        #[test]
        fn prop_tnorm_min_commutative(a in arb_unit(), b in arb_unit()) {
            prop_assert_eq!(TNorm::Min.apply(a, b), TNorm::Min.apply(b, a));
        }

        #[test]
        fn prop_tnorm_product_commutative(a in arb_unit(), b in arb_unit()) {
            prop_assert!((TNorm::Product.apply(a, b) - TNorm::Product.apply(b, a)).abs() < 1e-6);
        }

        #[test]
        fn prop_tnorm_lukasiewicz_commutative(a in arb_unit(), b in arb_unit()) {
            prop_assert!((TNorm::Lukasiewicz.apply(a, b) - TNorm::Lukasiewicz.apply(b, a)).abs() < 1e-6);
        }

        // -- Associativity --

        #[test]
        fn prop_tnorm_min_associative(a in arb_unit(), b in arb_unit(), c in arb_unit()) {
            let lhs = TNorm::Min.apply(a, TNorm::Min.apply(b, c));
            let rhs = TNorm::Min.apply(TNorm::Min.apply(a, b), c);
            prop_assert!((lhs - rhs).abs() < 1e-6);
        }

        #[test]
        fn prop_tnorm_product_associative(a in arb_unit(), b in arb_unit(), c in arb_unit()) {
            let lhs = TNorm::Product.apply(a, TNorm::Product.apply(b, c));
            let rhs = TNorm::Product.apply(TNorm::Product.apply(a, b), c);
            prop_assert!((lhs - rhs).abs() < 1e-5);
        }

        #[test]
        fn prop_tnorm_lukasiewicz_associative(a in arb_unit(), b in arb_unit(), c in arb_unit()) {
            let lhs = TNorm::Lukasiewicz.apply(a, TNorm::Lukasiewicz.apply(b, c));
            let rhs = TNorm::Lukasiewicz.apply(TNorm::Lukasiewicz.apply(a, b), c);
            prop_assert!((lhs - rhs).abs() < 1e-6);
        }

        // -- Boundary conditions --

        #[test]
        fn prop_tnorm_identity(a in arb_unit()) {
            // T(a, 1) = a for all t-norms.
            prop_assert!((TNorm::Min.apply(a, 1.0) - a).abs() < 1e-6);
            prop_assert!((TNorm::Product.apply(a, 1.0) - a).abs() < 1e-6);
            prop_assert!((TNorm::Lukasiewicz.apply(a, 1.0) - a).abs() < 1e-6);
        }

        #[test]
        fn prop_tnorm_annihilator(a in arb_unit()) {
            // T(a, 0) = 0 for all t-norms.
            prop_assert!((TNorm::Min.apply(a, 0.0)).abs() < 1e-6);
            prop_assert!((TNorm::Product.apply(a, 0.0)).abs() < 1e-6);
            prop_assert!((TNorm::Lukasiewicz.apply(a, 0.0)).abs() < 1e-6);
        }

        // -- De Morgan: neg(T(a,b)) = S(neg(a), neg(b)) --

        #[test]
        fn prop_de_morgan_min(a in arb_unit(), b in arb_unit()) {
            let lhs = fuzzy_negation(TNorm::Min.apply(a, b));
            let rhs = TConorm::Max.apply(fuzzy_negation(a), fuzzy_negation(b));
            prop_assert!((lhs - rhs).abs() < 1e-6,
                "De Morgan min: {lhs} != {rhs}");
        }

        #[test]
        fn prop_de_morgan_product(a in arb_unit(), b in arb_unit()) {
            let lhs = fuzzy_negation(TNorm::Product.apply(a, b));
            let rhs = TConorm::Probabilistic.apply(fuzzy_negation(a), fuzzy_negation(b));
            prop_assert!((lhs - rhs).abs() < 1e-5,
                "De Morgan product: {lhs} != {rhs}");
        }

        #[test]
        fn prop_de_morgan_lukasiewicz(a in arb_unit(), b in arb_unit()) {
            let lhs = fuzzy_negation(TNorm::Lukasiewicz.apply(a, b));
            let rhs = TConorm::Lukasiewicz.apply(fuzzy_negation(a), fuzzy_negation(b));
            prop_assert!((lhs - rhs).abs() < 1e-6,
                "De Morgan Lukasiewicz: {lhs} != {rhs}");
        }

        // -- Output range --

        #[test]
        fn prop_tnorm_output_in_unit(a in arb_unit(), b in arb_unit()) {
            for t in [TNorm::Min, TNorm::Product, TNorm::Lukasiewicz] {
                let v = t.apply(a, b);
                prop_assert!((-1e-7..=1.0 + 1e-7).contains(&v),
                    "t-norm {:?} output {v} out of [0,1]", t);
            }
        }

        #[test]
        fn prop_tconorm_output_in_unit(a in arb_unit(), b in arb_unit()) {
            for s in [TConorm::Max, TConorm::Probabilistic, TConorm::Lukasiewicz] {
                let v = s.apply(a, b);
                prop_assert!((-1e-7..=1.0 + 1e-7).contains(&v),
                    "t-conorm {:?} output {v} out of [0,1]", s);
            }
        }

        // -- Double negation --

        #[test]
        fn prop_double_negation(a in arb_unit()) {
            let result = fuzzy_negation(fuzzy_negation(a));
            prop_assert!((result - a).abs() < 1e-6, "double negation: {result} != {a}");
        }
    }
}
