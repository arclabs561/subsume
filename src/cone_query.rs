//! Composable cone query operators for first-order logical query answering.
//!
//! Provides a query algebra over [`NdarrayCone`] embeddings supporting
//! conjunction (intersection), disjunction (union), negation (complement),
//! and relation projection -- the four operators for answering FOL queries
//! over knowledge graphs.
//!
//! # References
//!
//! - Zhang et al. (NeurIPS 2021), "ConE: Cone Embeddings for Multi-Hop Reasoning"
//! - Chen et al. (AAAI 2022), "FuzzQE: Fuzzy Logic Based Logical Query Answering"

use crate::cone::ConeError;
use crate::ndarray_backend::NdarrayCone;

/// A composable first-order logic query over cone embeddings.
///
/// Queries form a tree that evaluates bottom-up to produce [`NdarrayCone`] results.
#[derive(Debug, Clone)]
pub enum ConeQuery {
    /// A leaf node: a single cone (entity or concept embedding).
    Atom(NdarrayCone),

    /// Relation projection: transform a query cone through a relation.
    Projection {
        /// The input query to project.
        query: Box<ConeQuery>,
        /// Per-dimension relation axis offsets.
        relation_axes: ndarray::Array1<f32>,
        /// Per-dimension relation aperture offsets.
        relation_apertures: ndarray::Array1<f32>,
    },

    /// Conjunction (AND): intersection of multiple query cones.
    Intersection(Vec<ConeQuery>),

    /// Disjunction (OR): union via DNF (min distance across components).
    Union(Vec<ConeQuery>),

    /// Negation (NOT): complement of a query cone.
    Complement(Box<ConeQuery>),
}

impl ConeQuery {
    /// Evaluate this query to a single cone.
    ///
    /// For [`Union`](ConeQuery::Union), returns the first component. Use
    /// [`score_entity`](ConeQuery::score_entity) for correct union scoring.
    ///
    /// # Errors
    ///
    /// Returns [`ConeError`] on dimension mismatch or empty operand lists.
    pub fn evaluate(&self) -> Result<NdarrayCone, ConeError> {
        match self {
            ConeQuery::Atom(cone) => Ok(cone.clone()),

            ConeQuery::Projection {
                query,
                relation_axes,
                relation_apertures,
            } => {
                let base = query.evaluate()?;
                base.project(relation_axes, relation_apertures)
            }

            ConeQuery::Intersection(operands) => {
                if operands.is_empty() {
                    return Err(ConeError::InvalidBounds {
                        reason: "intersection requires at least one operand".into(),
                    });
                }
                let mut result = operands[0].evaluate()?;
                for op in &operands[1..] {
                    let cone = op.evaluate()?;
                    result = result.intersection(&cone)?;
                }
                Ok(result)
            }

            ConeQuery::Union(operands) => {
                if operands.is_empty() {
                    return Err(ConeError::InvalidBounds {
                        reason: "union requires at least one operand".into(),
                    });
                }
                operands[0].evaluate()
            }

            ConeQuery::Complement(inner) => {
                let cone = inner.evaluate()?;
                Ok(cone.complement())
            }
        }
    }

    /// Score an entity against this query (lower = better match).
    ///
    /// For union queries, returns the minimum distance across all components.
    pub fn score_entity(&self, entity: &NdarrayCone, cen: f32) -> Result<f32, ConeError> {
        match self {
            ConeQuery::Union(operands) => {
                if operands.is_empty() {
                    return Err(ConeError::InvalidBounds {
                        reason: "union requires at least one operand".into(),
                    });
                }
                let mut min_dist = f32::INFINITY;
                for op in operands {
                    let cone = op.evaluate()?;
                    let d = cone.cone_distance(entity, cen)?;
                    if d < min_dist {
                        min_dist = d;
                    }
                }
                Ok(min_dist)
            }
            _ => {
                let cone = self.evaluate()?;
                cone.cone_distance(entity, cen)
            }
        }
    }

    /// Score a batch of entities against this query.
    pub fn score_entities(
        &self,
        entities: &[NdarrayCone],
        cen: f32,
        out: &mut [f32],
    ) -> Result<(), ConeError> {
        assert_eq!(entities.len(), out.len());

        match self {
            ConeQuery::Union(operands) => {
                if operands.is_empty() {
                    return Err(ConeError::InvalidBounds {
                        reason: "union requires at least one operand".into(),
                    });
                }
                for v in out.iter_mut() {
                    *v = f32::INFINITY;
                }
                for op in operands {
                    let cone = op.evaluate()?;
                    for (i, entity) in entities.iter().enumerate() {
                        let d = cone.cone_distance(entity, cen)?;
                        if d < out[i] {
                            out[i] = d;
                        }
                    }
                }
                Ok(())
            }
            _ => {
                let cone = self.evaluate()?;
                for (i, entity) in entities.iter().enumerate() {
                    out[i] = cone.cone_distance(entity, cen)?;
                }
                Ok(())
            }
        }
    }

    /// Rank entities by distance (ascending: best matches first).
    pub fn rank_entities(
        &self,
        entities: &[NdarrayCone],
        cen: f32,
    ) -> Result<Vec<usize>, ConeError> {
        let mut scores = vec![0.0f32; entities.len()];
        self.score_entities(entities, cen, &mut scores)?;
        let mut indices: Vec<usize> = (0..entities.len()).collect();
        indices.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(indices)
    }
}

/// Convert a ConE distance to a fuzzy containment score in \[0, 1\].
///
/// Uses sigmoid: `score = 1 / (1 + exp(gamma * distance))`.
/// Bridges cone distances to the [`fuzzy`](crate::fuzzy) module.
pub fn cone_containment_score(
    query: &NdarrayCone,
    entity: &NdarrayCone,
    cen: f32,
    gamma: f32,
) -> Result<f32, ConeError> {
    let dist = query.cone_distance(entity, cen)?;
    Ok(1.0 / (1.0 + (gamma * dist).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f32::consts::PI;

    const CEN: f32 = 0.02;

    fn make_cone(axes: &[f32], apertures: &[f32]) -> NdarrayCone {
        NdarrayCone::new(
            ndarray::Array1::from_vec(axes.to_vec()),
            ndarray::Array1::from_vec(apertures.to_vec()),
        )
        .unwrap()
    }

    #[test]
    fn atom_evaluates_to_itself() {
        let cone = make_cone(&[0.5, -0.3], &[1.0, 0.8]);
        let query = ConeQuery::Atom(cone.clone());
        let result = query.evaluate().unwrap();
        for i in 0..cone.dim() {
            assert!((result.axes()[i] - cone.axes()[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn complement_query_matches_cone_complement() {
        let cone = make_cone(&[0.5, -0.3], &[0.8, 1.2]);
        let query = ConeQuery::Complement(Box::new(ConeQuery::Atom(cone.clone())));
        let result = query.evaluate().unwrap();
        let direct = cone.complement();
        for i in 0..cone.dim() {
            assert!((result.axes()[i] - direct.axes()[i]).abs() < 1e-6);
            assert!((result.apertures()[i] - direct.apertures()[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn intersection_takes_min_aperture() {
        let a = make_cone(&[0.0, 0.0], &[1.0, 0.5]);
        let b = make_cone(&[0.0, 0.0], &[0.3, 0.8]);
        let query = ConeQuery::Intersection(vec![ConeQuery::Atom(a), ConeQuery::Atom(b)]);
        let result = query.evaluate().unwrap();
        assert!((result.apertures()[0] - 0.3).abs() < 1e-6);
        assert!((result.apertures()[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn intersection_empty_errors() {
        assert!(ConeQuery::Intersection(vec![]).evaluate().is_err());
    }

    #[test]
    fn union_score_takes_minimum_distance() {
        let a = make_cone(&[0.0, 0.0], &[1.0, 1.0]);
        let b = make_cone(&[2.0, 2.0], &[1.0, 1.0]);
        let entity_near_a = NdarrayCone::point(array![0.1, 0.1]);
        let entity_near_b = NdarrayCone::point(array![2.1, 2.1]);
        let query = ConeQuery::Union(vec![ConeQuery::Atom(a), ConeQuery::Atom(b)]);
        let d_a = query.score_entity(&entity_near_a, CEN).unwrap();
        let d_b = query.score_entity(&entity_near_b, CEN).unwrap();
        assert!(d_a < 0.5, "entity near a should score well: {d_a}");
        assert!(d_b < 0.5, "entity near b should score well: {d_b}");
    }

    #[test]
    fn projection_shifts_axes() {
        let cone = make_cone(&[0.0, 0.0], &[0.5, 0.5]);
        let query = ConeQuery::Projection {
            query: Box::new(ConeQuery::Atom(cone)),
            relation_axes: array![0.5, -0.3],
            relation_apertures: array![0.0, 0.0],
        };
        let result = query.evaluate().unwrap();
        assert!((result.axes()[0] - 0.5).abs() < 1e-6);
        assert!((result.axes()[1] - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn containment_score_inside_higher_than_outside() {
        let query = make_cone(&[0.0, 0.0], &[2.0, 2.0]);
        let inside = NdarrayCone::point(array![0.1, 0.1]);
        let outside = NdarrayCone::point(array![PI, PI]);
        let score_in = cone_containment_score(&query, &inside, 0.02, 10.0).unwrap();
        let score_out = cone_containment_score(&query, &outside, 0.02, 10.0).unwrap();
        assert!(
            score_in > score_out,
            "inside ({score_in}) > outside ({score_out})"
        );
    }

    #[test]
    fn rank_entities_orders_by_distance() {
        let query_cone = make_cone(&[0.0, 0.0], &[1.0, 1.0]);
        let near = NdarrayCone::point(array![0.1, 0.1]);
        let far = NdarrayCone::point(array![PI, PI]);
        let query = ConeQuery::Atom(query_cone);
        let entities = vec![far, near]; // out of order
        let ranked = query.rank_entities(&entities, CEN).unwrap();
        assert_eq!(ranked[0], 1, "nearest should be first");
    }

    #[test]
    fn animal_and_not_cat_prefers_dog() {
        let animal = make_cone(&[0.0, 0.0], &[2.0, 2.0]);
        let cat = make_cone(&[-0.5, -0.5], &[0.5, 0.5]);
        let dog = NdarrayCone::point(array![0.5, 0.5]);
        let kitten = NdarrayCone::point(array![-0.4, -0.4]);
        let query = ConeQuery::Intersection(vec![
            ConeQuery::Atom(animal),
            ConeQuery::Complement(Box::new(ConeQuery::Atom(cat))),
        ]);
        let dog_dist = query.score_entity(&dog, CEN).unwrap();
        let kitten_dist = query.score_entity(&kitten, CEN).unwrap();
        assert!(
            dog_dist < kitten_dist,
            "dog ({dog_dist}) < kitten ({kitten_dist})"
        );
    }

    #[test]
    fn two_hop_projection() {
        let start = make_cone(&[0.0, 0.0], &[0.5, 0.5]);
        let query = ConeQuery::Projection {
            query: Box::new(ConeQuery::Projection {
                query: Box::new(ConeQuery::Atom(start)),
                relation_axes: array![0.3, 0.0],
                relation_apertures: array![0.1, 0.0],
            }),
            relation_axes: array![0.2, 0.0],
            relation_apertures: array![0.1, 0.0],
        };
        let result = query.evaluate().unwrap();
        assert!((result.axes()[0] - 0.5).abs() < 1e-5);
        assert!((result.apertures()[0] - 0.7).abs() < 1e-5);
    }
}
