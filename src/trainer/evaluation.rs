use crate::dataset::Triple;
use crate::metrics::{hits_at_k, mean_rank, mean_reciprocal_rank};
use std::collections::{HashMap, HashSet};

use super::{EvaluationResults, PerRelationResults, RelationTransform};

/// An index of "known true" triples for filtered link prediction evaluation.
///
/// In standard KGE evaluation (e.g. FB15k-237, WN18RR), **filtered ranking** removes any
/// candidate that is already a true triple in train/valid/test, except for the test triple
/// being evaluated. This avoids penalizing the model for ranking other true answers above
/// the held-out one.
///
/// This index is intentionally shaped for the most common evaluation query we currently
/// support in `subsume`:
/// - tail prediction: `(h, r, ?)`
///
/// Notes:
/// - Building this index **allocates**, but using it during evaluation does not.
/// - Memory can be large for big KGs; prefer using `FilteredTripleIndexIds` with interned IDs.
#[derive(Debug, Default, Clone)]
pub struct FilteredTripleIndex {
    tails_by_head_rel: HashMap<String, HashMap<String, HashSet<String>>>,
    heads_by_tail_rel: HashMap<String, HashMap<String, HashSet<String>>>,
}

impl FilteredTripleIndex {
    /// Build a filtered-ranking index from an iterator of triples.
    pub fn from_triples<'a, I>(triples: I) -> Self
    where
        I: IntoIterator<Item = &'a Triple>,
    {
        let mut index = Self::default();
        index.extend(triples);
        index
    }

    /// Build a filtered-ranking index from all splits of a [`Dataset`](crate::dataset::Dataset).
    ///
    /// Indexes train + valid + test triples so that filtered evaluation can
    /// exclude all known-true triples.
    pub fn from_dataset(dataset: &crate::dataset::Dataset) -> Self {
        Self::from_triples(
            dataset
                .train
                .iter()
                .chain(dataset.valid.iter())
                .chain(dataset.test.iter()),
        )
    }

    /// Extend the index with more known-true triples.
    pub fn extend<'a, I>(&mut self, triples: I)
    where
        I: IntoIterator<Item = &'a Triple>,
    {
        for t in triples {
            self.tails_by_head_rel
                .entry(t.head.clone())
                .or_default()
                .entry(t.relation.clone())
                .or_default()
                .insert(t.tail.clone());
            self.heads_by_tail_rel
                .entry(t.tail.clone())
                .or_default()
                .entry(t.relation.clone())
                .or_default()
                .insert(t.head.clone());
        }
    }

    /// True iff `(head, relation, tail)` is a known true triple.
    #[inline]
    pub fn is_known_tail(&self, head: &str, relation: &str, tail: &str) -> bool {
        self.tails_by_head_rel
            .get(head)
            .and_then(|by_rel| by_rel.get(relation))
            .is_some_and(|tails| tails.contains(tail))
    }

    /// Return all known-true tails for the query `(head, relation, ?)`.
    #[inline]
    pub fn known_tails(&self, head: &str, relation: &str) -> Option<&HashSet<String>> {
        self.tails_by_head_rel
            .get(head)
            .and_then(|by_rel| by_rel.get(relation))
    }

    /// True iff `(head, relation, tail)` is a known true triple (head lookup).
    #[inline]
    pub fn is_known_head(&self, tail: &str, relation: &str, head: &str) -> bool {
        self.heads_by_tail_rel
            .get(tail)
            .and_then(|by_rel| by_rel.get(relation))
            .is_some_and(|heads| heads.contains(head))
    }

    /// Return all known-true heads for the query `(?, relation, tail)`.
    #[inline]
    pub fn known_heads(&self, tail: &str, relation: &str) -> Option<&HashSet<String>> {
        self.heads_by_tail_rel
            .get(tail)
            .and_then(|by_rel| by_rel.get(relation))
    }
}

/// Like [`FilteredTripleIndex`], but for interned integer IDs.
///
/// This is the preferred form for performance-sensitive evaluation, because it avoids
/// hashing/cloning `String` IDs in the hot loop.
#[derive(Debug, Default, Clone)]
pub struct FilteredTripleIndexIds {
    // Keyed by (head_id, relation_id).
    //
    // Using a flat key avoids a nested HashMap allocation per distinct (head, relation).
    tails_by_head_rel: HashMap<(usize, usize), HashSet<usize>>,
    // Keyed by (tail_id, relation_id).
    heads_by_tail_rel: HashMap<(usize, usize), HashSet<usize>>,
}

impl FilteredTripleIndexIds {
    /// Build a filtered-ranking index from an iterator of ID triples.
    pub fn from_triples<'a, I>(triples: I) -> Self
    where
        I: IntoIterator<Item = &'a crate::dataset::TripleIds>,
    {
        let mut index = Self::default();
        index.extend(triples);
        index
    }

    /// Build a filtered-ranking index from all splits of an [`InternedDataset`](crate::dataset::InternedDataset).
    ///
    /// Indexes train + valid + test triples so that filtered evaluation can
    /// exclude all known-true triples.
    pub fn from_dataset(dataset: &crate::dataset::InternedDataset) -> Self {
        Self::from_triples(
            dataset
                .train
                .iter()
                .chain(dataset.valid.iter())
                .chain(dataset.test.iter()),
        )
    }

    /// Extend the index with more known-true triples.
    pub fn extend<'a, I>(&mut self, triples: I)
    where
        I: IntoIterator<Item = &'a crate::dataset::TripleIds>,
    {
        for t in triples {
            self.tails_by_head_rel
                .entry((t.head, t.relation))
                .or_default()
                .insert(t.tail);
            self.heads_by_tail_rel
                .entry((t.tail, t.relation))
                .or_default()
                .insert(t.head);
        }
    }

    /// True iff `(head, relation, tail)` is a known true triple.
    #[inline]
    pub fn is_known_tail(&self, head: usize, relation: usize, tail: usize) -> bool {
        self.tails_by_head_rel
            .get(&(head, relation))
            .is_some_and(|tails| tails.contains(&tail))
    }

    /// Return all known-true tails for the query `(head, relation, ?)`.
    #[inline]
    pub fn known_tails(&self, head: usize, relation: usize) -> Option<&HashSet<usize>> {
        self.tails_by_head_rel.get(&(head, relation))
    }

    /// True iff `(head, relation, tail)` is a known true triple (head lookup).
    #[inline]
    pub fn is_known_head(&self, tail: usize, relation: usize, head: usize) -> bool {
        self.heads_by_tail_rel
            .get(&(tail, relation))
            .is_some_and(|heads| heads.contains(&head))
    }

    /// Return all known-true heads for the query `(?, relation, tail)`.
    #[inline]
    pub fn known_heads(&self, tail: usize, relation: usize) -> Option<&HashSet<usize>> {
        self.heads_by_tail_rel.get(&(tail, relation))
    }
}

/// Compute the rank of `target` among all entities, scoring each candidate via `score_fn`.
///
/// `score_fn(candidate_box) -> f32` returns a score (higher = more likely).
/// Deterministic tie-break: among equal scores, lexicographically smaller entity id ranks first.
fn rank_among_entities<B, F>(
    entity_boxes: &HashMap<String, B>,
    target: &str,
    score_fn: F,
    filter_known: Option<&HashSet<String>>,
) -> Result<usize, crate::BoxError>
where
    B: crate::Box,
    F: Fn(&B) -> Result<f32, crate::BoxError>,
{
    let target_box = match entity_boxes.get(target) {
        Some(b) => b,
        None => return Ok(usize::MAX),
    };
    let target_score = score_fn(target_box)?;
    if target_score.is_nan() {
        return Err(crate::BoxError::Internal(
            "NaN containment score encountered (target)".to_string(),
        ));
    }

    let mut better = 0usize;
    let mut tie_before = 0usize;
    for (entity, box_) in entity_boxes {
        if entity == target {
            continue;
        }
        let score = score_fn(box_)?;
        if score.is_nan() {
            return Err(crate::BoxError::Internal(
                "NaN containment score encountered".to_string(),
            ));
        }
        if score > target_score {
            better += 1;
        } else if score == target_score && entity.as_str() < target {
            tie_before += 1;
        }
    }

    // Filtered ranking: subtract contributions from known-true entities.
    if let Some(known) = filter_known {
        let mut filtered_better = 0usize;
        let mut filtered_tie_before = 0usize;
        for known_entity in known {
            if known_entity == target {
                continue;
            }
            let Some(box_) = entity_boxes.get(known_entity) else {
                continue;
            };
            let score = score_fn(box_)?;
            if score.is_nan() {
                return Err(crate::BoxError::Internal(
                    "NaN containment score encountered".to_string(),
                ));
            }
            if score > target_score {
                filtered_better += 1;
            } else if score == target_score && known_entity.as_str() < target {
                filtered_tie_before += 1;
            }
        }
        better = better.saturating_sub(filtered_better);
        tie_before = tie_before.saturating_sub(filtered_tie_before);
    }

    Ok(better + tie_before + 1)
}

pub(crate) fn evaluate_link_prediction_inner<B>(
    test_triples: &[Triple],
    entity_boxes: &HashMap<String, B>,
    relation_transforms: Option<&HashMap<String, RelationTransform>>,
    filter: Option<&FilteredTripleIndex>,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box,
{
    // The generic (string-keyed) path only supports Identity transforms.
    // Non-Identity transforms require constructing translated boxes from concrete
    // min/max coordinates, which needs the interned NdarrayBox path.
    if let Some(transforms) = relation_transforms {
        for (rel, transform) in transforms {
            if !transform.is_identity() {
                return Err(crate::BoxError::Internal(format!(
                    "Non-Identity RelationTransform for relation '{}' requires the interned \
                     evaluation path with NdarrayBox",
                    rel
                )));
            }
        }
    }

    let mut tail_ranks = Vec::with_capacity(test_triples.len());
    let mut head_ranks = Vec::with_capacity(test_triples.len());
    // (relation, tail_rank, head_rank) per triple for per-relation aggregation.
    let mut per_triple: Vec<(&str, usize, usize)> = Vec::with_capacity(test_triples.len());

    for triple in test_triples {
        let head_box = entity_boxes
            .get(&triple.head)
            .ok_or_else(|| crate::BoxError::Internal(format!("Missing entity: {}", triple.head)))?;

        // -- Tail prediction: (h, r, ?) --
        let filter_tails = filter.and_then(|f| f.known_tails(&triple.head, &triple.relation));
        let t_rank = rank_among_entities(
            entity_boxes,
            &triple.tail,
            |candidate| head_box.containment_prob_fast(candidate),
            filter_tails,
        )?;

        // -- Head prediction: (?, r, t) --
        let tail_box = entity_boxes
            .get(&triple.tail)
            .ok_or_else(|| crate::BoxError::Internal(format!("Missing entity: {}", triple.tail)))?;
        let filter_heads = filter.and_then(|f| f.known_heads(&triple.tail, &triple.relation));
        let h_rank = rank_among_entities(
            entity_boxes,
            &triple.head,
            |candidate| candidate.containment_prob_fast(tail_box),
            filter_heads,
        )?;

        tail_ranks.push(t_rank);
        head_ranks.push(h_rank);
        per_triple.push((triple.relation.as_str(), t_rank, h_rank));
    }

    // Combined ranks: both head and tail ranks contribute equally (Bordes 2013 protocol).
    let all_ranks: Vec<usize> = tail_ranks
        .iter()
        .chain(head_ranks.iter())
        .copied()
        .collect();

    let mrr = mean_reciprocal_rank(&all_ranks) as f32;
    let tail_mrr = mean_reciprocal_rank(&tail_ranks) as f32;
    let head_mrr = mean_reciprocal_rank(&head_ranks) as f32;
    let hits_at_1 = hits_at_k(&all_ranks, 1) as f32;
    let hits_at_3 = hits_at_k(&all_ranks, 3) as f32;
    let hits_at_10 = hits_at_k(&all_ranks, 10) as f32;
    let mean_rank_val = mean_rank(&all_ranks) as f32;

    // Per-relation aggregation.
    let per_relation = aggregate_per_relation(&per_triple);

    Ok(EvaluationResults {
        mrr,
        head_mrr,
        tail_mrr,
        hits_at_1,
        hits_at_3,
        hits_at_10,
        mean_rank: mean_rank_val,
        per_relation,
    })
}

/// Aggregate per-relation metrics from per-triple (relation, tail_rank, head_rank) tuples.
fn aggregate_per_relation(per_triple: &[(&str, usize, usize)]) -> Vec<PerRelationResults> {
    let mut by_rel: HashMap<&str, Vec<usize>> = HashMap::new();
    for &(rel, t_rank, h_rank) in per_triple {
        let ranks = by_rel.entry(rel).or_default();
        ranks.push(t_rank);
        ranks.push(h_rank);
    }
    let mut results: Vec<PerRelationResults> = by_rel
        .into_iter()
        .map(|(rel, ranks)| {
            let count = ranks.len() / 2; // number of triples (each contributes 2 ranks)
            let mrr = mean_reciprocal_rank(&ranks) as f32;
            let h10 = hits_at_k(&ranks, 10) as f32;
            PerRelationResults {
                relation: rel.to_string(),
                mrr,
                hits_at_10: h10,
                count,
            }
        })
        .collect();
    results.sort_by(|a, b| a.relation.cmp(&b.relation));
    results
}

/// Scoring direction for interned rank computation.
pub(crate) enum ScoreDirection {
    /// Tail prediction: score = query.containment_prob(candidate).
    /// Can use batch `containment_prob_many`.
    Forward,
    /// Head prediction: score = candidate.containment_prob(query).
    /// Falls back to per-entity `containment_prob_fast`.
    Reverse,
}

/// Compute the rank of `target_id` among all entity boxes in the interned setting.
///
/// - `Forward`: score = query_box.containment_prob(candidate) (tail prediction).
/// - `Reverse`: score = candidate.containment_prob(query_box) (head prediction).
pub(crate) fn rank_among_entities_interned<B>(
    entity_boxes: &[B],
    entities: &crate::dataset::Vocab,
    target_id: usize,
    query_box: &B,
    direction: &ScoreDirection,
    filter_known: Option<&HashSet<usize>>,
    scores_buf: &mut Vec<f32>,
) -> Result<usize, crate::BoxError>
where
    B: crate::Box,
{
    const CHUNK: usize = 4096;

    let target_box = match entity_boxes.get(target_id) {
        Some(b) => b,
        None => return Ok(usize::MAX),
    };
    let target_name = entities.get(target_id).ok_or_else(|| {
        crate::BoxError::Internal(format!("Missing entity label (target): {}", target_id))
    })?;
    let target_score = match direction {
        ScoreDirection::Forward => query_box.containment_prob_fast(target_box)?,
        ScoreDirection::Reverse => target_box.containment_prob_fast(query_box)?,
    };
    if target_score.is_nan() {
        return Err(crate::BoxError::Internal(
            "NaN containment score encountered (target)".to_string(),
        ));
    }

    if scores_buf.len() < CHUNK {
        scores_buf.resize(CHUNK, 0.0);
    }

    let mut better = 0usize;
    let mut tie_before = 0usize;

    match direction {
        ScoreDirection::Forward => {
            // Batch scoring: query_box.containment_prob_many(candidates).
            for start in (0..entity_boxes.len()).step_by(CHUNK) {
                let end = (start + CHUNK).min(entity_boxes.len());
                let slice = &entity_boxes[start..end];
                let len = end - start;

                query_box.containment_prob_many(slice, &mut scores_buf[..len])?;

                for (i, &score) in scores_buf[..len].iter().enumerate() {
                    let entity_id = start + i;
                    if entity_id == target_id {
                        continue;
                    }
                    if score.is_nan() {
                        return Err(crate::BoxError::Internal(
                            "NaN containment score encountered".to_string(),
                        ));
                    }
                    if score > target_score {
                        better += 1;
                    } else if score == target_score {
                        let name = entities.get(entity_id).ok_or_else(|| {
                            crate::BoxError::Internal(format!(
                                "Missing entity label (candidate): {}",
                                entity_id
                            ))
                        })?;
                        if name < target_name {
                            tie_before += 1;
                        }
                    }
                }
            }
        }
        ScoreDirection::Reverse => {
            // Per-entity scoring: candidate.containment_prob_fast(query_box).
            for (entity_id, candidate) in entity_boxes.iter().enumerate() {
                if entity_id == target_id {
                    continue;
                }
                let score = candidate.containment_prob_fast(query_box)?;
                if score.is_nan() {
                    return Err(crate::BoxError::Internal(
                        "NaN containment score encountered".to_string(),
                    ));
                }
                if score > target_score {
                    better += 1;
                } else if score == target_score {
                    let name = entities.get(entity_id).ok_or_else(|| {
                        crate::BoxError::Internal(format!(
                            "Missing entity label (candidate): {}",
                            entity_id
                        ))
                    })?;
                    if name < target_name {
                        tie_before += 1;
                    }
                }
            }
        }
    }

    if let Some(known) = filter_known {
        let mut filtered_better = 0usize;
        let mut filtered_tie_before = 0usize;
        for &known_id in known {
            if known_id == target_id {
                continue;
            }
            let Some(box_) = entity_boxes.get(known_id) else {
                continue;
            };
            let score = match direction {
                ScoreDirection::Forward => query_box.containment_prob_fast(box_)?,
                ScoreDirection::Reverse => box_.containment_prob_fast(query_box)?,
            };
            if score.is_nan() {
                return Err(crate::BoxError::Internal(
                    "NaN containment score encountered".to_string(),
                ));
            }
            if score > target_score {
                filtered_better += 1;
            } else if score == target_score {
                let name = entities.get(known_id).ok_or_else(|| {
                    crate::BoxError::Internal(format!(
                        "Missing entity label (filtered): {}",
                        known_id
                    ))
                })?;
                if name < target_name {
                    filtered_tie_before += 1;
                }
            }
        }
        better = better.saturating_sub(filtered_better);
        tie_before = tie_before.saturating_sub(filtered_tie_before);
    }

    Ok(better + tie_before + 1)
}

/// Rank target among all entities using a translated query box (tail prediction).
///
/// Constructs a translated `NdarrayBox` from `query_box` + `transform`, then
/// scores `translated.containment_prob_fast(candidate)` for each entity.
#[cfg(feature = "ndarray-backend")]
pub(crate) fn rank_with_translated_query_forward(
    entity_boxes: &[crate::ndarray_backend::NdarrayBox],
    entities: &crate::dataset::Vocab,
    target_id: usize,
    query_box: &crate::ndarray_backend::NdarrayBox,
    transform: &RelationTransform,
    filter_known: Option<&HashSet<usize>>,
) -> Result<usize, crate::BoxError> {
    use crate::Box as BoxTrait;

    let (new_min, new_max) =
        transform.apply_to_bounds(query_box.min().as_slice(), query_box.max().as_slice());
    let translated = crate::ndarray_backend::NdarrayBox::new(
        ndarray::Array1::from_vec(new_min),
        ndarray::Array1::from_vec(new_max),
        1.0,
    )?;

    let target_score =
        translated.containment_prob_fast(entity_boxes.get(target_id).ok_or_else(|| {
            crate::BoxError::Internal(format!("Missing entity id (target): {target_id}"))
        })?)?;
    if target_score.is_nan() {
        return Err(crate::BoxError::Internal(
            "NaN containment score encountered (target)".to_string(),
        ));
    }
    let target_name = entities
        .get(target_id)
        .ok_or_else(|| crate::BoxError::Internal(format!("Missing entity label: {target_id}")))?;

    let mut better = 0usize;
    let mut tie_before = 0usize;
    for (entity_id, candidate) in entity_boxes.iter().enumerate() {
        if entity_id == target_id {
            continue;
        }
        let score = translated.containment_prob_fast(candidate)?;
        if score.is_nan() {
            return Err(crate::BoxError::Internal(
                "NaN containment score encountered".to_string(),
            ));
        }
        if score > target_score {
            better += 1;
        } else if score == target_score {
            let name = entities.get(entity_id).ok_or_else(|| {
                crate::BoxError::Internal(format!("Missing entity label: {entity_id}"))
            })?;
            if name < target_name {
                tie_before += 1;
            }
        }
    }

    if let Some(known) = filter_known {
        let mut filtered_better = 0usize;
        let mut filtered_tie_before = 0usize;
        for &known_id in known {
            if known_id == target_id {
                continue;
            }
            let Some(box_) = entity_boxes.get(known_id) else {
                continue;
            };
            let score = translated.containment_prob_fast(box_)?;
            if score.is_nan() {
                continue;
            }
            if score > target_score {
                filtered_better += 1;
            } else if score == target_score {
                let name = entities.get(known_id).ok_or_else(|| {
                    crate::BoxError::Internal(format!("Missing entity label: {known_id}"))
                })?;
                if name < target_name {
                    filtered_tie_before += 1;
                }
            }
        }
        better = better.saturating_sub(filtered_better);
        tie_before = tie_before.saturating_sub(filtered_tie_before);
    }

    Ok(better + tie_before + 1)
}

/// Rank target among all entities using a translated query box (head prediction).
///
/// For head prediction `(?, r, t)`, we score `candidate.containment_prob_fast(translated_tail)`.
/// The transform is applied inversely: `Translation(d)` becomes `Translation(-d)` so that
/// the tail is shifted to the "un-transformed" space where candidates live.
#[cfg(feature = "ndarray-backend")]
pub(crate) fn rank_with_translated_query_reverse(
    entity_boxes: &[crate::ndarray_backend::NdarrayBox],
    entities: &crate::dataset::Vocab,
    target_id: usize,
    query_box: &crate::ndarray_backend::NdarrayBox,
    transform: &RelationTransform,
    filter_known: Option<&HashSet<usize>>,
) -> Result<usize, crate::BoxError> {
    use crate::Box as BoxTrait;

    // Inverse transform for head prediction: negate the translation.
    let inverse_transform = match transform {
        RelationTransform::Identity => RelationTransform::Identity,
        RelationTransform::Translation(d) => {
            RelationTransform::Translation(d.iter().map(|x| -x).collect())
        }
    };

    let (new_min, new_max) =
        inverse_transform.apply_to_bounds(query_box.min().as_slice(), query_box.max().as_slice());
    let translated = crate::ndarray_backend::NdarrayBox::new(
        ndarray::Array1::from_vec(new_min),
        ndarray::Array1::from_vec(new_max),
        1.0,
    )?;

    let target_box = entity_boxes.get(target_id).ok_or_else(|| {
        crate::BoxError::Internal(format!("Missing entity id (target): {target_id}"))
    })?;
    let target_score = target_box.containment_prob_fast(&translated)?;
    if target_score.is_nan() {
        return Err(crate::BoxError::Internal(
            "NaN containment score encountered (target)".to_string(),
        ));
    }
    let target_name = entities
        .get(target_id)
        .ok_or_else(|| crate::BoxError::Internal(format!("Missing entity label: {target_id}")))?;

    let mut better = 0usize;
    let mut tie_before = 0usize;
    for (entity_id, candidate) in entity_boxes.iter().enumerate() {
        if entity_id == target_id {
            continue;
        }
        let score = candidate.containment_prob_fast(&translated)?;
        if score.is_nan() {
            return Err(crate::BoxError::Internal(
                "NaN containment score encountered".to_string(),
            ));
        }
        if score > target_score {
            better += 1;
        } else if score == target_score {
            let name = entities.get(entity_id).ok_or_else(|| {
                crate::BoxError::Internal(format!("Missing entity label: {entity_id}"))
            })?;
            if name < target_name {
                tie_before += 1;
            }
        }
    }

    if let Some(known) = filter_known {
        let mut filtered_better = 0usize;
        let mut filtered_tie_before = 0usize;
        for &known_id in known {
            if known_id == target_id {
                continue;
            }
            let Some(box_) = entity_boxes.get(known_id) else {
                continue;
            };
            let score = box_.containment_prob_fast(&translated)?;
            if score.is_nan() {
                continue;
            }
            if score > target_score {
                filtered_better += 1;
            } else if score == target_score {
                let name = entities.get(known_id).ok_or_else(|| {
                    crate::BoxError::Internal(format!("Missing entity label: {known_id}"))
                })?;
                if name < target_name {
                    filtered_tie_before += 1;
                }
            }
        }
        better = better.saturating_sub(filtered_better);
        tie_before = tie_before.saturating_sub(filtered_tie_before);
    }

    Ok(better + tie_before + 1)
}

pub(crate) fn evaluate_link_prediction_interned_inner<B>(
    test_triples: &[crate::dataset::TripleIds],
    entity_boxes: &[B],
    entities: &crate::dataset::Vocab,
    filter: Option<&FilteredTripleIndexIds>,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box,
{
    let mut tail_ranks = Vec::with_capacity(test_triples.len());
    let mut head_ranks = Vec::with_capacity(test_triples.len());
    let mut per_triple: Vec<(usize, usize, usize)> = Vec::with_capacity(test_triples.len());
    let mut scores_buf = vec![0.0f32; 4096];

    for triple in test_triples {
        let head_box = entity_boxes.get(triple.head).ok_or_else(|| {
            crate::BoxError::Internal(format!("Missing entity id (head): {}", triple.head))
        })?;
        let tail_box = entity_boxes.get(triple.tail).ok_or_else(|| {
            crate::BoxError::Internal(format!("Missing entity id (tail): {}", triple.tail))
        })?;

        let filter_tails = filter.and_then(|f| f.known_tails(triple.head, triple.relation));
        let t_rank = rank_among_entities_interned(
            entity_boxes,
            entities,
            triple.tail,
            head_box,
            &ScoreDirection::Forward,
            filter_tails,
            &mut scores_buf,
        )?;

        let filter_heads = filter.and_then(|f| f.known_heads(triple.tail, triple.relation));
        let h_rank = rank_among_entities_interned(
            entity_boxes,
            entities,
            triple.head,
            tail_box,
            &ScoreDirection::Reverse,
            filter_heads,
            &mut scores_buf,
        )?;

        tail_ranks.push(t_rank);
        head_ranks.push(h_rank);
        per_triple.push((triple.relation, t_rank, h_rank));
    }

    collect_evaluation_results(&tail_ranks, &head_ranks, &per_triple)
}

/// Evaluate interned link prediction with relation-specific transforms (NdarrayBox only).
///
/// This is the concrete implementation backing
/// [`evaluate_link_prediction_interned_with_transforms`](super::evaluate_link_prediction_interned_with_transforms).
/// It handles both identity and non-identity transforms by dispatching to the translated
/// ranking helpers.
#[cfg(feature = "ndarray-backend")]
pub(crate) fn evaluate_interned_with_transforms_inner(
    test_triples: &[crate::dataset::TripleIds],
    entity_boxes: &[crate::ndarray_backend::NdarrayBox],
    entities: &crate::dataset::Vocab,
    relation_transforms: &[RelationTransform],
    filter: Option<&FilteredTripleIndexIds>,
) -> Result<EvaluationResults, crate::BoxError> {
    let mut tail_ranks = Vec::with_capacity(test_triples.len());
    let mut head_ranks = Vec::with_capacity(test_triples.len());
    let mut per_triple: Vec<(usize, usize, usize)> = Vec::with_capacity(test_triples.len());
    let mut scores_buf = vec![0.0f32; 4096];

    for triple in test_triples {
        let head_box = entity_boxes.get(triple.head).ok_or_else(|| {
            crate::BoxError::Internal(format!("Missing entity id (head): {}", triple.head))
        })?;
        let tail_box = entity_boxes.get(triple.tail).ok_or_else(|| {
            crate::BoxError::Internal(format!("Missing entity id (tail): {}", triple.tail))
        })?;

        let transform = relation_transforms
            .get(triple.relation)
            .unwrap_or(&RelationTransform::Identity);

        let filter_tails = filter.and_then(|f| f.known_tails(triple.head, triple.relation));
        let t_rank = if transform.is_identity() {
            rank_among_entities_interned(
                entity_boxes,
                entities,
                triple.tail,
                head_box,
                &ScoreDirection::Forward,
                filter_tails,
                &mut scores_buf,
            )?
        } else {
            rank_with_translated_query_forward(
                entity_boxes,
                entities,
                triple.tail,
                head_box,
                transform,
                filter_tails,
            )?
        };

        let filter_heads = filter.and_then(|f| f.known_heads(triple.tail, triple.relation));
        let h_rank = if transform.is_identity() {
            rank_among_entities_interned(
                entity_boxes,
                entities,
                triple.head,
                tail_box,
                &ScoreDirection::Reverse,
                filter_heads,
                &mut scores_buf,
            )?
        } else {
            rank_with_translated_query_reverse(
                entity_boxes,
                entities,
                triple.head,
                tail_box,
                transform,
                filter_heads,
            )?
        };

        tail_ranks.push(t_rank);
        head_ranks.push(h_rank);
        per_triple.push((triple.relation, t_rank, h_rank));
    }

    collect_evaluation_results(&tail_ranks, &head_ranks, &per_triple)
}

/// Collect tail/head ranks into [`EvaluationResults`].
pub(crate) fn collect_evaluation_results(
    tail_ranks: &[usize],
    head_ranks: &[usize],
    per_triple: &[(usize, usize, usize)],
) -> Result<EvaluationResults, crate::BoxError> {
    let all_ranks: Vec<usize> = tail_ranks
        .iter()
        .chain(head_ranks.iter())
        .copied()
        .collect();

    let mrr = mean_reciprocal_rank(&all_ranks) as f32;
    let tail_mrr = mean_reciprocal_rank(tail_ranks) as f32;
    let head_mrr = mean_reciprocal_rank(head_ranks) as f32;
    let hits_at_1 = hits_at_k(&all_ranks, 1) as f32;
    let hits_at_3 = hits_at_k(&all_ranks, 3) as f32;
    let hits_at_10 = hits_at_k(&all_ranks, 10) as f32;
    let mean_rank_val = mean_rank(&all_ranks) as f32;

    let per_relation = aggregate_per_relation_ids(per_triple);

    Ok(EvaluationResults {
        mrr,
        head_mrr,
        tail_mrr,
        hits_at_1,
        hits_at_3,
        hits_at_10,
        mean_rank: mean_rank_val,
        per_relation,
    })
}

/// Aggregate per-relation metrics from per-triple (relation_id, tail_rank, head_rank) tuples.
fn aggregate_per_relation_ids(per_triple: &[(usize, usize, usize)]) -> Vec<PerRelationResults> {
    let mut by_rel: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(rel, t_rank, h_rank) in per_triple {
        let ranks = by_rel.entry(rel).or_default();
        ranks.push(t_rank);
        ranks.push(h_rank);
    }
    let mut results: Vec<PerRelationResults> = by_rel
        .into_iter()
        .map(|(rel, ranks)| {
            let count = ranks.len() / 2;
            let mrr = mean_reciprocal_rank(&ranks) as f32;
            let h10 = hits_at_k(&ranks, 10) as f32;
            PerRelationResults {
                relation: rel.to_string(),
                mrr,
                hits_at_10: h10,
                count,
            }
        })
        .collect();
    results.sort_by(|a, b| a.relation.cmp(&b.relation));
    results
}

/// Evaluate link prediction performance.
///
/// # Research Background
///
/// Link prediction evaluation follows the standard protocol established by **Bordes et al. (2013)**
/// for TransE and used consistently across knowledge graph embedding literature. The metrics
/// (MRR, Hits@K, Mean Rank) are standard benchmarks for knowledge graph completion.
///
/// **Reference**: Bordes et al. (2013), "Translating Embeddings for Modeling Multi-relational Data"
///
/// # Intuitive Explanation
///
/// Link prediction evaluates both directions for each test triple (Bordes 2013 protocol):
/// - **Tail prediction**: given (head, relation, ?), rank all entities as candidate tails
/// - **Head prediction**: given (?, relation, tail), rank all entities as candidate heads
///
/// **The process**:
/// 1. For each test triple (e.g., (Paris, located_in, France))
/// 2. Tail prediction: score all entities as candidates for (Paris, located_in, ?)
/// 3. Head prediction: score all entities as candidates for (?, located_in, France)
/// 4. Average both directions into aggregate metrics
///
/// **Metrics computed**:
/// - **MRR (Mean Reciprocal Rank)**: Average of 1/rank for correct answers
///   - If correct answer is rank 1 → 1/1 = 1.0 (perfect)
///   - If correct answer is rank 5 → 1/5 = 0.2
///   - Higher is better (range: 0 to 1)
///
/// - **Hits@K**: Fraction of queries where correct answer is in top K
///   - Hits@10 = 0.8 means 80% of queries have correct answer in top 10
///   - Higher is better (range: 0 to 1)
///
/// - **Mean Rank**: Average position of correct answers
///   - Lower is better (best = 1.0, worst = number of entities)
///
/// **Why this matters**: These metrics tell you if the model learned meaningful geometric
/// relationships. High MRR means boxes are arranged so containment probabilities match
/// knowledge graph structure.
///
/// # Arguments
///
/// * `test_triples` - Test set triples (held-out true facts)
/// * `entity_boxes` - Map from entity ID to box embedding
/// # Returns
///
/// Evaluation results with bidirectional MRR (head/tail breakdown), Hits@K, Mean Rank,
/// and per-relation metrics.
///
/// # Note
///
/// This function works with any `Box` implementation.
pub fn evaluate_link_prediction<B>(
    test_triples: &[Triple],
    entity_boxes: &HashMap<String, B>,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box,
{
    evaluate_link_prediction_inner(test_triples, entity_boxes, None, None)
}

/// Evaluate link prediction in the **filtered** setting.
///
/// Filtered ranking excludes known-true candidates: for tail prediction, excludes
/// `t'` where `(h, r, t')` is known; for head prediction, excludes `h'`
/// where `(h', r, t)` is known. The test triple's own entity is never filtered.
pub fn evaluate_link_prediction_filtered<B>(
    test_triples: &[Triple],
    entity_boxes: &HashMap<String, B>,
    filter: &FilteredTripleIndex,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box,
{
    evaluate_link_prediction_inner(test_triples, entity_boxes, None, Some(filter))
}

/// Evaluate link prediction using interned IDs (`usize`) for entities/relations.
///
/// This avoids string hashing/cloning in the candidate loop, which is often the dominant
/// overhead once the scoring kernel itself is optimized.
pub fn evaluate_link_prediction_interned<B>(
    test_triples: &[crate::dataset::TripleIds],
    entity_boxes: &[B],
    entities: &crate::dataset::Vocab,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box,
{
    evaluate_link_prediction_interned_inner(test_triples, entity_boxes, entities, None)
}

/// Evaluate link prediction in the **filtered** setting, using interned IDs.
pub fn evaluate_link_prediction_interned_filtered<B>(
    test_triples: &[crate::dataset::TripleIds],
    entity_boxes: &[B],
    entities: &crate::dataset::Vocab,
    filter: &FilteredTripleIndexIds,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box,
{
    evaluate_link_prediction_interned_inner(test_triples, entity_boxes, entities, Some(filter))
}

/// Evaluate link prediction with relation-specific transforms (interned IDs).
///
/// The `relation_transforms` slice is indexed by relation ID. Use
/// [`RelationTransform::Identity`] for relations without a transform.
/// [`RelationTransform::Translation`] is supported because this function
/// requires the `ndarray-backend` feature and concrete `NdarrayBox` entities.
#[cfg(feature = "ndarray-backend")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
pub fn evaluate_link_prediction_interned_with_transforms(
    test_triples: &[crate::dataset::TripleIds],
    entity_boxes: &[crate::ndarray_backend::NdarrayBox],
    entities: &crate::dataset::Vocab,
    relation_transforms: &[RelationTransform],
    filter: Option<&FilteredTripleIndexIds>,
) -> Result<EvaluationResults, crate::BoxError> {
    evaluate_interned_with_transforms_inner(
        test_triples,
        entity_boxes,
        entities,
        relation_transforms,
        filter,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filtered_triple_index_membership() {
        let triples = [
            Triple {
                head: "h".to_string(),
                relation: "r".to_string(),
                tail: "t1".to_string(),
            },
            Triple {
                head: "h".to_string(),
                relation: "r".to_string(),
                tail: "t2".to_string(),
            },
            Triple {
                head: "h".to_string(),
                relation: "r2".to_string(),
                tail: "t3".to_string(),
            },
        ];

        let idx = FilteredTripleIndex::from_triples(triples.iter());

        assert!(idx.is_known_tail("h", "r", "t1"));
        assert!(idx.is_known_tail("h", "r", "t2"));
        assert!(!idx.is_known_tail("h", "r", "t3"));
        assert!(idx.is_known_tail("h", "r2", "t3"));
        assert!(!idx.is_known_tail("missing", "r", "t1"));
    }

    #[test]
    fn link_prediction_rank_linear_matches_deterministic_sort() {
        // The ranking logic in `evaluate_link_prediction` is intentionally O(|E|)
        // and uses a deterministic tie-break on entity id. This test ensures that
        // the linear-time rank matches an explicit sort using the same ordering.
        //
        // Ordering:
        //   higher score first; among equal scores, lexicographically smaller id first.
        for n in [1usize, 2, 10, 100] {
            let ids: Vec<String> = (0..n).map(|i| format!("e{i:03}")).collect();

            // Create a score pattern with ties, and a deterministic non-sorted iteration order
            // to ensure tie-breaking does not depend on input order.
            //
            // Scores are bucketed into 7 bins to force collisions:
            //   score(i) = (i % 7) / 7
            let mut scores: Vec<(String, f32)> = Vec::with_capacity(n);
            for j in 0..n {
                // A simple permutation (invertible when n is odd, but we don't need that).
                let i = (j.wrapping_mul(17) + 3) % n;
                let s = (i % 7) as f32 / 7.0;
                scores.push((ids[i].clone(), s));
            }

            // Pick a deterministic target (middle id if present).
            let tail = ids[n / 2].clone();
            let tail_score = ((n / 2) % 7) as f32 / 7.0;

            // Linear-time rank (same as evaluate_link_prediction).
            let mut better = 0usize;
            let mut tie_before = 0usize;
            for (id, s) in &scores {
                if id == &tail {
                    continue;
                }
                if *s > tail_score {
                    better += 1;
                } else if *s == tail_score && id.as_str() < tail.as_str() {
                    tie_before += 1;
                }
            }
            let rank_linear = better + tie_before + 1;

            // Deterministic sort-based rank.
            scores.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .expect("no NaNs in test scores")
                    .then_with(|| a.0.cmp(&b.0))
            });
            let rank_sort = scores
                .iter()
                .position(|(id, _)| id == &tail)
                .map(|pos| pos + 1)
                .unwrap_or(usize::MAX);

            assert_eq!(rank_linear, rank_sort);
        }
    }

    // -----------------------------------------------------------------------
    // evaluate_link_prediction with NdarrayBox
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn evaluate_link_prediction_with_ndarray_boxes() {
        use crate::ndarray_backend::NdarrayBox;
        use ndarray::array;

        // Three entities: A contains B, C is disjoint.
        // Query: (A, r, ?) -- correct tail is B.
        let a = NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let c = NdarrayBox::new(array![50.0, 50.0], array![51.0, 51.0], 1.0).unwrap();

        let mut entity_boxes = HashMap::new();
        entity_boxes.insert("A".to_string(), a);
        entity_boxes.insert("B".to_string(), b);
        entity_boxes.insert("C".to_string(), c);

        let test_triples = vec![Triple {
            head: "A".to_string(),
            relation: "r".to_string(),
            tail: "B".to_string(),
        }];

        let results = evaluate_link_prediction(&test_triples, &entity_boxes).unwrap();

        // B is contained in A and should rank highly; C is disjoint.
        // With 3 entities, best rank is 1, so MRR should be > 0.
        assert!(
            results.mrr > 0.0,
            "MRR should be positive, got {}",
            results.mrr
        );
        assert!(results.mean_rank >= 1.0, "mean_rank should be >= 1");
    }

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn evaluate_link_prediction_empty_triples() {
        use crate::ndarray_backend::NdarrayBox;
        use ndarray::array;

        let a = NdarrayBox::new(array![0.0], array![1.0], 1.0).unwrap();
        let mut entity_boxes = HashMap::new();
        entity_boxes.insert("A".to_string(), a);

        let results = evaluate_link_prediction::<NdarrayBox>(&[], &entity_boxes).unwrap();
        // No triples to evaluate: metrics should be NaN or 0 depending on implementation.
        // mean_reciprocal_rank([]) and hits_at_k([]) return NaN per standard behavior.
        // Just check it does not panic.
        let _ = results;
    }

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn evaluate_link_prediction_filtered_excludes_known_tails() {
        use crate::ndarray_backend::NdarrayBox;
        use ndarray::array;

        // Four entities: A, B, C, D. A contains B and C; D is disjoint.
        // Known true: (A, r, C). Test triple: (A, r, B).
        // In filtered setting, C should be excluded from ranking.
        let a = NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let c = NdarrayBox::new(array![2.0, 2.0], array![4.0, 4.0], 1.0).unwrap();
        let d = NdarrayBox::new(array![50.0, 50.0], array![51.0, 51.0], 1.0).unwrap();

        let mut entity_boxes = HashMap::new();
        entity_boxes.insert("A".to_string(), a);
        entity_boxes.insert("B".to_string(), b);
        entity_boxes.insert("C".to_string(), c);
        entity_boxes.insert("D".to_string(), d);

        let test_triples = vec![Triple {
            head: "A".to_string(),
            relation: "r".to_string(),
            tail: "B".to_string(),
        }];

        let filter_triples = [
            Triple {
                head: "A".into(),
                relation: "r".into(),
                tail: "C".into(),
            },
            Triple {
                head: "A".into(),
                relation: "r".into(),
                tail: "B".into(),
            },
        ];
        let filter = FilteredTripleIndex::from_triples(filter_triples.iter());

        let unfiltered = evaluate_link_prediction(&test_triples, &entity_boxes).unwrap();
        let filtered =
            evaluate_link_prediction_filtered(&test_triples, &entity_boxes, &filter).unwrap();

        // Filtered rank should be <= unfiltered rank (fewer competitors).
        assert!(
            filtered.mean_rank <= unfiltered.mean_rank,
            "filtered rank ({}) should be <= unfiltered rank ({})",
            filtered.mean_rank,
            unfiltered.mean_rank
        );
    }

    // -----------------------------------------------------------------------
    // evaluate_link_prediction_interned with NdarrayBox
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn evaluate_link_prediction_interned_with_ndarray_boxes() {
        use crate::dataset::{TripleIds, Vocab};
        use crate::ndarray_backend::NdarrayBox;
        use ndarray::array;

        let mut vocab = Vocab::default();
        let id_a = vocab.intern("A".to_string());
        let id_b = vocab.intern("B".to_string());
        let _id_c = vocab.intern("C".to_string());
        let id_r = 0usize; // relation id

        // A contains B; C disjoint.
        let boxes = vec![
            NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap(), // A
            NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap(),   // B
            NdarrayBox::new(array![50.0, 50.0], array![51.0, 51.0], 1.0).unwrap(), // C
        ];

        let test_triples = vec![TripleIds {
            head: id_a,
            relation: id_r,
            tail: id_b,
        }];

        let results = evaluate_link_prediction_interned(&test_triples, &boxes, &vocab).unwrap();
        assert!(
            results.mrr > 0.0,
            "MRR should be positive, got {}",
            results.mrr
        );
        assert!(results.mean_rank >= 1.0);
    }

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn evaluate_link_prediction_interned_filtered_with_ndarray_boxes() {
        use crate::dataset::{TripleIds, Vocab};
        use crate::ndarray_backend::NdarrayBox;
        use ndarray::array;

        let mut vocab = Vocab::default();
        let id_a = vocab.intern("A".to_string());
        let id_b = vocab.intern("B".to_string());
        let id_c = vocab.intern("C".to_string());
        let id_r = 0usize;

        let boxes = vec![
            NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap(),
            NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap(),
            NdarrayBox::new(array![2.0, 2.0], array![4.0, 4.0], 1.0).unwrap(),
        ];

        let test_triples = vec![TripleIds {
            head: id_a,
            relation: id_r,
            tail: id_b,
        }];
        let known_triples = [
            TripleIds {
                head: id_a,
                relation: id_r,
                tail: id_c,
            },
            TripleIds {
                head: id_a,
                relation: id_r,
                tail: id_b,
            },
        ];
        let filter = FilteredTripleIndexIds::from_triples(known_triples.iter());

        let unfiltered = evaluate_link_prediction_interned(&test_triples, &boxes, &vocab).unwrap();
        let filtered =
            evaluate_link_prediction_interned_filtered(&test_triples, &boxes, &vocab, &filter)
                .unwrap();

        assert!(
            filtered.mean_rank <= unfiltered.mean_rank,
            "filtered rank ({}) should be <= unfiltered rank ({})",
            filtered.mean_rank,
            unfiltered.mean_rank
        );
    }

    // -----------------------------------------------------------------------
    // FilteredTripleIndexIds tests
    // -----------------------------------------------------------------------

    #[test]
    fn filtered_triple_index_ids_membership() {
        use crate::dataset::TripleIds;

        let triples = [
            TripleIds {
                head: 0,
                relation: 0,
                tail: 1,
            },
            TripleIds {
                head: 0,
                relation: 0,
                tail: 2,
            },
            TripleIds {
                head: 0,
                relation: 1,
                tail: 3,
            },
        ];

        let idx = FilteredTripleIndexIds::from_triples(triples.iter());

        assert!(idx.is_known_tail(0, 0, 1));
        assert!(idx.is_known_tail(0, 0, 2));
        assert!(!idx.is_known_tail(0, 0, 3)); // different relation
        assert!(idx.is_known_tail(0, 1, 3));
        assert!(!idx.is_known_tail(1, 0, 1)); // different head
    }

    #[test]
    fn filtered_triple_index_ids_known_tails() {
        use crate::dataset::TripleIds;

        let triples = [
            TripleIds {
                head: 0,
                relation: 0,
                tail: 10,
            },
            TripleIds {
                head: 0,
                relation: 0,
                tail: 20,
            },
        ];
        let idx = FilteredTripleIndexIds::from_triples(triples.iter());

        let tails = idx.known_tails(0, 0).unwrap();
        assert!(tails.contains(&10));
        assert!(tails.contains(&20));
        assert!(!tails.contains(&30));
        assert!(idx.known_tails(1, 0).is_none());
    }

    // -----------------------------------------------------------------------
    // Evaluation determinism
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn evaluate_link_prediction_deterministic() {
        use crate::ndarray_backend::NdarrayBox;
        use ndarray::array;

        let a = NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let c = NdarrayBox::new(array![50.0, 50.0], array![51.0, 51.0], 1.0).unwrap();

        let mut entity_boxes = HashMap::new();
        entity_boxes.insert("A".to_string(), a);
        entity_boxes.insert("B".to_string(), b);
        entity_boxes.insert("C".to_string(), c);

        let test_triples = vec![Triple {
            head: "A".to_string(),
            relation: "r".to_string(),
            tail: "B".to_string(),
        }];

        let r1 = evaluate_link_prediction(&test_triples, &entity_boxes).unwrap();
        let r2 = evaluate_link_prediction(&test_triples, &entity_boxes).unwrap();

        assert_eq!(r1.mrr, r2.mrr, "MRR differs across runs");
        assert_eq!(r1.hits_at_1, r2.hits_at_1, "Hits@1 differs across runs");
        assert_eq!(r1.hits_at_3, r2.hits_at_3, "Hits@3 differs across runs");
        assert_eq!(r1.hits_at_10, r2.hits_at_10, "Hits@10 differs across runs");
        assert_eq!(r1.mean_rank, r2.mean_rank, "mean_rank differs across runs");
    }
}
