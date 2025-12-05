# Remaining Work: Implementation, Evaluation, and Datasets

## Summary

This document outlines what remains to be implemented, evaluated, or requires dataset integration.

## Not Yet Implemented

### 1. Center-Offset Representation
**Status**: ✅ Implemented (conversion utilities)  
**Priority**: Low (conversion utilities sufficient for most use cases)  
**Location**: `subsume-core/src/center_offset.rs`

**What**: Alternative parameterization using center point + offset vector instead of min-max coordinates.

**Why**: Better gradient flow for neural network training, automatic constraint satisfaction.

**Current Implementation**:
- ✅ Conversion utilities: `center_offset_to_min_max`, `min_max_to_center_offset`
- ✅ Uses sigmoid + softplus for stable numerical conversion
- ✅ Round-trip tests verify correctness

**Future Enhancement** (optional):
- Add `CenterOffsetBox` trait/struct for direct center-offset operations
- Backend implementations (`NdarrayCenterOffsetBox`, `CandleCenterOffsetBox`)
- Update training utilities to support both representations natively

---

### 2. Full Training Loops for Quantitative Verification
**Status**: ✅ Core infrastructure implemented, examples available  
**Priority**: Medium (examples exist, full paper reproduction pending)  
**Location**: `subsume-core/src/trainer.rs`, `subsume-ndarray/src/optimizer.rs`, `subsume-ndarray/examples/`

**What**: Complete training implementations to reproduce paper claims:
- Dasgupta 2020: "6 F1 improvement" (GumbelBox vs SmoothBox)
- Concept2Box 2023: "Joint learning improves performance"
- Boratko 2020: "SOTA performance on FB15k-237, WN18RR, YAGO3-10"

**Current Implementation**:
- ✅ Dataset loading/parsing (WN18RR, FB15k-237, YAGO3-10)
- ✅ Complete training loops with optimizers (Adam, AdamW, SGD)
- ✅ Evaluation protocol (link prediction, MRR, Hits@K, Mean Rank)
- ✅ Real training examples: `real_training_wn18rr.rs`, `real_training_fb15k237.rs`, `real_training_boxe.rs`
- ✅ Automated evaluation infrastructure with plotting
- ✅ Hyperparameter search utilities
- ✅ Result logging and comparison

**Remaining Work**:
- Full quantitative verification against paper claims (requires dataset downloads and extended training runs)
- Statistical significance testing across multiple runs
- Baseline comparisons (other embedding methods)

---

### 3. Candle Examples for New Metrics
**Status**: ✅ All examples ported  
**Priority**: Medium  
**Location**: `subsume-candle/examples/candle_recent_research_metrics.rs`, `candle_concept2box.rs`, `candle_regd_training.rs`

**What**: Examples demonstrating:
- Depth distance (`depth_distance`)
- Boundary distance (`boundary_distance`)
- Vector-to-box distance (`vector_to_box_distance`)
- Recent research metrics integration

**Why**: Ensure Candle backend has parity with ndarray examples.

**Implementation Requirements**:
- Port examples from `subsume-ndarray/examples/recent_research_metrics.rs`
- Port `concept2box_joint_learning.rs` example
- Port `regd_training.rs` example

**Estimated Effort**: Low (1 day)

---

### 4. Full BoxE Implementation
**Status**: ✅ Core BoxE scoring and loss implemented  
**Priority**: Medium  
**Location**: `subsume-core/src/boxe.rs`, `subsume-ndarray/examples/boxe_training.rs`

**What**: Complete BoxE model with:
- Translational bumps (relation-specific transformations)
- Full training loop for knowledge graph completion
- Performance benchmarks on standard datasets

**Why**: Reproduce Boratko 2020 SOTA results.

**Implementation Requirements**:
- Add "bump" operations (translation vectors per relation)
- Implement BoxE scoring function
- Add relation-specific transformations
- Training loop with negative sampling

**Estimated Effort**: High (1 week)

---

### 5. Full EL++ Axiom Handling (TransBox)
**Status**: ⏳ Containment properties verified, full axiom handling pending  
**Priority**: Low  
**Location**: `docs/PAPER_VERIFICATION.md`

**What**: Support for EL++ Description Logic axioms:
- Concept inclusion (C ⊑ D)
- Role inclusion (R ⊑ S)
- Domain/range restrictions
- Complex concept constructors

**Why**: Enable formal ontology embedding applications.

**Implementation Requirements**:
- Axiom parser (OWL/EL++ format)
- Axiom-to-box constraint mapping
- Logical closure verification
- Training with logical constraints

**Estimated Effort**: High (1-2 weeks)

---

### 6. Full Gaussian Convolution (SmoothBox)
**Status**: ⏳ Simplified version exists, full convolution pending  
**Priority**: Low  
**Location**: `docs/PAPER_VERIFICATION.md`

**What**: Complete Gaussian convolution for soft volume calculation (Li et al. 2019).

**Why**: Verify SmoothBox performance claims.

**Implementation Requirements**:
- Gaussian convolution implementation
- Integration with volume calculation
- Performance comparison with GumbelBox

**Estimated Effort**: Medium (2-3 days)

---

## Evaluation Gaps

### 1. Dataset Integration
**Status**: ✅ Loading utilities implemented, needs actual dataset files  
**Priority**: High  
**Location**: `subsume-core/src/dataset.rs`, `subsume-ndarray/examples/dataset_loading.rs`
**Datasets Needed**:
- **WN18RR**: WordNet knowledge graph (link prediction)
- **FB15k-237**: Freebase knowledge graph (link prediction)
- **YAGO3-10**: YAGO knowledge graph (link prediction)
- **DBpedia**: Two-view KG (concepts + entities) for Concept2Box

**Implementation Requirements**:
- Dataset download/loading utilities
- Data format parsers (triples: head, relation, tail)
- Train/validation/test splits
- Negative sampling for training

**Estimated Effort**: Medium (3-5 days)

---

### 2. Quantitative Performance Verification
**Status**: ⏳ Methodology documented, results pending  
**Priority**: High

**Claims to Verify**:
1. **Dasgupta 2020**: GumbelBox achieves ~6 F1 improvement over SmoothBox
2. **RegD 2025**: Depth distance addresses crowding effect (verified via tests, needs full training)
3. **Concept2Box 2023**: Joint learning improves performance
4. **Boratko 2020**: BoxE achieves SOTA on standard datasets

**Implementation Requirements**:
- Full training loops (see above)
- Evaluation metrics (MRR, Hits@K, F1)
- Baseline comparisons
- Statistical significance testing

**Estimated Effort**: High (depends on training loop implementation)

---

### 3. Regression Tests for Performance
**Status**: ✅ Implemented  
**Priority**: Low  
**Location**: `subsume-ndarray/src/regression_tests.rs`

**What**: Automated tests that detect performance regressions.

**Implementation Requirements**:
- Baseline performance metrics storage
- CI/CD integration
- Threshold-based failure detection
- Performance trend tracking

**Estimated Effort**: Low (1-2 days)

---

### 4. Expressiveness Empirical Validation
**Status**: ⏳ Theoretical expressiveness verified, empirical validation pending  
**Priority**: Medium

**What**: Compare box embeddings to hyperbolic embeddings on hierarchy tasks.

**Implementation Requirements**:
- Hyperbolic embedding baseline (Poincaré/Lorentz)
- Hierarchy dataset (tree structures)
- Distance metric comparison
- Expressiveness quantification

**Estimated Effort**: Medium (3-5 days)

---

## New Datasets Needed

### 1. Standard Knowledge Graph Datasets
**Priority**: High

- **WN18RR**: WordNet (hierarchical, good for containment)
- **FB15k-237**: Freebase (large-scale, diverse relations)
- **YAGO3-10**: YAGO (high-quality, curated)
- **DBpedia**: Two-view KG (concepts + entities)

**Format**: Triples (head, relation, tail) with train/val/test splits

**Usage**: Link prediction evaluation, training loop verification

---

### 2. Hierarchy-Specific Datasets
**Priority**: Medium

- **WordNet hierarchy**: Full hyponym-hypernym tree
- **DBpedia ontology**: Concept hierarchy
- **Medical ontologies**: SNOMED CT, UMLS (healthcare applications)

**Format**: Parent-child relationships, depth annotations

**Usage**: Depth distance evaluation, crowding effect verification

---

### 3. Two-View Knowledge Graphs
**Priority**: Medium

- **DBpedia**: Concepts (boxes) + Entities (vectors)
- **Wikidata**: Items (vectors) + Categories (boxes)

**Format**: Separate concept and entity files, cross-view relationships

**Usage**: Concept2Box joint learning evaluation

---

## Priority Summary

### High Priority (Blocking Quantitative Verification)
1. ✅ Dataset integration (WN18RR, FB15k-237)
2. ✅ Full training loops for paper claims
3. ✅ Evaluation protocol implementation

### Medium Priority (Feature Completeness)
1. ✅ Center-offset representation (conversion utilities)
2. ✅ Candle examples for new metrics (all ported)
3. ⏳ Expressiveness empirical validation (needs datasets)

### Low Priority (Nice to Have)
1. ⏳ Full BoxE implementation
2. ⏳ Full EL++ axiom handling
3. ⏳ Regression tests for performance
4. ⏳ Full Gaussian convolution

---

## Next Steps

1. **Start with dataset integration** (highest impact, enables everything else)
2. **Implement minimal training loop** (SGD, basic negative sampling)
3. **Add evaluation metrics** (MRR, Hits@K already implemented, need integration)
4. **Run baseline experiments** (establish performance baselines)
5. **Verify paper claims** (compare to documented results)

---

## Estimated Timeline

- **Dataset integration**: 3-5 days
- **Training loop implementation**: 1 week
- **Evaluation integration**: 2-3 days
- **Baseline experiments**: 1 week
- **Total**: ~3-4 weeks for high-priority items

