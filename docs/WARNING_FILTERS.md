# Warning Filters and Acceptable Patterns

This document explains which clippy warnings we suppress and why, given our project structure.

## Suppressed Warnings

### Module-Level Allows

#### `subsume-ndarray/src/lib.rs`
- `clippy::useless_vec` - `vec![]` is often clearer than `Vec::new()` in tests and examples
- `clippy::needless_range_loop` - Indexing is sometimes necessary and clearer than iterators
- `clippy::module_inception` - Test modules often match parent name (e.g., `tests::tests`)

#### `subsume-core/src/lib.rs`
- `clippy::module_inception` - Test modules often match parent name

#### `subsume-candle/src/lib.rs`
- `clippy::module_inception` - Test modules often match parent name

#### `subsume-core/src/trainer.rs`
- `clippy::module_inception` - Test module name matches parent

#### `subsume-ndarray/src/trainer_integration_tests.rs`
- `clippy::useless_vec` - `vec![]` is clearer in tests
- `clippy::needless_range_loop` - Indexing is intentional in tests

### Function-Level Allows

#### Test Functions
- `#[allow(unused_variables)]` - Used in test stubs or to document test structure
  - Example: `test_evaluate_link_prediction_basic` uses `_empty_boxes` to document structure

## Fixed Warnings

### Clamp Patterns
- **Before**: `.min(1.0).max(0.0)` or `.max(1e-7).min(1.0 - 1e-7)`
- **After**: `.clamp(0.0, 1.0)` or `.clamp(1e-7, 1.0 - 1e-7)`
- **Files**: `subsume-core/src/boxe.rs`, `subsume-core/src/center_offset.rs`

### Taken Reference
- **Before**: `point.dims() != &[self.dim()]`
- **After**: `point.dims() != [self.dim()]`
- **Files**: `subsume-candle/src/candle_gumbel.rs`, `subsume-candle/src/distance.rs`

## Remaining Acceptable Warnings

### Missing Documentation
- **Location**: Examples, internal test modules, plotting functions
- **Reason**: Examples are self-documenting, test modules are internal, plotting is optional feature
- **Action**: Can be addressed incrementally, not blocking

### Redundant Imports
- **Location**: Examples
- **Reason**: Examples import more than needed for clarity/demonstration
- **Action**: Acceptable for educational purposes

### Unused Imports in Examples
- **Location**: Example files
- **Reason**: Examples may import things for demonstration that aren't used in minimal examples
- **Action**: Acceptable for educational purposes

## Guidelines

1. **Suppress warnings that are stylistic preferences**, not correctness issues
2. **Fix warnings that indicate actual problems** (clamp patterns, taken references)
3. **Document why warnings are suppressed** in code comments or this file
4. **Review suppressed warnings periodically** to ensure they're still appropriate

## When to Add New Suppressions

Add `#[allow(...)]` when:
- The warning is a false positive for our use case
- The pattern is intentional and clearer than the suggested alternative
- The warning is about style preferences we don't want to enforce
- The code is in examples/tests where clarity > strictness

Do NOT suppress:
- Actual bugs or correctness issues
- Performance problems
- Security issues
- Warnings that indicate missing functionality

