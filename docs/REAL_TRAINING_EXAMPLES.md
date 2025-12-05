# Real Training Examples

This document describes the real training examples that work with actual knowledge graph datasets.

## Available Examples

### 1. `real_training_wn18rr.rs` - WN18RR Link Prediction

Complete training pipeline for WordNet knowledge graph link prediction.

**Features:**
- Loads WN18RR dataset from `data/wn18rr/`
- Adam optimizer with paper defaults
- Early stopping based on validation MRR
- Full evaluation metrics (MRR, Hits@K, Mean Rank)

**Usage:**
```bash
# Download WN18RR dataset first
# Place in data/wn18rr/ with train.txt, valid.txt, test.txt

cargo run --example real_training_wn18rr

# Or specify custom path
cargo run --example real_training_wn18rr -- data/custom_path
```

**Dataset Download:**
- Source: https://github.com/kkteru/grail
- Format: Tab or whitespace-separated triples
- Files needed: `train.txt`, `valid.txt`, `test.txt`

---

### 2. `real_training_fb15k237.rs` - FB15k-237 Knowledge Graph Completion

Training example optimized for Freebase subset (FB15k-237).

**Features:**
- AdamW optimizer (better generalization)
- Higher embedding dimension (100)
- Larger batch sizes for efficiency
- More negative samples per positive

**Usage:**
```bash
# Download FB15k-237 dataset
# Place in data/fb15k-237/

cargo run --example real_training_fb15k237
```

**Dataset Download:**
- Source: https://github.com/TimDettmers/ConvE
- Format: Tab or whitespace-separated triples

---

### 3. `real_training_boxe.rs` - BoxE Model Training

Complete BoxE implementation with translational bumps for relation-specific transformations.

**Features:**
- Entity boxes + relation bumps
- BoxE scoring function
- Margin-based ranking loss
- Works with any standard KG dataset

**Usage:**
```bash
cargo run --example real_training_boxe -- data/wn18rr
```

**Key Differences from Standard Training:**
- Each relation has a "bump" (translation vector)
- Scoring: `P(tail ⊆ (head + bump))`
- Better models relation-specific transformations

---

## Dataset Format

All examples expect datasets in the standard format:

```
data/
  wn18rr/
    train.txt      # Training triples
    valid.txt      # Validation triples
    test.txt       # Test triples
    entities.dict  # Optional: entity ID to name mapping
    relations.dict # Optional: relation ID to name mapping
```

**Triple Format:**
- Tab-separated: `head_entity\trelation\ttail_entity`
- Whitespace-separated: `head_entity relation tail_entity`

**Example:**
```
00001740	_hypernym	00002137
00002137	_hypernym	00004475
```

---

## Training Configuration

### Standard Configuration (WN18RR)
- **Optimizer**: Adam
- **Learning rate**: 1e-3
- **Embedding dim**: 50
- **Batch size**: 512
- **Negative samples**: 1
- **Epochs**: 100
- **Early stopping**: 10 epochs patience

### Large Dataset Configuration (FB15k-237)
- **Optimizer**: AdamW
- **Learning rate**: 5e-4
- **Weight decay**: 1e-4
- **Embedding dim**: 100
- **Batch size**: 1024
- **Negative samples**: 5
- **Epochs**: 50

---

## Expected Results

### WN18RR (WordNet)
- **MRR**: ~0.40-0.50 (varies with hyperparameters)
- **Hits@10**: ~0.50-0.60
- **Training time**: ~10-30 minutes (depending on hardware)

### FB15k-237 (Freebase)
- **MRR**: ~0.25-0.35
- **Hits@10**: ~0.40-0.50
- **Training time**: ~1-3 hours (full dataset)

### BoxE Results
- Similar to standard training but with relation-specific modeling
- Better performance on datasets with diverse relation types

---

## Downstream Use Cases

### 1. Link Prediction
Predict missing links in knowledge graphs:
- **Input**: (head, relation, ?)
- **Output**: Ranked list of tail entities
- **Evaluation**: MRR, Hits@K

### 2. Knowledge Graph Completion
Fill in missing triples:
- **Input**: Partial knowledge graph
- **Output**: Complete knowledge graph
- **Use case**: Database completion, ontology extension

### 3. Entity Classification
Classify entities into categories:
- **Input**: Entity box embedding
- **Output**: Category (using containment hierarchy)
- **Use case**: Taxonomy construction, concept organization

### 4. Relation Prediction
Predict relations between entities:
- **Input**: (head, ?, tail)
- **Output**: Ranked list of relations
- **Use case**: Relation discovery, schema learning

### 5. Query Answering
Answer queries over knowledge graphs:
- **Input**: Complex query (e.g., "What are all mammals?")
- **Output**: Set of entities
- **Use case**: Question answering, semantic search

---

## Performance Tips

1. **Use AdamW for large datasets** - Better generalization
2. **Increase batch size** - Faster training, more stable gradients
3. **Use more negative samples** - Better discrimination
4. **Early stopping** - Prevents overfitting
5. **Learning rate scheduling** - Start high, decay over time

---

## Troubleshooting

### Dataset Not Found
- Check dataset path
- Ensure `train.txt`, `valid.txt`, `test.txt` exist
- Verify file format (tab or whitespace-separated)

### Out of Memory
- Reduce batch size
- Use smaller embedding dimension
- Process dataset in chunks

### Poor Performance
- Increase embedding dimension
- Train for more epochs
- Tune learning rate
- Use more negative samples

---

## Next Steps

1. **Hyperparameter tuning**: Use `run_benchmarks` example to find optimal settings
2. **Evaluation**: Compare to paper baselines
3. **Visualization**: Use plotting features to analyze training
4. **Extension**: Add custom loss functions, evaluation metrics

