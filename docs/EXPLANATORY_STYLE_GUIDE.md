# Explanatory Style Guide: Inspired by Aho & Ullman

This document captures the explanatory style patterns we've adopted from **Aho & Ullman's Foundations of Computer Science** and applied throughout the codebase documentation.

## Core Principles

### 1. Paradigm Problems

Every abstract concept should be introduced with a concrete, memorable example that becomes a reference point.

**Pattern**: Start with a specific problem, then generalize.

**Example from our docs**:
- **Containment probability**: "Consider the knowledge graph triple (dog, is_a, mammal)..."
- **Depth distance**: "Consider a taxonomy where 'Animal' has many children: 'Dog', 'Cat', 'Bird'..."
- **BoxE training**: "Consider the knowledge graph triple (Paris, located_in, France)..."

**Why it works**: Readers remember concrete examples better than abstract definitions. The example becomes a mental anchor for understanding the general case.

### 2. Step-by-Step Reasoning

Break complex derivations into smaller, understandable steps.

**Pattern**: Show the reasoning process, not just the result.

**Example**:
```
**Step-by-step reasoning**:
1. Compute the intersection: What region do "dog" and "mammal" boxes share?
2. Compare volumes: If the intersection volume equals the "dog" volume, then "dog" is
   completely contained in "mammal"
3. Normalize: Divide by "dog" volume to get a probability between 0 and 1
```

**Why it works**: Readers can follow along and understand *why* the formula works, not just *what* it computes.

### 3. Visual Descriptions

Even without diagrams, describe visual structures clearly.

**Pattern**: Use spatial language and analogies.

**Example**: "Think of the relation as a 'magnet' that pulls the head entity box toward the tail entity. If the magnet is strong enough (the bump is right), the head box will contain the tail box after translation."

**Why it works**: Visual thinking helps intuition. Even text descriptions of visual structures aid understanding.

### 4. Breaking Complex into Simple

Decompose problems into simpler subproblems.

**Pattern**: Identify the subproblems explicitly.

**Example**: Training process broken into 4 steps: (1) Positive examples, (2) Negative examples, (3) Loss function, (4) Optimization.

**Why it works**: Complex systems are easier to understand when broken into manageable pieces.

### 5. Consistent Terminology

Define terms once and use them consistently.

**Pattern**: Use the same words for the same concepts throughout.

**Example**: Always use "containment probability" (not "containment score" or "subsumption probability" interchangeably).

**Why it works**: Reduces cognitive load—readers don't need to remember multiple synonyms.

### 6. Historical Context and References

Provide background on where concepts come from.

**Pattern**: "This approach was introduced in [Paper] (Year), Section X..."

**Example**: "Depth-based distance is introduced in **Yang & Chen (2025)**, 'RegD: Achieving Hyperbolic-Like Expressiveness with Arbitrary Euclidean Regions' (arXiv:2501.17518, Section 3.2)."

**Why it works**: Gives credibility, provides exploration paths, and helps readers understand the evolution of ideas.

### 7. Connecting Theory to Practice

Show how abstract concepts apply to real problems.

**Pattern**: Explain both the mathematical formulation and the practical application.

**Example**: "This directly models logical subsumption: if the probability is 1.0, then `self` completely subsumes `other` (entailment relationship)."

**Why it works**: Readers understand both *what* the math means and *why* it matters.

## Structure Pattern

Our documentation follows this structure (inspired by Aho & Ullman):

1. **Paradigm Problem**: Concrete example
2. **Step-by-Step Reasoning**: How to think about the problem
3. **Visual Intuition**: Spatial/geometric description
4. **Research Background**: Where it comes from
5. **Mathematical Formulation**: Precise definition
6. **Implementation Notes**: How it's computed in code

## Examples in Our Codebase

### Good Examples

- `box_trait.rs::containment_prob()`: Starts with paradigm problem (dog, is_a, mammal), then step-by-step reasoning, then mathematical formulation
- `distance.rs::depth_distance()`: Paradigm problem (crowding in taxonomies), visual intuition, then research background
- `trainer.rs`: Training process broken into 4 clear steps with examples

### Areas for Improvement

- More visual descriptions in mathematical sections
- More step-by-step derivations (not just final formulas)
- More paradigm problems for less intuitive concepts

## Key Takeaways

1. **Concrete before abstract**: Always start with a specific example
2. **Show the reasoning**: Don't just state results, show how to get there
3. **Use spatial language**: Even text descriptions of visual structures help
4. **Break it down**: Complex concepts are simpler when decomposed
5. **Be consistent**: Use the same terminology throughout
6. **Provide context**: Reference papers and explain where ideas come from
7. **Connect theory to practice**: Show both mathematical and practical aspects

## References

- Aho, A. V., & Ullman, J. D. (1992). *Foundations of Computer Science*. W. H. Freeman.
  - Available at: http://infolab.stanford.edu/~ullman/focs.html
  - Chapter 1: "Computer Science: The Mechanization of Abstraction"
  - Chapter 4: "Combinatorics and Probability"

