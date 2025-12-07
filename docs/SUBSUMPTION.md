# Subsumption

## Definition

**Subsumption** is a fundamental concept in formal logic. In box embeddings, when box \(A\) contains box \(B\) geometrically (i.e., \(B \subseteq A\)), we say that **\(A\) subsumes \(B\)**.

## Statement

**Theorem (Geometric Subsumption).** For boxes \(A\) and \(B\):

\[
\text{Box } A \text{ subsumes Box } B \iff B \subseteq A \iff P(B \subseteq A) = 1
\]

Geometric containment directly corresponds to logical subsumption.

## Proof

Under the uniform base measure on \([0,1]^d\), containment probability is:

\[
P(B \subseteq A) = \frac{\text{Vol}(A \cap B)}{\text{Vol}(B)}
\]

When \(B \subseteq A\), we have \(A \cap B = B\), so \(\text{Vol}(A \cap B) = \text{Vol}(B)\), giving \(P(B \subseteq A) = 1\).

For Gumbel boxes, this becomes:

\[
P(B \subseteq A) = \frac{\mathbb{E}[\text{Vol}(A \cap B)]}{\mathbb{E}[\text{Vol}(B)]}
\]

using the first-order Taylor approximation (see [`CONTAINMENT_PROBABILITY.md`](CONTAINMENT_PROBABILITY.md)).

## Interpretation

Subsumption encodes three fundamental relationships:

1. **Entailment**: If premise box \(P\) subsumes hypothesis box \(H\), then \(P\) entails \(H\)
2. **Hierarchies**: Parent concepts subsume child concepts (e.g., "animal" subsumes "dog")
3. **Logical consequence**: The containment relationship represents logical subsumption

## Example

Consider the hierarchy: "dog" ⊆ "mammal" ⊆ "animal"

**Box embeddings:**
- "dog" = box from \([0.2, 0.4]\) to \([0.4, 0.6]\) (volume = 0.04)
- "mammal" = box from \([0.1, 0.3]\) to \([0.5, 0.7]\) (volume = 0.16) ← contains "dog"
- "animal" = box from \([0.0, 0.0]\) to \([1.0, 1.0]\) (volume = 1.0) ← contains "mammal"

**Verification:**
- \(P(\text{dog} \subseteq \text{mammal}) = \text{Vol}(\text{dog}) / \text{Vol}(\text{dog}) = 1.0\) ✓
- \(P(\text{mammal} \subseteq \text{animal}) = \text{Vol}(\text{mammal}) / \text{Vol}(\text{mammal}) = 1.0\) ✓

Containment is geometrically explicit and directly encodes logical relationships.

