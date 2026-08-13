# Mathematical Foundations

This document provides a concise guide to the mathematical foundations of box embeddings. Each concept is covered in a focused one-page document following textbook-style exposition (Definition → Statement → Proof → Example).

The documents are organized as a narrative story, from historical context through modern applications to future directions.

## Narrative Structure

### Introduction: The Embedding Evolution
**PDF:** [`typst-output/pdf/00-introduction.pdf`](typst-output/pdf/00-introduction.pdf)  
Historical context and motivation. From points to regions: the evolution of embeddings and why box embeddings emerged. Sets the stage for the mathematical foundations.

### Core Mathematical Concepts

**PDF:** [`typst-output/pdf/subsumption.pdf`](typst-output/pdf/subsumption.pdf)
Geometric containment as logical subsumption. How box containment encodes entailment, hierarchies, and logical consequence.

**PDF:** [`typst-output/pdf/gumbel-box-volume.pdf`](typst-output/pdf/gumbel-box-volume.pdf)
Expected volume for Gumbel boxes. Derivation from Gumbel distributions to Bessel function $K_0$, with numerical approximation.

**PDF:** [`typst-output/pdf/containment-probability.pdf`](typst-output/pdf/containment-probability.pdf)
First-order Taylor approximation for containment probability. Error analysis and validity conditions.

**PDF:** [`typst-output/pdf/gumbel-max-stability.pdf`](typst-output/pdf/gumbel-max-stability.pdf)
Max-stability and min-stability of Gumbel distributions. Why intersection operations preserve the Gumbel family (algebraic closure).

**PDF:** [`typst-output/pdf/log-sum-exp-intersection.pdf`](typst-output/pdf/log-sum-exp-intersection.pdf)
Log-sum-exp function and its role in Gumbel intersection. Numerical stability and the Gumbel-max property.

**PDF:** [`typst-output/pdf/local-identifiability.pdf`](typst-output/pdf/local-identifiability.pdf)
The local identifiability problem and how Gumbel boxes solve it. Why probabilistic boundaries enable gradient-based learning.

### Modern Applications and Future Directions

**PDF:** [`typst-output/pdf/07-applications.pdf`](typst-output/pdf/07-applications.pdf)  
Modern applications (2023-2025): RegD, TransBox, Concept2Box, and related developments. Demonstrates how the mathematical foundations enable competitive applications.

**PDF:** [`typst-output/pdf/08-future.pdf`](typst-output/pdf/08-future.pdf)  
Future directions and open questions. Scaling, expressiveness, uncertainty quantification, multi-modal extensions, and integration with large language models.

## Formats

- **PDF (Typst)**: Professional typesetting, optimal for printing and detailed study. See [`typst/README.md`](typst/README.md) for build instructions.
- **Typst source**: The corresponding source files are in [`typst/`](typst/).

The PDFs are pre-rendered from the Typst sources.

## Quick Reference

For implementation details connecting theory to code, see [`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md).

## Reading Order

**For understanding (narrative flow):** Read in order:
1. Introduction (historical context and motivation)
2. Subsumption (foundation concept)
3. Gumbel Box Volume (core computation)
4. Containment Probability (application)
5. Gumbel Max-Stability (mathematical property)
6. Log-Sum-Exp (numerical methods)
7. Local Identifiability (learning theory)
8. Applications (modern competitive)
9. Future Directions (open questions)

**For reference:** Jump to a specific document or use the implementation guide.
