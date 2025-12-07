#import "template.typ": theorem, definition, proof, example

#align(center)[
  #text(size: 24pt, weight: "bold")[Future Directions and Open Questions]
]

#v(1em)

== Overview

This document explores open questions and future research directions in box embeddings. While the mathematical foundations are well-established, several frontiers remain for exploration and development.

== Open Theoretical Questions

=== Scaling to Large Knowledge Graphs

*Question*: How do box embeddings scale to very large knowledge graphs (millions of entities, billions of triples)?

*Current State*: Most work focuses on medium-scale KGs (thousands to hundreds of thousands of entities). Large-scale evaluation is limited.

*Challenges*:
- Volume computation complexity: O(d) per box, but with millions of boxes, this becomes expensive
- Memory requirements: Each box requires 2d parameters (min and max coordinates)
- Training time: Gradient computation scales with the number of boxes

*Future Directions*:
- Efficient volume approximation methods (beyond the Bessel function approximation)
- Sparse box representations for large-scale KGs
- Hierarchical box structures (boxes of boxes) for multi-scale representation
- Distributed training strategies for massive KGs

*Connection to Foundations*: The volume calculation (see the Gumbel-Box Volume document) and numerical approximation methods provide a starting point, but more efficient methods may be needed for very large scales.

=== Optimal Temperature Scheduling

*Question*: What is the optimal temperature schedule for Gumbel boxes during training?

*Current State*: Most work uses fixed or simple annealing schedules (linear, exponential decay). The optimal schedule is not well-understood.

*Challenges*:
- Temperature $beta$ controls the "softness" of boundaries (see the Log-Sum-Exp and Gumbel Intersection document)
- Too high: boxes become too soft, losing geometric structure
- Too low: boxes become too hard, losing gradient signal (see the Local Identifiability document)
- Optimal schedule likely depends on dataset, task, and model architecture

*Future Directions*:
- Adaptive temperature scheduling based on training dynamics
- Task-specific temperature optimization
- Theoretical analysis of temperature's effect on learning dynamics
- Connection to curriculum learning and annealing strategies

*Connection to Foundations*: The temperature parameter appears throughout the mathematical foundations (volume calculations, log-sum-exp, local identifiability), but its optimal scheduling remains an open question.

=== Expressiveness Comparison

*Question*: How do box embeddings compare to hyperbolic embeddings in practice? When is each preferable?

*Current State*: RegD (2025) shows that Euclidean box embeddings can achieve hyperbolic-like expressiveness, but empirical comparisons are limited.

*Challenges*:
- Hyperbolic embeddings excel at tree-like hierarchies
- Box embeddings excel at DAGs and partial orders
- Direct comparison requires careful experimental design
- Expressiveness depends on data structure (tree vs DAG vs general graph)

*Future Directions*:
- Large-scale empirical comparison on diverse datasets
- Theoretical analysis of expressiveness for different graph structures
- Hybrid approaches combining boxes and hyperbolic embeddings
- Task-specific recommendations (when to use boxes vs hyperbolic)

*Connection to Foundations*: The max-stability property (see the Gumbel Max-Stability document) and volume calculations provide theoretical advantages, but empirical validation is needed.

== Practical Research Directions

=== Uncertainty Quantification

*Question*: How can we better quantify and utilize uncertainty in box embeddings?

*Current State*: Gumbel boxes provide probabilistic boundaries, but uncertainty quantification is not fully exploited.

*Future Directions*:
- Uncertainty-aware training objectives
- Calibrated uncertainty estimates for downstream tasks
- Connection to Bayesian neural networks and variational inference
- Applications in active learning and few-shot learning

*Connection to Foundations*: The probabilistic formulation (see the Containment Probability document) and Gumbel distributions provide a natural foundation for uncertainty quantification, but more work is needed to fully exploit this.

=== Multi-Modal Box Embeddings

*Question*: Can box embeddings be extended to multi-modal settings (text + images, text + knowledge graphs)?

*Current State*: Most work focuses on single-modal settings (text or knowledge graphs). Multi-modal extensions are limited.

*Future Directions*:
- Joint embedding of text and images using boxes
- Cross-modal containment relationships (e.g., image contains text description)
- Integration with vision-language models (CLIP, BLIP)
- Applications in multi-modal retrieval and reasoning

*Connection to Foundations*: The geometric interpretation of containment (see the Subsumption document) could naturally extend to multi-modal settings, but the mathematical framework needs extension.

=== Temporal Knowledge Graphs

*Question*: How can box embeddings handle temporal dynamics in knowledge graphs?

*Current State*: BoxTE (2022) introduces temporal box embeddings, but temporal dynamics are not fully explored.

*Future Directions*:
- Time-aware box embeddings (boxes that evolve over time)
- Temporal containment relationships (A contains B at time t)
- Integration with temporal reasoning and forecasting
- Applications in event prediction and temporal link prediction

*Connection to Foundations*: The max-stability property (see the Gumbel Max-Stability document) could enable efficient temporal updates, but temporal dynamics need more theoretical development.

=== Integration with Large Language Models

*Question*: How can box embeddings be integrated with large language models (LLMs) for enhanced reasoning?

*Current State*: LLMs excel at language understanding but struggle with structured knowledge. Box embeddings could provide structured knowledge representation.

*Future Directions*:
- LLM-generated box embeddings for knowledge extraction
- Box embeddings as external knowledge for LLM reasoning
- Hybrid architectures combining LLMs and box embeddings
- Applications in knowledge-augmented language models

*Connection to Foundations*: The subsumption relationship (see the Subsumption document) could enable LLMs to reason about hierarchical knowledge, but integration methods need development.

== Implementation Challenges

=== Numerical Stability at Scale

*Question*: How can we ensure numerical stability for box embeddings in high dimensions or with extreme parameter values?

*Current State*: The log-sum-exp trick (see the Log-Sum-Exp and Gumbel Intersection document) and Bessel function approximation (see the Gumbel-Box Volume document) provide stability, but edge cases remain.

*Future Directions*:
- Improved numerical approximations for extreme cases
- Adaptive precision strategies
- Robust volume computation methods
- Connection to numerical analysis and floating-point arithmetic

=== Efficient Intersection Computation

*Question*: Can we compute box intersections more efficiently, especially for many boxes?

*Current State*: Intersection computation is O(d) per pair, but with many boxes, this becomes expensive.

*Future Directions*:
- Spatial data structures for efficient intersection queries
- Approximate intersection methods for large-scale settings
- Parallel intersection computation strategies
- Connection to computational geometry algorithms

*Connection to Foundations*: The log-sum-exp function (see the Log-Sum-Exp and Gumbel Intersection document) provides the mathematical foundation, but computational efficiency needs improvement.

== Applications and Use Cases

=== Healthcare and Bioinformatics

*Potential*: Medical ontologies (SNOMED CT, UMLS) are natural applications for box embeddings.

*Future Directions*:
- Embedding of medical ontologies for clinical decision support
- Integration with electronic health records
- Applications in drug discovery and protein function prediction
- Validation of logical consistency in medical knowledge

*Connection to Foundations*: The subsumption relationship (see the Subsumption document) naturally models medical hierarchies, and formal ontology embedding (TransBox 2024) demonstrates feasibility.

=== Semantic Web and Knowledge Graphs

*Potential*: Large-scale knowledge graphs (DBpedia, Wikidata) could benefit from box embeddings.

*Future Directions*:
- Scalable embedding of web-scale knowledge graphs
- Integration with semantic web standards (RDF, OWL)
- Applications in knowledge graph completion and reasoning
- Hybrid approaches combining boxes with other geometric structures

*Connection to Foundations*: The mathematical foundations provide the theoretical basis, but scalability and efficiency need improvement.

=== Natural Language Processing

*Potential*: Hyponym-hypernym relationships in natural language are natural applications.

*Future Directions*:
- Embedding of WordNet and other lexical resources
- Integration with language models for semantic understanding
- Applications in question answering and semantic parsing
- Cross-lingual box embeddings for multilingual knowledge

*Connection to Foundations*: The subsumption relationship (see the Subsumption document) naturally models semantic hierarchies, but NLP-specific extensions may be needed.

== Theoretical Extensions

=== Beyond Axis-Aligned Boxes

*Question*: Can we extend box embeddings to rotated or non-axis-aligned boxes?

*Current State*: All work focuses on axis-aligned boxes (hyperrectangles). Rotated boxes would increase expressiveness but complicate computation.

*Future Directions*:
- Rotated box embeddings with learnable rotation angles
- Theoretical analysis of expressiveness gains
- Efficient computation methods for rotated boxes
- Applications where rotation is semantically meaningful

*Connection to Foundations*: The current mathematical framework assumes axis-alignment. Extensions would require new volume calculations and intersection methods.

=== Higher-Order Relationships

*Question*: Can box embeddings handle relationships beyond binary (e.g., ternary, n-ary)?

*Current State*: Most work focuses on binary relationships (head, relation, tail). Higher-order relationships are limited.

*Future Directions*:
- Box embeddings for n-ary relationships
- Compositional reasoning with multiple boxes
- Applications in event extraction and relation extraction
- Theoretical analysis of expressiveness for higher-order structures

*Connection to Foundations*: The intersection operation (see the Log-Sum-Exp and Gumbel Intersection document) could be extended, but the mathematical framework needs development.

== Summary

The mathematical foundations of box embeddings are well-established, but many frontiers remain:

*Theoretical*: Scaling, expressiveness, optimal scheduling
*Practical*: Uncertainty quantification, multi-modal extensions, temporal dynamics
*Implementation*: Numerical stability, efficient computation
*Applications*: Healthcare, semantic web, natural language processing
*Extensions*: Rotated boxes, higher-order relationships

The future of box embeddings is bright, with active research addressing these challenges and exploring new directions. The mathematical foundations we've established provide a solid base for future developments.

== Connection to Mathematical Foundations

All future directions build on the mathematical foundations:

- *Subsumption*: Enables logical reasoning and hierarchical modeling
- *Gumbel Box Volume*: Provides computational tractability
- *Containment Probability*: Enables probabilistic reasoning
- *Gumbel Max-Stability*: Ensures algebraic closure and efficiency
- *Log-Sum-Exp*: Provides numerical stability
- *Local Identifiability*: Enables effective learning

These foundations provide the theoretical basis for addressing open questions and exploring new directions.

== Next Steps

For implementation details and practical guidance, see the code documentation. For recent applications, see the Applications document. For mathematical foundations, see the previous documents in this series.

