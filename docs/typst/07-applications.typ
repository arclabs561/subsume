#import "template.typ": theorem, definition, proof, example

#align(center)[
  #text(24pt, weight: "bold")[Modern Applications and State-of-the-Art]
]

#v(1em)

== Overview

This document surveys modern applications of box embeddings (2023-2025), demonstrating their effectiveness and exploring current research frontiers. The mathematical foundations established in the previous documents enable these practical applications.

== 2025: RegD - Hyperbolic-Like Expressiveness

**Yang & Chen (2025)** - arXiv:2501.17518

*Key Contribution*: A flexible Euclidean framework that supports arbitrary geometric regions (boxes, balls) as embeddings while achieving hyperbolic-like expressiveness.

*Innovations*:
- *Depth-based dissimilarity*: Incorporates depth distance and boundary distance metrics to emulate hyperbolic geometry properties
- *Eliminates precision issues*: Uses elementary arithmetic operations instead of specialized hyperbolic constructions
- *Addresses crowding effect*: As the number of children increases in hierarchies, traditional Euclidean embeddings cluster together; RegD's depth distance solves this by incorporating region "size"
- *Boundary distance for containment*: When one region is fully contained within another, boundary distance captures both (i) whether containment exists and (ii) discrimination between regions in inclusion chains

*Theoretical Significance*: RegD proves that Euclidean box embeddings can achieve hyperbolic-like expressiveness without the computational complexity of hyperbolic operations. This validates the choice of Euclidean space while maintaining theoretical guarantees.

*Relevance to Foundations*: This work builds directly on the max-stability property (see the Gumbel Max-Stability document) and volume calculations (see the Gumbel-Box Volume document), demonstrating that the mathematical structure we've established enables state-of-the-art performance.

== 2024: TransBox - EL++-closed Ontology Embedding

**Yang, Chen & Sattler (2024)** - arXiv:2410.14571

*Key Contribution*: Box embeddings for OWL ontologies with Description Logic (DL) axioms, ensuring logical closure properties.

*Innovations*:
- Extends box embeddings to handle complex DL axioms (not just simple triples)
- Ensures embeddings respect logical closure (EL++ semantics)
- Applications in healthcare and bioinformatics where ontologies are critical

*Theoretical Foundation*: This work relies on the subsumption relationship (see the Subsumption document) and demonstrates that box embeddings naturally encode logical subsumption in formal ontologies. The containment probability (see the Containment Probability document) provides the probabilistic foundation for handling uncertain or noisy ontological data.

*Practical Impact*: TransBox enables embedding of formal ontologies used in healthcare (SNOMED CT, UMLS) and bioinformatics, where logical consistency is critical. The geometric interpretation of subsumption makes these embeddings interpretable and verifiable.

== 2023: Concept2Box - Joint Geometric Embeddings

**Huang et al. (2023)** - arXiv:2307.01933 (ACL 2023)

*Key Contribution*: Jointly embeds two views of knowledge graphs: ontology-view concepts (as boxes) and instance-view entities (as vectors).

*Innovations*:
- *Dual geometric representations*: Concepts as box embeddings, entities as vector embeddings
- *Box volumes as granularity*: Interprets box volumes as concepts' granularity (probabilistic semantics)
- *Vector-to-box distance metric*: Novel metric to bridge concept boxes and entity vectors
- *Two-view modeling*: Addresses the fact that many KGs contain both high-level concepts and fine-grained entities

*Key Results*:
- Experiments on DBpedia KG and industrial KG show effectiveness
- Box embeddings capture hierarchy structure and complex relations (overlap, disjoint) among concepts
- Joint learning of both representations improves performance

*Mathematical Connection*: This work leverages the volume calculations (see the Gumbel-Box Volume document) to interpret box volumes as concept granularity. The containment probability (see the Containment Probability document) enables modeling of complex relationships between concepts and entities.

== 2023: Dual Box Embeddings for EL++

**Jackermeier, Chen & Horrocks (2023)** - arXiv:2301.11118

*Key Contribution*: Dual box embeddings for OWL ontologies, handling both ABox (data-level) and TBox (concept-level) knowledge.

*Innovations*:
- Dual box representation for concepts and individuals
- Handles complex DL axioms beyond simple triples
- Maintains logical consistency with EL++ semantics

*Theoretical Foundation*: This work demonstrates that box embeddings can handle both instance-level (ABox) and concept-level (TBox) knowledge, relying on the subsumption relationship (see the Subsumption document) and the max-stability property (see the Gumbel Max-Stability document) to maintain logical consistency.

== Recent Extensions: Beyond Traditional Boxes

=== Octagon Embeddings (2024)

**Charpenay & Schockaert (2024)** - arXiv:2401.16270

*Key Contribution*: Octagon embeddings represent relations as geometric regions (specifically octagons) in the embedding space, making the rules captured by the model explicit and enabling straightforward incorporation of prior knowledge.

*Innovations*:
- *Region-based representation*: Relations are represented as geometric regions rather than transformations, making learned patterns interpretable
- *Rule explicitness*: The geometric representation makes logical rules explicit, enabling inspection and validation of learned models
- *Prior knowledge integration*: The explicit geometric structure facilitates incorporating domain knowledge and logical constraints

*Relevance*: Demonstrates that region-based embeddings beyond boxes (octagons) can provide interpretability advantages. The explicit rule representation could inform future extensions of box embedding frameworks.

=== ExpressivE: Spatio-Functional Embeddings (2022)

**Pavlović & Sallinger (2022)** - ICLR 2023, arXiv:2206.04192

*Key Contribution*: ExpressivE embeds pairs of entities as points and relations as hyper-parallelograms in a virtual triple space $RR^(2d)$, enabling joint capture of a rich set of relational patterns.

*Innovations*:
- *Hyper-parallelogram representation*: Relations are represented as hyper-parallelograms rather than simple transformations
- *Spatial relationship encoding*: Relational patterns are characterized through spatial relationships of hyper-parallelograms
- *Joint pattern capture*: Supports symmetry, inversion, composition, and other relational patterns simultaneously

*Relevance*: Shows that more flexible geometric shapes (parallelograms) can capture complex relational patterns. The spatio-functional approach demonstrates how geometric embeddings can be extended beyond axis-aligned boxes while maintaining interpretability.

=== Geometric Algebra Embeddings

**Xu et al. (2020, 2022)** - arXiv:2010.00989, arXiv:2202.09464

*Key Contribution*: Knowledge graph embeddings using geometric algebras (Clifford algebras), extending beyond quaternions to more general algebraic structures.

*Innovations*:
- *Clifford algebra framework*: Uses geometric algebras $Cl_{p,q}$ to model entities and relations
- *Degenerate Clifford algebras*: Recent work (Kamdem Teyou et al., 2024) extends to nilpotent base vectors, enabling more expressive representations
- *Temporal extension*: Geometric algebra embeddings extended to temporal knowledge graphs

*Relevance*: Demonstrates that algebraic structures beyond real numbers (complex, quaternion, octonion, Clifford algebras) can provide additional expressivity. The geometric algebra framework provides a unified perspective on different embedding approaches.

=== Annular Sector Embeddings (2025)

**Zhu & Zeng (2025)** - arXiv:2506.11099

*Key Contribution*: Knowledge graph embeddings using annular sectors (regions between two concentric circles) to represent relations, providing a novel geometric structure for modeling relational patterns.

*Innovations*:
- *Annular sector representation*: Relations represented as sectors of annuli (regions between concentric circles)
- *Geometric pattern modeling*: The annular structure naturally captures certain relational patterns
- *Recent development*: Published in 2025, representing cutting-edge research in geometric embeddings

*Relevance*: Shows continued innovation in geometric embedding shapes beyond boxes, cones, and polygons. The annular sector approach demonstrates that different geometric structures can be optimal for different relational patterns.

=== Geometry Interaction Knowledge Graph Embeddings (GIE)

**Cao et al. (2022)** - arXiv:2206.12418

*Key Contribution*: Combines Euclidean, hyperbolic, and hyperspherical spaces with an attention mechanism to emphasize the most relevant geometry for different portions of knowledge graphs.

*Innovations*:
- *Mixed curvature spaces*: Simultaneously uses Euclidean (zero curvature), hyperbolic (negative curvature), and spherical (positive curvature) spaces
- *Geometry interaction*: Learns interactions between different geometric spaces through attention
- *Heterogeneous structure modeling*: Different portions of knowledge graphs are embedded in different geometric spaces based on their structure

*Relevance*: Demonstrates that combining multiple geometric spaces can overcome limitations of single-space embeddings. The attention mechanism for geometry selection could inform hybrid approaches combining box embeddings with other geometric structures.

=== Query2Box: Attention-Based Box Intersection

**Ren, Hu & Leskovec (2020)** - ICLR 2020

*Key Contribution*: Query2Box demonstrates that box embeddings excel at logical query answering, using an attention mechanism for box intersection and an entity-to-box distance metric.

*Innovations*:
- *Attention-based intersection*: The intersection of multiple boxes uses an attention mechanism over box centers: $Cen(p_"inter") = sum_i a_i odot Cen(p_i)$ where $a_i = exp(MLP(p_i)) / sum_j exp(MLP(p_j))$. This weighted average ensures the intersection center lies within the convex hull of input boxes while allowing the model to emphasize more relevant boxes.
- *Offset shrinking*: The intersection offset is computed as $Off(p_"inter") = Min({Off(p_i)}) odot sigma(DeepSets({p_i}))$, ensuring the intersection is contained within all input boxes while allowing adaptive shrinking based on the set of boxes.
- *Entity-to-box distance*: Uses a weighted combination $dist_"box"(v; q) = dist_"outside"(v; q) + alpha * dist_"inside"(v; q)$ where $alpha < 1$ (typically 0.2), ensuring entities inside boxes are "close enough" while penalizing entities outside boxes proportionally to their distance from the boundary.
- *Adaptive box sizing*: Box offsets (sizes) correlate with the number of answer entities—one-to-many relations tend to have larger offset embeddings, demonstrating that larger boxes model sets of more points.

*Mathematical Connection*: The attention mechanism for intersection builds on the geometric intersection operation (see the Log-Sum-Exp and Gumbel Intersection document), while the entity-to-box distance metric provides a bridge between point embeddings and box embeddings, enabling hybrid representations.

*DNF Transformation Strategy*: Query2Box handles disjunction by transforming EPFO queries to Disjunctive Normal Form (DNF). The transformation algorithm works as follows:

1. *Identify union nodes*: Find all nodes in the computation graph whose incoming edges are of type "union"
2. *Generate conjunctive queries*: For each union node $v$ with parent nodes $P_v$, generate $N = prod_(v in V_"union") |P_v|$ different computation graphs by selecting one parent for each union node
3. *Remove union edges*: Merge each union node with its selected parent, removing all union edges
4. *Create new target*: Convert all target sink nodes to existentially quantified bound variables, then create a new target sink node with union edges from all bound variables

This transformation is logically equivalent to the original query but enables low-dimensional embedding: each conjunctive query is embedded as a box in $d$-dimensional space, then the distance to the EPFO query is $dist_"agg"(v; q) = min({dist_"box"(v; q^((1))), ..., dist_"box"(v; q^((N)))$)$, taking the minimum distance to any box. This strategy avoids the theoretical requirement of dimension $Theta(|V|)$ for embedding EPFO queries as single points or boxes.

*Theoretical Foundation*: Query2Box proves that embedding EPFO queries as single points/boxes requires dimension proportional to the number of KG entities (VC dimension argument). The DNF transformation provides a practical workaround: by decomposing the query into conjunctive sub-queries, each can be embedded in low-dimensional space, then aggregated efficiently. This demonstrates that theoretical limitations can be overcome through clever algorithmic transformations.

*Practical Significance*: Query2Box achieves up to 25% relative improvement over Graph Query Embedding (GQE) on knowledge graph benchmarks, demonstrating that attention-based geometric operations can significantly improve performance on complex logical queries. The DNF transformation enables handling of arbitrary EPFO queries (conjunction, disjunction, existential quantification) while maintaining computational efficiency.

== Related Developments

=== Spherical Knowledge Graph Embeddings

**Quan et al. (2025)** - arXiv:2511.02460

*Key Contribution*: SKGE embeds knowledge graphs into spherical space with geometric regularization.

*Relevance*: While not box embeddings, this shows continued interest in non-Euclidean geometric embeddings, suggesting the field is exploring various geometric structures. RegD (2025) suggests that Euclidean box embeddings can achieve similar expressiveness with better computational properties.

=== Hyperbolic Embeddings

Multiple recent papers (2024-2025) continue to explore hyperbolic embeddings for hierarchical data:
- *Low-distortion Tree Embeddings in Hyperbolic Space* (van Spengler & Mettes, 2025)
- *Embeddings with constant additive error for hyperbolic spaces* (Park & Vigneron, 2024)

*Relevance*: These papers show that hyperbolic geometry remains active, but RegD (2025) suggests that Euclidean box embeddings can achieve similar expressiveness with better computational properties. The max-stability property of Gumbel distributions (see the Gumbel Max-Stability document) enables efficient computation in Euclidean space.

== Relation-Specific Patterns and Box Geometry

Different relation types manifest in distinct geometric patterns in box embedding space. Understanding these patterns provides interpretability and guides model design.

=== Symmetric Relations

*Geometric Pattern*: Symmetric relations (e.g., "married_to", "sibling_of") require that if $(h, r, t)$ holds, then $(t, r, h)$ also holds. In box embeddings, this can be modeled by ensuring that the relation box for $r$ is symmetric with respect to head and tail positions.

*BoxE Approach*: BoxE models symmetric relations by learning identical boxes for both argument positions: $Box_r^1 = Box_r^2$ where $Box_r^1$ is the box for head entities and $Box_r^2$ is the box for tail entities. This ensures that if entity $h$ is in $Box_r^1$ and entity $t$ is in $Box_r^2$, then the reverse also holds when the boxes are identical.

*Geometric Interpretation*: The symmetry constraint forces the relation box to be "centered" in a sense—it cannot favor one direction over another. This geometric constraint naturally encodes the logical property of symmetry.

=== Transitive Relations

*Geometric Pattern*: Transitive relations (e.g., "ancestor_of", "part_of") require that if $(A, r, B)$ and $(B, r, C)$ hold, then $(A, r, C)$ also holds. In box embeddings, transitivity is naturally encoded through containment: if box $A$ contains box $B$, and box $B$ contains box $C$, then box $A$ contains box $C$ by geometric transitivity.

*Subsumption Connection*: Transitive relations are a special case of subsumption hierarchies (see the Subsumption document). The geometric containment relationship directly models logical transitivity without requiring explicit constraints.

=== One-to-Many Relations

*Geometric Pattern*: One-to-many relations (e.g., "has_child", "contains") connect one entity to many others. Query2Box demonstrates that such relations have larger box offsets (sizes) compared to one-to-one relations.

*Empirical Evidence*: On FB15k, relations with largest box sizes include `/common/.../topic` (3616.0 average answers, box size 147.0) and `/people/.../spouse` (889.8 average answers, box size 114.3). Relations with smallest box sizes include `/architecture/.../owner` (1.0 average answers, box size 2.3).

*Interpretation*: Larger boxes naturally contain more entities, so one-to-many relations require larger boxes to encompass all answer entities. This adaptive sizing provides interpretability: box size directly indicates relation cardinality.

=== Composition Patterns

*Geometric Pattern*: Composition requires that if $(A, r_1, B)$ and $(B, r_2, C)$ hold, then there exists a relation $r_3$ such that $(A, r_3, C)$ holds. In box embeddings, composition can be modeled through box transformations: if relation $r_1$ transforms box $A$ to contain box $B$, and relation $r_2$ transforms box $B$ to contain box $C$, then the composition $r_3 = r_1 @ r_2$ should transform box $A$ to contain box $C$.

*BoxE Approach*: BoxE models composition through translational bumps: if $e_B = e_A + bump_(r_1)(e_A)$ and $e_C = e_B + bump_(r_2)(e_B)$, then $e_C = e_A + bump_(r_1)(e_A) + bump_(r_2)(e_A + bump_(r_1)(e_A))$. The composition relation $r_3$ learns a box that captures this composed transformation.

*Geometric Interpretation*: Composition corresponds to chaining geometric transformations. The learned composition relation box represents the "net effect" of applying two relations in sequence.

== Trends and Insights

=== 1. Hybrid Representations

Recent work (Concept2Box, RegD) suggests combining different geometric structures (boxes + vectors, boxes + depth metrics) rather than using a single representation. This hybrid approach leverages the strengths of each representation:
- Boxes for hierarchical/containment relationships
- Vectors for symmetric/similarity relationships
- Depth metrics for enhanced expressiveness

=== 2. Uncertainty Quantification

Multiple papers (NFE 2024, UKGE 2021) focus on introducing uncertainty into embeddings, which aligns with our probabilistic box embeddings approach. The Gumbel distribution provides a natural way to quantify uncertainty through the scale parameter $beta$, and the containment probability (see the Containment Probability document) enables probabilistic reasoning.

=== 3. Formal Logic Integration

Papers on EL++ and ontology embeddings (TransBox, Dual Box Embeddings) show box embeddings are being applied to formal knowledge representation, validating the subsumption-based approach. The geometric interpretation of subsumption (see the Subsumption document) makes these embeddings interpretable and verifiable.

=== 4. Computational Efficiency

RegD (2025) emphasizes that Euclidean box embeddings can achieve hyperbolic-like expressiveness without the computational complexity of hyperbolic operations. The max-stability property (see the Gumbel Max-Stability document) enables efficient computation, and the log-sum-exp function (see the Log-Sum-Exp and Gumbel Intersection document) provides numerical stability.

=== 5. Two-View Knowledge Graphs

Concept2Box highlights that real-world KGs often have two views (concepts vs entities), suggesting that hybrid box+vector representations are practically important. This validates the need for flexible embedding frameworks that can handle multiple geometric structures.

== Applications in Practice

=== Knowledge Graph Completion

Box embeddings excel at link prediction in knowledge graphs, where the task is to predict missing relationships. The containment probability (see the Containment Probability document) provides a natural scoring function, and the local identifiability solution (see the Local Identifiability document) enables effective gradient-based learning.

=== Ontology Embedding

Formal ontologies (OWL, Description Logic) require logical consistency, which box embeddings naturally provide through geometric subsumption (see the Subsumption document). Applications include:
- Healthcare: SNOMED CT, UMLS medical ontologies
- Bioinformatics: Gene ontologies, protein function hierarchies
- Semantic web: OWL ontologies for knowledge representation

=== Natural Language Understanding

Box embeddings can model hyponym-hypernym relationships in natural language, where containment represents semantic subsumption. For example, "dog" is a hyponym of "mammal", which is a hyponym of "animal"—this hierarchy is naturally encoded as nested boxes.

=== Two-View Knowledge Graphs

Many real-world knowledge graphs contain both high-level concepts and fine-grained entities. Concept2Box demonstrates that joint learning of concepts (as boxes) and entities (as vectors) improves performance, validating the hybrid approach.

== Performance Benchmarks

Recent work demonstrates box embeddings' effectiveness:

- *Concept2Box (2023)*: Shows effectiveness on DBpedia KG and industrial KGs, achieving competitive performance with point-based methods while providing interpretability through geometric structure
- *RegD (2025)*: Achieves hyperbolic-like expressiveness with Euclidean efficiency, demonstrating that box embeddings can capture hierarchical structures without the computational complexity of hyperbolic operations
- *TransBox (2024)*: Handles complex DL axioms while maintaining logical consistency, validating the subsumption-based approach for formal ontologies
- *Dual Box Embeddings (2023)*: Successfully embeds both ABox and TBox knowledge, showing that box embeddings can handle both instance-level and concept-level knowledge simultaneously
- *BoxE (2020)*: Achieves state-of-the-art performance on FB15k-237 and WN18RR knowledge graph completion benchmarks, proving full expressivity with dimension $d = |E|^(n-1) * |R|$
- *Query2Box (2020)*: Achieves up to 25% relative improvement over Graph Query Embedding (GQE) on logical query answering tasks, demonstrating that attention-based geometric operations significantly improve performance

These results validate the mathematical foundations we've established: the subsumption relationship, volume calculations, max-stability, and local identifiability all contribute to practical effectiveness. The $O(n * d)$ computational complexity (where $n$ is relation arity and $d$ is dimension) makes box embeddings efficient even for large knowledge graphs, while the VC dimension bound of $2d$ provides theoretical guarantees on representational capacity.

== Connection to Mathematical Foundations

All these applications build on the mathematical foundations established in the previous documents:

- *Subsumption*: Geometric containment enables logical subsumption in ontologies
- *Gumbel Box Volume*: Volume calculations enable granularity interpretation and depth metrics
- *Containment Probability*: Probabilistic reasoning enables uncertainty quantification
- *Gumbel Max-Stability*: Algebraic closure enables efficient computation and logical consistency
- *Log-Sum-Exp*: Numerical stability enables practical implementation
- *Local Identifiability*: Smooth gradients enable effective learning

The theoretical elegance of the mathematical foundations translates directly to practical effectiveness in modern applications.

== Next Steps

For future directions and open questions, see the Future Directions document. For implementation details, see the code documentation and practical guides.

