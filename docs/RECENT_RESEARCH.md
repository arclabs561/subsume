# Recent Research in Box Embeddings (2023-2025)

This document summarizes the most recent developments in box embeddings and related geometric embedding methods from 2023-2025.

## 2025 Papers

### RegD: Achieving Hyperbolic-Like Expressiveness with Arbitrary Euclidean Regions
**Yang & Chen (2025)** - arXiv:2501.17518

**Key Contribution**: A flexible Euclidean framework that supports arbitrary geometric regions (boxes, balls) as embeddings while achieving hyperbolic-like expressiveness.

**Innovations**:
- **Depth-based dissimilarity**: Incorporates depth distance and boundary distance metrics to emulate hyperbolic geometry properties
- **Eliminates precision issues**: Uses elementary arithmetic operations instead of specialized hyperbolic constructions
- **Addresses crowding effect**: As the number of children increases in hierarchies, traditional Euclidean embeddings cluster together; RegD's depth distance solves this by incorporating region "size"
- **Boundary distance for containment**: When one region is fully contained within another, boundary distance captures both (i) whether containment exists and (ii) discrimination between regions in inclusion chains

**Relevance to `subsume`**: This work validates the approach of using box embeddings (as one type of "arbitrary Euclidean region") and provides theoretical justification for depth-based metrics that could enhance our distance calculations.

### TransBox: EL++-closed Ontology Embedding
**Yang, Chen & Sattler (2024)** - arXiv:2410.14571

**Key Contribution**: Box embeddings for OWL ontologies with Description Logic (DL) axioms, ensuring logical closure properties.

**Innovations**:
- Extends box embeddings to handle complex DL axioms (not just simple triples)
- Ensures embeddings respect logical closure (EL++ semantics)
- Applications in healthcare and bioinformatics where ontologies are critical

**Relevance to `subsume`**: Demonstrates box embeddings' applicability to formal ontologies, which aligns with our subsumption-based approach.

## 2024 Papers

### Knowledge Graph Embedding by Normalizing Flows
**Xiao, He & Cao (2024)** - arXiv:2409.19977

**Key Contribution**: Introduces uncertainty into knowledge graph embeddings using normalizing flows and group theory.

**Innovations**:
- Embeds entities/relations as permutations of random variables (symmetric group elements)
- Uses normalizing flows to transform simple random variables into complex ones for expressiveness
- Proves ability to learn logical rules
- Provides unified perspective that can incorporate existing models

**Relevance to `subsume`**: While not directly about boxes, this work on uncertainty quantification in embeddings is complementary to our probabilistic box embeddings approach.

## 2023 Papers

### Concept2Box: Joint Geometric Embeddings for Learning Two-View Knowledge Graphs
**Huang et al. (2023)** - arXiv:2307.01933 (ACL 2023)

**Key Contribution**: Jointly embeds two views of knowledge graphs: ontology-view concepts (as boxes) and instance-view entities (as vectors).

**Innovations**:
- **Dual geometric representations**: Concepts as box embeddings, entities as vector embeddings
- **Box volumes as granularity**: Interprets box volumes as concepts' granularity (probabilistic semantics)
- **Vector-to-box distance metric**: Novel metric to bridge concept boxes and entity vectors
- **Two-view modeling**: Addresses the fact that many KGs contain both high-level concepts and fine-grained entities

**Key Results**:
- Experiments on DBpedia KG and industrial KG show effectiveness
- Box embeddings capture hierarchy structure and complex relations (overlap, disjoint) among concepts
- Joint learning of both representations improves performance

**Relevance to `subsume`**: Directly relevant! This paper uses box embeddings for concepts and demonstrates their effectiveness. The vector-to-box distance metric could be useful for our library if we want to support hybrid representations.

### Dual Box Embeddings for the Description Logic EL++
**Jackermeier, Chen & Horrocks (2023)** - arXiv:2301.11118

**Key Contribution**: Dual box embeddings for OWL ontologies, handling both ABox (data-level) and TBox (concept-level) knowledge.

**Innovations**:
- Dual box representation for concepts and individuals
- Handles complex DL axioms beyond simple triples
- Maintains logical consistency with EL++ semantics

**Relevance to `subsume`**: Another demonstration of box embeddings for formal ontologies, validating the subsumption-based approach.

## Related Geometric Embedding Developments

### Spherical Knowledge Graph Embeddings
**Quan et al. (2025)** - arXiv:2511.02460

**Key Contribution**: SKGE embeds knowledge graphs into spherical space with geometric regularization.

**Relevance**: While not box embeddings, this shows continued interest in non-Euclidean geometric embeddings, suggesting the field is exploring various geometric structures.

### Hyperbolic Embeddings
Multiple recent papers (2024-2025) continue to explore hyperbolic embeddings for hierarchical data:
- **Low-distortion Tree Embeddings in Hyperbolic Space** (van Spengler & Mettes, 2025)
- **Embeddings with constant additive error for hyperbolic spaces** (Park & Vigneron, 2024)

**Relevance**: These papers show that hyperbolic geometry remains active, but RegD (2025) suggests that Euclidean box embeddings can achieve similar expressiveness with better computational properties.

## Trends and Insights

### 1. **Hybrid Representations**
Recent work (Concept2Box, RegD) suggests combining different geometric structures (boxes + vectors, boxes + depth metrics) rather than using a single representation.

### 2. **Uncertainty Quantification**
Multiple papers (NFE 2024, UKGE 2021) focus on introducing uncertainty into embeddings, which aligns with our probabilistic box embeddings approach.

### 3. **Formal Logic Integration**
Papers on EL++ and ontology embeddings (TransBox, Dual Box Embeddings) show box embeddings are being applied to formal knowledge representation, validating the subsumption-based approach.

### 4. **Computational Efficiency**
RegD (2025) emphasizes that Euclidean box embeddings can achieve hyperbolic-like expressiveness without the computational complexity of hyperbolic operations, which is a key advantage.

### 5. **Two-View Knowledge Graphs**
Concept2Box highlights that real-world KGs often have two views (concepts vs entities), suggesting our library might benefit from supporting hybrid box+vector representations.

## Implications for `subsume`

1. **Validation**: Recent papers continue to validate box embeddings as an effective approach for hierarchical and containment relationships.

2. **Extensions to Consider**:
   - Vector-to-box distance metrics (from Concept2Box)
   - Depth-based distance metrics (from RegD)
   - Support for hybrid box+vector representations
   - Enhanced uncertainty quantification methods

3. **Applications**:
   - Two-view knowledge graphs (concepts as boxes, entities as vectors)
   - Formal ontologies (EL++, OWL)
   - Healthcare and bioinformatics (where ontologies are critical)

4. **Theoretical Developments**:
   - RegD proves that Euclidean box embeddings can achieve hyperbolic-like expressiveness
   - This validates our choice of Euclidean space while maintaining theoretical guarantees

## References

1. Yang, H., & Chen, J. (2025). Achieving Hyperbolic-Like Expressiveness with Arbitrary Euclidean Regions: A New Approach to Hierarchical Embeddings. arXiv:2501.17518

2. Yang, H., Chen, J., & Sattler, U. (2024). TransBox: EL++-closed Ontology Embedding. arXiv:2410.14571

3. Xiao, C., He, X., & Cao, Y. (2024). Knowledge Graph Embedding by Normalizing Flows. arXiv:2409.19977

4. Huang, Z., et al. (2023). Concept2Box: Joint Geometric Embeddings for Learning Two-View Knowledge Graphs. ACL 2023. arXiv:2307.01933

5. Jackermeier, M., Chen, J., & Horrocks, I. (2023). Dual Box Embeddings for the Description Logic EL++. arXiv:2301.11118

6. Quan, X.-T., et al. (2025). SKGE: Spherical Knowledge Graph Embedding with Geometric Regularization. arXiv:2511.02460

