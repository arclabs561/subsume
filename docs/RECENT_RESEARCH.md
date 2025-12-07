# Recent Research in Box Embeddings (2023-2025)

This document summarizes the most recent developments in box embeddings and related geometric embedding methods from 2023-2025.

## 2025 Papers

### Geometric Relational Embeddings: Progress and Prospects
**Xiong et al. (2025)** - Handbook on Neurosymbolic AI and Knowledge Graphs, DOI: 10.3233/FAIA250208

**Key Contribution**: A comprehensive survey of geometric relational embeddings, providing a systematic taxonomy and analysis of methods based on embedding geometries and relational reasoning tasks.

**Taxonomy**:
- **Distribution-based**: Gaussian, Dirichlet, Beta, Gamma embeddings
- **Region-based**: Ball, Box, Cone, Polygon embeddings
- **Manifold-based**: Hyperbolic, Spherical, Mixed manifold embeddings
- **Hybrid**: Combinations of the above (e.g., Gumbel Box, Hyperbolic Cone)

**Key Insights**:
- **Box embeddings** are classified as region-based embeddings that excel at modeling set theory (inclusion, exclusion, overlap, difference) and logical operations (intersection, union, negation)
- **Gumbel Box** is highlighted as a hybrid method combining Gumbel distributions with box embeddings to solve local identifiability problems
- **BoxEL** and related methods are discussed for ontology embedding (EL++ description logic)
- **BoxE** is noted for its ability to model n-ary relations and full expressiveness
- **Query2Box** is discussed for logical query answering using box embeddings

**Applications Covered**:
1. Knowledge Graph Completion (BoxE, BEUrRE, ShrinkE)
2. Ontology/Hierarchy Completion (BoxEL, ELBE, Box2EL)
3. Hierarchical Multi-label Classification (MBM, Box4Type)
4. Logical Query Answering (Query2Box, ConE)

**Key Technical Details**:
- **Box embeddings** are classified as region-based embeddings that excel at modeling set theory (inclusion, exclusion, overlap, difference) and logical operations (intersection, union, negation)
- **Gumbel Box** is highlighted as a hybrid method combining Gumbel distributions with box embeddings to solve local identifiability problems
- **BoxEL** and related methods are discussed for ontology embedding (EL++ description logic)
- **BoxE** is noted for its ability to model n-ary relations and full expressiveness
- **Query2Box** is discussed for logical query answering using box embeddings

**Future Directions Identified**:
1. **Heterogeneous hierarchies**: Most methods encode one hierarchy relation (e.g., is_a), but real-world KGs contain multiple hierarchical relations (e.g., is_a and has_part) that may be intertwined
2. **Deep geometric embeddings**: Most current methods are non-parametric; developing deep architectures is an exciting direction
3. **Learning with symbolic knowledge**: Incorporating logical constraints (exclusion, intersection equivalence) beyond hierarchical constraints

**Relevance to `subsume`**: **Highly Relevant** - This survey provides comprehensive context for box embeddings within the broader landscape of geometric relational embeddings. It validates our approach by showing box embeddings are a well-established category with proven applications across multiple domains. The taxonomy helps understand where box embeddings fit relative to other geometric methods (hyperbolic, spherical, distribution-based). The identified future directions (heterogeneous hierarchies, deep architectures, symbolic constraints) align with potential extensions to our library. The survey's systematic categorization by geometry type and application domain provides a roadmap for understanding the field's evolution.

### Out-of-the-Box Conditional Text Embeddings from Large Language Models
**Yamada & Zhang (2025)** - arXiv:2504.16411

**Key Contribution**: PonTE (Prompt-based conditional Text Embedding) generates conditional text embeddings using causal LLMs with prompts, without fine-tuning.

**Innovations**:
- **Unsupervised conditional embeddings**: Uses prompts like "Express this text '{A}' in one word in terms of {B}:" to generate embeddings that vary based on conditions
- **No fine-tuning required**: Leverages instruction-tuned LLMs (Mistral, Llama) to generate embeddings directly from hidden states
- **Interpretability**: Generated words following prompts provide interpretability of embeddings
- **Performance**: Achieves performance comparable to supervised methods on conditional semantic text similarity (C-STS) tasks

**Relevance to `subsume`**: **Indirect** - While not directly about box embeddings, this work demonstrates the value of conditional/multi-aspect representations. The same text can have different embeddings based on different conditions (e.g., "product category" vs "rating"). This concept could inspire **conditional box embeddings**, where the same concept could have different box representations based on different aspects or contexts. For example, "dog" might have different box embeddings when conditioned on "taxonomy" vs "behavior" vs "size". This could be useful for multi-faceted knowledge representation where entities have different hierarchical relationships depending on the aspect being considered.

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

### 6. **Conditional/Multi-Aspect Representations**
PonTE (2025) demonstrates the value of conditional embeddings where the same entity can have different representations based on different aspects or contexts. This could inspire conditional box embeddings for multi-faceted knowledge representation.

### 7. **Logical Query Answering with Box Embeddings**
Query2Box (Ren et al., 2020) demonstrates that box embeddings excel at logical query answering beyond simple containment. The geometric intersection operator (using attention) and entity-to-box distance metrics are directly applicable to our library. The DNF transformation strategy for handling disjunction shows how to scale box embeddings to complex logical queries while maintaining low-dimensional representations.

### 8. **Comprehensive Taxonomy**
The Xiong et al. (2025) survey provides a systematic taxonomy of geometric relational embeddings, categorizing methods by geometry type (distribution-based, region-based, manifold-based, hybrid) and application domain. This taxonomy helps understand where box embeddings fit in the broader landscape and identifies complementary approaches (e.g., hyperbolic embeddings for hierarchies, distribution-based for uncertainty).

## Implications for `subsume`

1. **Validation**: Recent papers continue to validate box embeddings as an effective approach for hierarchical and containment relationships.

2. **Extensions to Consider**:
   - Vector-to-box distance metrics (from Concept2Box)
   - Depth-based distance metrics (from RegD)
   - Support for hybrid box+vector representations
   - Enhanced uncertainty quantification methods
   - Cone embeddings for negation support (as discussed in Xiong et al. survey)
   - Polygon embeddings for n-ary relations (ExpressivE, PolygonE)

3. **Applications**:
   - Two-view knowledge graphs (concepts as boxes, entities as vectors)
   - Formal ontologies (EL++, OWL)
   - Healthcare and bioinformatics (where ontologies are critical)

4. **Theoretical Developments**:
   - RegD proves that Euclidean box embeddings can achieve hyperbolic-like expressiveness
   - This validates our choice of Euclidean space while maintaining theoretical guarantees

## Mathematical Foundations and Theoretical Properties

### Lattice-Theoretic Foundations

Box embeddings are fundamentally grounded in **lattice theory** and **order theory**, providing a geometric representation system that extends beyond traditional vector-based approaches. The mathematical structure establishes box embeddings as a proper **box lattice** with well-defined lattice operations.

**Core Lattice Structure**: Box embeddings form a lattice under the **reverse product order** on \(\mathbb{R}^n_+\), where an embedding is below another in a hierarchy if all of its coordinates are larger. This ordering creates a natural partial order structure—the foundation of lattice theory. The **join operation** (union) produces the smallest enclosing box that contains both input boxes, while the **meet operation** (intersection) produces the intersection of two boxes. Critically, this lattice structure is **strictly more general** than order embedding lattices in any dimension.

**Intersectional Closure**: A fundamental property is **intersectional closure**: the intersection of two boxes (representing concepts) is itself a box. This ensures the lattice remains closed under the meet operation, essential for modeling concept hierarchies where any two concepts maintain a well-defined relationship.

**Lattice-Preserving Properties**: The mathematical framework ensures embeddings function as **lattice-preserving mappings**. If a loss function \(L = 0\), then the embedding function \(f_\theta\) preserves lattice relationships: whenever \(A \sqsubseteq B\) in the conceptual lattice \((\tilde{C}, \sqsubseteq)\), we have \(f_\theta(A) \preceq f_\theta(B)\) in the embedded space \((X, \preceq)\).

### Order-Theoretic Foundations

The theoretical foundation of box embeddings rests on **order theory**. The "enclose" relation between a child box and parent box creates a **poset (partially ordered set)** where the order relation represents hierarchical containment. The asymmetrical nature of this relation—a fundamental property of hierarchies—cannot be captured by symmetric distance metrics in vector spaces, making box embeddings superior for representing parent-child relationships.

Unlike traditional vector embeddings (which represent entities as single points and can only measure **symmetrical similarity** through distance metrics), box embeddings capture **asymmetrical hierarchical relations**. Vector embeddings inherently cannot differentiate parent and child nodes in a pair, whereas boxes can: a child box is entirely enclosed inside its parent box, and completely separate boxes represent non-hierarchical entities.

### Probabilistic Extension and Geometric Probability

Box embeddings extend into **probabilistic lattice theory** through volume-based measures. Probabilities associated with concepts are derived from the **volume of boxes in [0,1]ⁿ**. This probabilistic interpretation enables modeling of:

- **Disjoint concepts** (exactly -1 correlation when total volume equals 1)
- **Identical concepts** (correlation of 1)
- **Negative correlation**, which previous order embedding models could never achieve

The box lattice can represent all possible correlations between pairs of variables through the continuity of the correlation function with respect to box translations and intersections.

### Gumbel Max-Stability and Algebraic Closure

The Gumbel distribution's **max-stability property** is fundamental to Gumbel box embeddings. This property ensures that the maximum of independent Gumbel random variables remains Gumbel-distributed, providing **algebraic closure** for box operations. This closure property is essential for maintaining the probabilistic interpretation while enabling gradient-based learning through the Gumbel-Softmax reparameterization trick.

The max-stability property directly enables the analytical derivation of expected intersection volumes, which would be intractable for arbitrary distributions. The connection to **log-sum-exp** functions provides numerical stability in high-dimensional settings, as log-sum-exp is a smooth approximation to the maximum function that naturally appears in Gumbel intersection calculations.

## Recent Extensions and Variations (2024-2025)

### Octagon Embeddings for Knowledge Graphs

**Charpenay & Schockaert (2024)** - arXiv:2401.16270

**Key Contribution**: Octagon embeddings represent relations as geometric regions (specifically octagons) in the embedding space, making the rules captured by the model explicit and enabling straightforward incorporation of prior knowledge.

**Innovations**:
- **Region-based representation**: Relations are represented as geometric regions rather than transformations, making learned patterns interpretable
- **Rule explicitness**: The geometric representation makes logical rules explicit, enabling inspection and validation of learned models
- **Prior knowledge integration**: The explicit geometric structure facilitates incorporating domain knowledge and logical constraints

**Relevance to `subsume`**: Demonstrates that region-based embeddings beyond boxes (octagons) can provide interpretability advantages. The explicit rule representation could inform future extensions of our library for explainable reasoning.

### ExpressivE: Spatio-Functional Embeddings

**Pavlović & Sallinger (2022)** - ICLR 2023, arXiv:2206.04192

**Key Contribution**: ExpressivE embeds pairs of entities as points and relations as hyper-parallelograms in a virtual triple space \(\mathbb{R}^{2d}\), enabling joint capture of a rich set of relational patterns.

**Innovations**:
- **Hyper-parallelogram representation**: Relations are represented as hyper-parallelograms rather than simple transformations
- **Spatial relationship encoding**: Relational patterns are characterized through spatial relationships of hyper-parallelograms
- **Joint pattern capture**: Supports symmetry, inversion, composition, and other relational patterns simultaneously

**Relevance to `subsume`**: Shows that more flexible geometric shapes (parallelograms) can capture complex relational patterns. The spatio-functional approach demonstrates how geometric embeddings can be extended beyond axis-aligned boxes while maintaining interpretability.

### Geometric Algebra Embeddings

**Xu et al. (2020, 2022)** - arXiv:2010.00989, arXiv:2202.09464

**Key Contribution**: Knowledge graph embeddings using geometric algebras (Clifford algebras), extending beyond quaternions to more general algebraic structures.

**Innovations**:
- **Clifford algebra framework**: Uses geometric algebras \(Cl_{p,q}\) to model entities and relations
- **Degenerate Clifford algebras**: Recent work (Kamdem Teyou et al., 2024) extends to nilpotent base vectors, enabling more expressive representations
- **Temporal extension**: Geometric algebra embeddings extended to temporal knowledge graphs

**Relevance to `subsume`**: Demonstrates that algebraic structures beyond real numbers (complex, quaternion, octonion, Clifford algebras) can provide additional expressivity. The geometric algebra framework provides a unified perspective on different embedding approaches.

### Annular Sector Embeddings

**Zhu & Zeng (2025)** - arXiv:2506.11099

**Key Contribution**: Knowledge graph embeddings using annular sectors (regions between two concentric circles) to represent relations, providing a novel geometric structure for modeling relational patterns.

**Innovations**:
- **Annular sector representation**: Relations represented as sectors of annuli (regions between concentric circles)
- **Geometric pattern modeling**: The annular structure naturally captures certain relational patterns
- **Recent development**: Published in 2025, representing cutting-edge research in geometric embeddings

**Relevance to `subsume`**: Shows continued innovation in geometric embedding shapes beyond boxes, cones, and polygons. The annular sector approach demonstrates that different geometric structures can be optimal for different relational patterns.

### Universal Orthogonal Parameterization

**Li et al. (2024)** - arXiv:2405.08540

**Key Contribution**: A universal orthogonal parameterization framework that generalizes knowledge graph embeddings with flexible dimension and heterogeneous geometric spaces.

**Innovations**:
- **Universal orthogonalization**: Extends beyond rigid relational orthogonalization to flexible parameterization
- **Heterogeneous spaces**: Supports mixed geometric spaces rather than homogeneous spaces
- **Dimension flexibility**: Allows adaptive dimensionality rather than fixed dimensions

**Relevance to `subsume`**: Addresses limitations of fixed geometric structures by enabling adaptive parameterization. This could inform future extensions of box embeddings to support more flexible geometric configurations.

### Geometry Interaction Knowledge Graph Embeddings (GIE)

**Cao et al. (2022)** - arXiv:2206.12418

**Key Contribution**: Combines Euclidean, hyperbolic, and hyperspherical spaces with an attention mechanism to emphasize the most relevant geometry for different portions of knowledge graphs.

**Innovations**:
- **Mixed curvature spaces**: Simultaneously uses Euclidean (zero curvature), hyperbolic (negative curvature), and spherical (positive curvature) spaces
- **Geometry interaction**: Learns interactions between different geometric spaces through attention
- **Heterogeneous structure modeling**: Different portions of knowledge graphs are embedded in different geometric spaces based on their structure

**Relevance to `subsume`**: Demonstrates that combining multiple geometric spaces can overcome limitations of single-space embeddings. The attention mechanism for geometry selection could inform hybrid approaches combining box embeddings with other geometric structures.

## Theoretical Limitations and Proofs

### Expressivity Limitations

While BoxE proves **full expressivity** for knowledge graph completion with dimensionality \(d = |E|^{n-1}|R|\), this dimensionality grows rapidly with the number of entities and relation arities. For large knowledge bases, this suggests that full expressivity may not translate to practical expressivity with reasonable parameter budgets.

**Formal Limitations**: Research has established formal limitations of box embeddings for description logic ontologies. Box embeddings cannot faithfully represent certain description logic axioms even with appropriate parameter tuning, particularly complex intersection patterns and qualified role restrictions. These limitations stem from geometric constraints of axis-aligned boxes: arbitrary collections of axis-aligned hyperrectangles cannot exactly represent all possible concept hierarchies.

### Optimization Challenges

Box embeddings present significant optimization challenges distinct from point-based embeddings. Large flat regions of local minima occur when disjoint boxes should remain separated, yet optimization discovers minimal-energy configurations where boxes approach boundaries without crossing. These flat regions substantially slow convergence and can trap optimization in suboptimal configurations.

The **local identifiability problem** arises because multiple different box configurations can produce identical loss values, creating flat regions where gradient descent cannot effectively guide learning. This problem is particularly acute when boxes do not intersect, resulting in zero intersection volume and preventing gradients from flowing through the intersection operation.

### Closure Properties

Boxes are **closed under intersection** but **not under union** in the classical sense. While the intersection of two axis-aligned boxes is always an axis-aligned box (or empty), the union of two boxes may require many boxes to represent exactly. This asymmetry limits expressiveness for certain types of relational structures requiring symmetric treatment of union and intersection operations.

## References

1. Xiong, B., Nayyeri, M., Jin, M., He, Y., Cochez, M., Pan, S., & Staab, S. (2025). Geometric Relational Embeddings: Progress and Prospects. In P. Hitzler, A. Dalal, M. S. Mahdavinejad, & S. S. Norouzi (Eds.), Handbook on Neurosymbolic AI and Knowledge Graphs (pp. 213-229). IOS Press. DOI: 10.3233/FAIA250208

2. Yamada, K., & Zhang, P. (2025). Out-of-the-Box Conditional Text Embeddings from Large Language Models. arXiv:2504.16411

3. Yang, H., & Chen, J. (2025). Achieving Hyperbolic-Like Expressiveness with Arbitrary Euclidean Regions: A New Approach to Hierarchical Embeddings. arXiv:2501.17518

4. Yang, H., Chen, J., & Sattler, U. (2024). TransBox: EL++-closed Ontology Embedding. arXiv:2410.14571

5. Xiao, C., He, X., & Cao, Y. (2024). Knowledge Graph Embedding by Normalizing Flows. arXiv:2409.19977

6. Huang, Z., et al. (2023). Concept2Box: Joint Geometric Embeddings for Learning Two-View Knowledge Graphs. ACL 2023. arXiv:2307.01933

7. Jackermeier, M., Chen, J., & Horrocks, I. (2023). Dual Box Embeddings for the Description Logic EL++. arXiv:2301.11118

8. Ren, H., Hu, W., & Leskovec, J. (2020). Query2Box: Reasoning Over Knowledge Graphs in Vector Space Using Box Embeddings. ICLR 2020.

9. Quan, X.-T., et al. (2025). SKGE: Spherical Knowledge Graph Embedding with Geometric Regularization. arXiv:2511.02460

10. Charpenay, V., & Schockaert, S. (2024). Capturing Knowledge Graphs and Rules with Octagon Embeddings. arXiv:2401.16270

11. Pavlović, A., & Sallinger, E. (2022). ExpressivE: A Spatio-Functional Embedding For Knowledge Graph Completion. ICLR 2023. arXiv:2206.04192

12. Xu, C., Nayyeri, M., Chen, Y.-Y., & Lehmann, J. (2020). Knowledge Graph Embeddings in Geometric Algebras. arXiv:2010.00989

13. Xu, C., Nayyeri, M., Chen, Y.-Y., & Lehmann, J. (2022). Geometric Algebra based Embeddings for Static and Temporal Knowledge Graph Completion. arXiv:2202.09464

14. Zhu, H., & Zeng, Y. (2025). Knowledge Graph Embeddings with Representing Relations as Annular Sectors. arXiv:2506.11099

15. Li, R., et al. (2024). Generalizing Knowledge Graph Embedding with Universal Orthogonal Parameterization. arXiv:2405.08540

16. Cao, Z., et al. (2022). Geometry Interaction Knowledge Graph Embeddings. arXiv:2206.12418

17. Vilnis, L., Li, X., Murty, S., & McCallum, A. (2018). Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures. ACL 2018. arXiv:1805.06627

