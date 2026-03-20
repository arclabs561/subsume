# subsume examples

## Examples

| Example | What it demonstrates |
|---------|---------------------|
| `containment_hierarchy` | Hand-constructed box hierarchy; containment/overlap probabilities; temperature effects on sharpness |
| `gumbel_box_exploration` | Gumbel box properties: soft containment, membership probability, temperature annealing (no training) |
| `cone_training` | Training cone embeddings on an 18-entity taxonomy (3+ levels); aperture reflects generality |
| `box_training` | Training box embeddings with direct coordinate updates on a 25-entity taxonomy; volume reflects generality |
| `query2box` | Query2Box-style compositional query answering: multi-hop KG queries via box intersection, containment ranking, and alpha-weighted distance scoring |
| `octagon_demo` | Octagon embeddings: diagonal constraints, point containment, intersection (closure), volume comparison, soft containment/overlap |
| `fuzzy_query` | Fuzzy query answering with t-norms (Min/Product/Lukasiewicz), t-conorms, negation, and De Morgan duality on a small KG |
| `taxobell_demo` | TaxoBell Gaussian box losses on a mini taxonomy (no training, loss inspection only) |
| `dataset_training` | Full pipeline: load WN18RR-format dataset, train box embeddings, evaluate with MRR/Hits@k |
| `save_checkpoint` | Generate the pretrained WordNet subset checkpoint (writes `pretrained/wordnet_subset.json`) |
| `imagenet_hierarchy` | Real-scale training on 252 Tiny ImageNet WordNet synsets; volume-depth correlation (Spearman) |
| `hyperbolic_demo` | Poincare ball embeddings: norm-depth correlation, hierarchy preservation, exponential capacity |
| `el_training` | End-to-end EL++ box embedding training on a biomedical-style ontology |
| `taxobell_training` | TaxoBell MLP encoder training with Candle autograd (requires `--features candle-backend`) |

## Decision tree

- **Want to understand box geometry (containment, overlap, volume)?**
  Start with `containment_hierarchy`.

- **Want to understand Gumbel temperature effects and soft membership?**
  Start with `gumbel_box_exploration`.

- **Want to answer multi-hop knowledge graph queries with box intersection?**
  Start with `query2box`.

- **Want to understand octagon embeddings (boxes + diagonal constraints)?**
  Start with `octagon_demo`.

- **Want to explore fuzzy logic operators for query answering?**
  Start with `fuzzy_query`.

- **Want to embed a tree-like hierarchy in hyperbolic space?**
  Start with `hyperbolic_demo`.

- **Want to train embeddings on a hierarchy?**
  - Real-scale (252 entities, Tiny ImageNet synsets): `imagenet_hierarchy`
  - Dataset pipeline (load, train, evaluate with metrics): `dataset_training`
  - Cone model (angular containment, supports negation): `cone_training`
  - Box model (axis-aligned hyperrectangles, volume-based): `box_training`
  - EL++ ontology (subsumption + roles): `el_training`
  - TaxoBell (Gaussian boxes + MLP encoder, Candle): `taxobell_training`

- **Want to see TaxoBell losses without training?**
  Start with `taxobell_demo`.

## Running

```bash
cargo run -p subsume --example containment_hierarchy
cargo run -p subsume --example gumbel_box_exploration
cargo run -p subsume --example cone_training --release
cargo run -p subsume --example box_training --release
cargo run -p subsume --example query2box
cargo run -p subsume --example octagon_demo
cargo run -p subsume --example fuzzy_query
cargo run -p subsume --example taxobell_demo
cargo run -p subsume --example dataset_training --release
cargo run -p subsume --example imagenet_hierarchy --release
cargo run -p subsume --features hyperbolic --example hyperbolic_demo
cargo run -p subsume --example el_training --release
cargo run -p subsume --features candle-backend --example taxobell_training --release
```

## Example output

Each example below shows actual output from a single run.

<details>
<summary><b>containment_hierarchy</b> -- box geometry basics</summary>

```text
=== Box Embeddings: Containment Hierarchy ===

--- Volumes (larger = more general) ---

    animal: volume = 1.0000
    mammal: volume = 0.7290
       dog: volume = 0.0270
       cat: volume = 0.0480
      fish: volume = 0.0048

--- Containment P(row inside col) ---

              animal    mammal       dog       cat      fish
    animal     1.000     0.729     0.027     0.048     0.005
    mammal     1.000     1.000     0.037     0.066     0.000
       dog     1.000     1.000     1.000     0.000     0.000
       cat     1.000     1.000     0.000     1.000     0.000
      fish     1.000     0.000     0.000     0.000     1.000

--- Overlap P(row intersects col) ---

              animal    mammal       dog       cat      fish
    animal     1.000     0.729     0.027     0.048     0.005
    mammal     0.729     1.000     0.037     0.066     0.000
       dog     0.027     0.037     1.000     0.000     0.000
       cat     0.048     0.066     0.000     1.000     0.000
      fish     0.005     0.000     0.000     0.000     1.000

--- Volume ratios (generality) ---

  animal / mammal = 1.4x
  mammal / dog    = 27.0x
  animal / dog    = 37.0x

--- Key relationships (computed) ---

  P(dog in mammal) = 1.0000, P(mammal in animal) = 1.0000, P(dog in animal) = 1.0000
  -> Transitive containment: dog inside mammal inside animal

  overlap(dog, cat) = 0.0000, overlap(cat, dog) = 0.0000
  -> Low overlap: dog and cat occupy different regions

  P(fish in mammal) = 0.0000, P(fish in animal) = 1.0000
  -> Fish is inside animal but outside mammal (different sub-hierarchy)

  volume(animal) = 1.0000, volume(dog) = 0.0270, ratio = 37.0x
  -> More general concepts have larger volume (animal is 37x bigger than dog)
```

</details>

<details>
<summary><b>gumbel_box_exploration</b> -- Gumbel temperature and soft membership</summary>

```text
=== Gumbel Box Embeddings: Probabilistic Containment ===

--- Temperature effect on membership probability ---

  At low temperature  (t=0.01, hard boxes):  P(center in animal) = 1.0000
  At high temperature (t=1.00, soft boxes):  P(center in animal) = 0.0582
  (Low temp recovers the intuitive ~1.0; high temp spreads probability mass.)

--- Part 1: Gumbel boxes created ---

    animal: dim=3, temp=1.00, volume=0.2371
    mammal: dim=3, temp=1.00, volume=0.1891
       dog: dim=3, temp=1.00, volume=0.0446
   vehicle: dim=3, temp=1.00, volume=0.2371

--- Part 2: Soft containment probabilities ---

  P(dog inside animal)  = 0.0549   (dog IS-A animal; <1 due to temp=1.0 softening)
  P(dog inside mammal)  = 0.0488   (dog IS-A mammal; <1 due to temp=1.0 softening)
  P(mammal inside animal) = 0.0337 (mammal IS-A animal; <1 due to temp=1.0 softening)
  P(animal inside dog)  = 0.0103   (should be < P(dog|animal): animal is NOT a dog)
  P(dog inside vehicle) = 0.0028   (should be ~0.0: dog is NOT a vehicle)

--- Part 3: Temperature effect on containment ---

 temperature        P(dog|animal)       P(dog|vehicle)
------------------------------------------------------
        0.01             0.999995             0.000000
        0.10             0.650159             0.000000
        0.50             0.109745             0.000021
        1.00             0.054885             0.002848
        2.00             0.035328             0.015144
        5.00             0.026238             0.022753
       10.00             0.023628             0.022793
      100.00             0.021449             0.021441

--- Part 4: Point membership probability ---

                         point    P(in animal)       P(in dog)
--------------------------------------------------------------
          center (0.5,0.5,0.5)        0.058166        0.023001
      inside dog (0.2,0.2,0.2)        0.054600        0.024050
        boundary (0.0,0.0,0.0)        0.048839        0.023001
         outside (5.0,5.0,5.0)        0.000006        0.000001

--- Part 5: Temperature sharpens membership ---

 temperature  P(boundary_pt in animal)
---------------------------------------
        0.01                  0.125000
        0.10                  0.124983
        0.50                  0.085416
        1.00                  0.048839
        5.00                  0.020778
      100.00                  0.018086

Key takeaways:
  - Gumbel boxes compute soft containment: gradients flow even for disjoint boxes
  - Low temperature  -> sharp boundaries (good for final predictions)
  - High temperature -> smooth gradients (good for early training)
  - Temperature annealing: start high, decrease during training
  - Reference: Dasgupta et al. (2020), NeurIPS 2020
```

</details>

<details>
<summary><b>octagon_demo</b> -- octagon embeddings with diagonal constraints</summary>

```text
=== Octagon Embeddings Demo ===

Part 1: Octagon vs Box

  Bounding box [0,4]x[0,4] volume: 16.00
  Octagon volume (with diagonal cuts): 8.00
  Ratio: 50.0% -- the diagonal constraints remove 50.0% of the box area

Part 2: Point containment

  (2.0, 2.0) center: INSIDE
  (0.1, 0.1) corner (0.1, 0.1) -- inside box, outside octagon: outside
  (1.0, 1.5) off-center (1.0, 1.5): INSIDE
  (3.5, 3.5) near corner (3.5, 3.5) -- x+y=7 > 6: outside
  (0.0, 2.0) edge (0.0, 2.0) -- x+y=2, x-y=-2: INSIDE

Part 3: Intersection (closure property)

  Octagon A: axis [0,3]x[0,3], volume=7.00
  Octagon B: axis [1,4]x[1,4], volume=4.00
  Intersection: axis [1,3]x[1,3], volume=2.00
  Volume ratio: 28.6% of A, 50.0% of B

Part 4: Soft containment and overlap probabilities

  P(wide contains narrow) = 0.5597  (should be high)
  P(narrow contains wide) = 0.0000  (should be low)
  P(overlap)              = 0.9998  (should be high)

Part 5: Bounding box (drop diagonal constraints)

  Original octagon: axis [0,4]x[0,4] with diagonal cuts
  Bounding box:     [0,4]x[0,4] (outer approximation)

--- Summary ---

  Octagons are strictly more expressive than boxes:
  - Any box is an octagon with vacuous diagonal constraints
  - Diagonal cuts remove unreachable corners, improving fit
  - Closed under intersection: composing relations stays in the octagon domain
  - O(d) storage: 2d axis bounds + 4(d-1) diagonal bounds
```

</details>

<details>
<summary><b>fuzzy_query</b> -- fuzzy logic operators for query answering</summary>

```text
=== Fuzzy Query Answering with T-norms ===

Query 1: aquatic AND mammal  (t-norm = intersection)

      entity |  mammal aquatic |     Min Product  Lukasz
  -----------+-----------------+------------------------
     dolphin |   0.950   0.900 |   0.900   0.855   0.850
       whale |   0.980   0.920 |   0.920   0.902   0.900
       shark |   0.050   0.990 |   0.050   0.049   0.040
      salmon |   0.020   0.950 |   0.020   0.019   0.000
       eagle |   0.010   0.050 |   0.010   0.001   0.000
       tiger |   0.970   0.100 |   0.100   0.097   0.070
       panda |   0.990   0.020 |   0.020   0.020   0.010

Query 2: aquatic OR endangered  (t-conorm = union)

      entity | aquatic  endgrd |     Max    Prob  Lukasz
  -----------+-----------------+------------------------
     dolphin |   0.900   0.600 |   0.900   0.960   1.000
       whale |   0.920   0.850 |   0.920   0.988   1.000
       shark |   0.990   0.700 |   0.990   0.997   1.000
      salmon |   0.950   0.300 |   0.950   0.965   1.000
       eagle |   0.050   0.750 |   0.750   0.762   0.800
       tiger |   0.100   0.900 |   0.900   0.910   1.000
       panda |   0.020   0.950 |   0.950   0.951   0.970

Query 3: NOT mammal AND aquatic  (negation + t-norm)

      entity |  mammal   NOT_m aquatic |  result
  -----------+-------------------------+--------
     dolphin |   0.950   0.050   0.900 |   0.045
       whale |   0.980   0.020   0.920 |   0.018
       shark |   0.050   0.950   0.990 |   0.941
      salmon |   0.020   0.980   0.950 |   0.931
       eagle |   0.010   0.990   0.050 |   0.049
       tiger |   0.970   0.030   0.100 |   0.003
       panda |   0.990   0.010   0.020 |   0.000

De Morgan: neg(T(a,b)) = S(neg(a), neg(b))

  Min: neg(T(0.7,0.4)) = 0.600000,  S(neg(0.7),neg(0.4)) = 0.600000,  match=true
  Product: neg(T(0.7,0.4)) = 0.720000,  S(neg(0.7),neg(0.4)) = 0.720000,  match=true
  Lukasiewicz: neg(T(0.7,0.4)) = 0.900000,  S(neg(0.7),neg(0.4)) = 0.900000,  match=true

Ranking: top-3 aquatic mammals by each t-norm

  Min: whale(0.920), dolphin(0.900), tiger(0.100)
  Product: whale(0.902), dolphin(0.855), tiger(0.097)
  Lukasiewicz: whale(0.900), dolphin(0.850), tiger(0.070)

--- Summary ---

  Min t-norm: conservative, score = weakest link
  Product t-norm: balanced, penalizes low scores multiplicatively
  Lukasiewicz t-norm: strictest, requires both inputs high (additive threshold)
  All three agree on top entity (dolphin/whale) but differ on cutoff sharpness
```

</details>

<details>
<summary><b>query2box</b> -- compositional multi-hop query answering</summary>

```text
=== Query2Box: Compositional Query Answering ===

Q1: What cities are in France?

  Rank by P(France contains city):
    1:    Paris score=1.0000 <-- answer
    2:     Lyon score=1.0000 <-- answer
    3:   London score=0.0000

Q2: What languages are spoken in France?

  Rank by P(France contains language):
    1:   French score=1.0000 <-- answer
    2:  English score=0.0000

Q3: Languages spoken in countries with French cities (2-hop)

  Hop 1: intersect France with city-containing region
    intersection volume: 0.1409  (> 0 confirms France has cities)
  Hop 2: rank languages by containment in hop-1 result

  Rank by P(hop1_box contains language):
    1:   French score=1.0000 <-- answer
    2:  English score=0.0000

Q4: Alpha-weighted distance scoring (Ren et al., 2020)

  alpha = 0.02  (inside penalty << outside penalty)

  Q1 re-scored: cities in France (by distance, ascending)

    1:     Lyon dist=0.0188 <-- answer
    2:    Paris dist=0.0424 <-- answer
    3:   London dist=8.0370

  Q2 re-scored: languages in France (by distance, ascending)

    1:   French dist=0.0212
    2:  English dist=8.9310

  Alpha sensitivity: distance for Paris across alpha values

    alpha=0.00: dist=0.0000
    alpha=0.02: dist=0.0424
    alpha=0.10: dist=0.2120
    alpha=0.50: dist=1.0600
    alpha=1.00: dist=2.1200

--- Summary ---

  Q1 correctly ranks Paris and Lyon above London.
  Q2 ranks French highest (fully inside France).
  Q3 chains two hops: city containment, then language containment.
  Intersection volume decreases at each hop, narrowing the answer set.
  Q4 shows Query2Box distance scoring: lower distance = better answer.
  Alpha controls inside-vs-outside penalty balance.
```

</details>

<details>
<summary><b>taxobell_demo</b> -- Gaussian box losses (no training)</summary>

```text
=== TaxoBell: Gaussian Box Embeddings for Taxonomy Expansion ===

--- Part 1: Taxonomy nodes (dim=8) ---

   concept   log-volume     sigma[0]
------------------------------------
    entity      11.0904       4.0000
    animal       5.5452       2.0000
   vehicle       5.5452       2.0000
       dog      -5.5452       0.5000
       cat      -5.5452       0.5000
       car      -5.5452       0.5000
     truck      -5.5452       0.5000

--- Part 2: KL divergence (child -> parent containment) ---

     child     parent         D_KL  interpretation
-------------------------------------------------------
    animal     entity       2.7952  partial containment
   vehicle     entity       4.7952  partial containment
       dog     animal       7.5904  poor containment
       cat     animal       7.5904  poor containment
       car    vehicle       7.5904  poor containment
     truck    vehicle       7.5904  poor containment

  Asymmetry: D_KL(dog||animal) = 7.5904, D_KL(animal||dog) = 52.9096
  dog fits in animal (small KL), but animal does NOT fit in dog (large KL)
  Cross-domain: D_KL(dog||vehicle) = 27.5904 (no containment)

--- Part 3: Bhattacharyya coefficient (symmetric similarity) ---

 concept_a  concept_b         BC  relationship
--------------------------------------------------
       dog        cat   0.018316  some overlap
       car      truck   0.018316  some overlap
       dog        car   0.000000  near-disjoint
    animal    vehicle   0.018316  some overlap

--- Part 4: TaxoBell combined loss ---

  L_sym  (symmetric triplet)  = 4.014859
  L_asym (KL containment)     = 5.868443
  L_reg  (volume regulation)  = 0.000000
  L_clip (sigma clipping)     = 0.857143

  Total = 1.00*4.0149 + 1.00*5.8684 + 0.01*0.0000 + 0.01*0.8571
        = 9.891873

Key takeaways:
  - KL divergence is asymmetric: it measures directed containment (child in parent)
  - Bhattacharyya coefficient is symmetric: it measures distributional overlap
  - TaxoBell combines both with regularization for end-to-end taxonomy training
  - Wider sigmas = more general concepts; narrower sigmas = more specific
  - Reference: TaxoBell (WWW 2026, arXiv:2601.09633)
```

</details>

<details>
<summary><b>hyperbolic_demo</b> -- Poincare ball embeddings</summary>

```text
=== Poincare Ball Embeddings: Biological Taxonomy ===

--- Norm by depth (general near origin, specific near boundary) ---

        entity  depth     norm  conformal
          Life      0   0.0000     2.0000
      Animalia      1   0.3000     2.1978
       Plantae      1   0.3000     2.1978
         Fungi      1   0.3000     2.1978
     Carnivora      2   0.6000     3.1250
      Primates      2   0.6000     3.1250
       Rosales      2   0.6000     3.1250
        Poales      2   0.6000     3.1250
    Agaricales      2   0.6000     3.1250
          Wolf      3   0.8800     8.8652
          Lion      3   0.8800     8.8652
         Human      3   0.8800     8.8652
         Chimp      3   0.8800     8.8652
          Rose      3   0.8800     8.8652
         Wheat      3   0.8800     8.8652
       Amanita      3   0.8800     8.8652

  Conformal factor = 2/(1 - ||x||^2). Grows sharply near the boundary,
  meaning a small coordinate shift near the boundary covers much more
  hyperbolic distance than the same shift near the origin.

--- Selected pairwise hyperbolic distances ---

  Life -> Animalia (depth 0->1)              d = 0.6190
  Animalia -> Carnivora (depth 1->2)         d = 0.7835
  Carnivora -> Wolf (depth 2->3)             d = 1.3846
  Wolf <-> Lion (siblings)                   d = 1.0433
  Wolf <-> Human (cross-order)               d = 1.9878
  Wolf <-> Rose (cross-kingdom)              d = 5.3872
  Human <-> Chimp (siblings)                 d = 1.0433
  Life -> Amanita (root to leaf)             d = 2.7515

--- Hierarchy preservation: 100.0% (15/15 parent-child pairs correct) ---

--- Exponential capacity: distance from origin by depth ---

 depth           norm    d(origin,p)          ratio
     0         0.0000         0.0000              -
     1         0.3000         0.6190              -
     2         0.6000         1.3863          2.24x
     3         0.8800         2.7515          1.98x

  Distance-to-origin grows faster than linearly with norm because the
  hyperbolic metric inflates near the boundary: d = 2 * arctanh(||x||).
  This is the exponential capacity that lets trees embed without distortion.

Key observations:
  - All parent-child norm orderings are satisfied (hierarchy preserved)
  - Cross-kingdom distances (Wolf<->Rose) exceed within-order distances (Wolf<->Lion)
  - Hyperbolic distance from origin accelerates with depth, encoding exponential
    branching capacity in bounded coordinates
```

</details>

<details>
<summary><b>cone_training</b> -- cone embeddings on an 18-entity taxonomy</summary>

```text
=== Cone Embeddings (ConE): Training on a Taxonomy (18 entities, 4 levels) ===

Training for 500 epochs (50 warmup + 450 joint, dim=16, 18 entities, 17 pos + 29 neg pairs)...

  Epoch    0 [warmup]: avg_loss = 0.0192
  Epoch  125 [joint]: avg_loss = 0.0678
  Epoch  250 [joint]: avg_loss = 0.0639
  Epoch  375 [joint]: avg_loss = 0.0648
  Epoch  499 [joint]: avg_loss = 0.0649

--- Learned Cone Properties ---

        entity    mean_aper     mean_deg
----------------------------------------
        entity       3.0610        175.4
        animal       2.5595        146.6
       vehicle       2.5235        144.6
        mammal       2.1850        125.2
          bird       1.5128         86.7
          fish       1.2583         72.1
  land_vehicle       1.7302         99.1
      aircraft       0.8409         48.2
           dog       0.0777          4.4
           cat       0.1512          8.7
         whale       0.1380          7.9
         eagle       0.0846          4.8
       sparrow       0.2093         12.0
        salmon       0.0767          4.4
          tuna       0.1543          8.8
           car       0.0665          3.8
         truck       0.2899         16.6
    helicopter       0.2095         12.0

--- Selected Containment Distances (lower = better containment) ---

  [POS] entity > animal                dist = 0.1324
  [POS] entity > vehicle               dist = 0.2503
  [POS] animal > mammal                dist = 0.1163
  [POS] animal > bird                  dist = 0.2217
  [POS] mammal > dog                   dist = 0.1396
  [POS] mammal > cat                   dist = 0.2195
  [POS] bird > eagle                   dist = 0.1114
  [POS] fish > salmon                  dist = 0.1318
  [POS] land_vehicle > car             dist = 0.1142
  [POS] aircraft > helicopter          dist = 0.0568
  [NEG] dog > entity (reverse)         dist = 14.3115
  [NEG] dog > cat (sibling)            dist = 13.9929
  [NEG] animal > vehicle (cross)       dist = 0.4439
  [NEG] mammal > land_vehicle (cross)  dist = 1.4487

Avg positive distance: 0.1494, Avg negative distance: 7.5492
Positive pairs have lower distance than negatives (as expected).

Key takeaways:
  - More general concepts (entity, animal) get wider mean apertures
  - More specific concepts (dog, car) get narrower mean apertures
  - Containment is directional: animal > mammal, but NOT mammal > animal
  - Cross-branch distance (animal > vehicle) stays high
  - Unlike boxes, cones support negation: complement of a cone is a cone
```

</details>

<details>
<summary><b>box_training</b> -- direct coordinate training on 25-entity taxonomy</summary>

```text
=== Box Embedding Training (25 entities, direct coordinate updates) ===

Entities: 25
Containment pairs: 24

Training for 300 epochs (dim=8, lr=0.05, neg_lr=0.04)...

  Epoch    0: total_violation = 34.9070
  Epoch   50: total_violation = 3.8661
  Epoch  100: total_violation = 1.7840
  Epoch  150: total_violation = 1.4926
  Epoch  200: total_violation = 1.3762
  Epoch  250: total_violation = 1.2435
  Epoch  299: total_violation = 1.1112

--- Learned Box Volumes (larger = more general) ---

        entity: volume = 2.401828e4
       vehicle: volume = 6.491940e2
           car: volume = 5.161266e1
         truck: volume = 5.140411e1
       bicycle: volume = 5.140404e1
        animal: volume = 2.756642e1
         plant: volume = 6.517150e0
        mammal: volume = 5.015008e-1
          bird: volume = 3.474687e-1
          tree: volume = 2.028825e-1
        flower: volume = 2.028822e-1
          fish: volume = 1.978809e-1
        salmon: volume = 1.235360e-3
          tuna: volume = 1.235357e-3
           oak: volume = 1.235354e-3
          pine: volume = 1.235352e-3
          rose: volume = 1.235350e-3
         tulip: volume = 1.235348e-3
         eagle: volume = 1.179650e-3
       penguin: volume = 1.161592e-3
       sparrow: volume = 1.161585e-3
           bat: volume = 1.108423e-3
           cat: volume = 1.091402e-3
           dog: volume = 1.056696e-3
         whale: volume = 1.038240e-3

--- Containment Checks ---

  [  OK] entity > animal                P = 1.000
  [  OK] entity > vehicle               P = 1.000
  [  OK] animal > mammal                P = 1.000
  [  OK] animal > bird                  P = 1.000
  [  OK] mammal > dog                   P = 1.000
  [  OK] mammal > cat                   P = 1.000
  [  OK] bird > eagle                   P = 1.000
  [  OK] fish > salmon                  P = 1.000
  [  OK] plant > tree                   P = 1.000
  [  OK] tree > oak                     P = 1.000
  [  OK] flower > rose                  P = 1.000
  [  OK] vehicle > car                  P = 0.875
  [  OK] dog > animal (reverse)         P = 0.000
  [  OK] cat > dog (sibling)            P = 0.000
  [  OK] animal > vehicle (cross)       P = 0.000

Hierarchy accuracy: 15/15 (100%)

Notes:
  - This uses direct coordinate updates, not backpropagation
  - Negative separation pushes sibling/cross-branch boxes apart
  - Leaf shrinkage produces varied volumes (more specific = smaller)
  - Volume ordering (general > specific) emerges from containment constraints
```

</details>

<details>
<summary><b>dataset_training</b> -- WN18RR-format pipeline with MRR/Hits@k</summary>

```text
=== Dataset-Driven Box Embedding Training ===

Dataset: 47 entities, 1 relations, 46 train / 7 valid / 7 test triples

Training 200 epochs (dim=12, 46 containment pairs, 74 negative pairs)...

  epoch   0: total_violation = 108.1936
  epoch  40: total_violation = 32.6241
  epoch  80: total_violation = 23.4292
  epoch 120: total_violation = 16.3861
  epoch 160: total_violation = 10.2297
  epoch 199: total_violation = 6.4860

--- Evaluation (test set) ---

  fox.n.01 -> carnivore.n.01: rank 8 / 47
  lion.n.01 -> carnivore.n.01: rank 2 / 47
  tiger.n.01 -> carnivore.n.01: rank 8 / 47
  trout.n.01 -> vertebrate.n.01: rank 7 / 47
  pine.n.01 -> plant.n.02: rank 4 / 47
  tulip.n.01 -> plant.n.02: rank 4 / 47
  truck.n.01 -> artifact.n.01: rank 1 / 47

--- Link Prediction Metrics ---

  MRR:       0.3418
  Hits@1:    0.1429
  Hits@3:    0.2857
  Hits@10:   1.0000
  Mean Rank: 4.9

  (7 test triples, 47 candidate entities)
```

</details>

<details>
<summary><b>imagenet_hierarchy</b> -- 252 Tiny ImageNet synsets, volume-depth correlation</summary>

```text
=== Tiny ImageNet Hierarchy: Box Embedding Training ===

Hierarchy: 252 entities (114 leaves, 138 internal), depth 12, 251 edges

Training 400 epochs (dim=16, lr=0.06, 251 containment, 470 negative pairs)...

... (training output truncated, showing final results)

  epoch 399: total_violation = 17.5900

--- Evaluation (held-out transitive hypernym triples) ---

  MRR:       0.3291
  Hits@1:    0.0938
  Hits@3:    0.4688
  Hits@10:   1.0000
  Mean Rank: 4.5
  (32 test triples, 252 candidate entities)

--- Volume-Depth Correlation ---

  depth   count   avg_log_vol
  -----  ------  ------------
      0       1         29.38
      1       5         14.66
      2      15          7.15
      3      40          1.39
      4      55         -3.80
      5      45        -10.47
      6      28        -18.99
      7      19        -26.28
      8      13        -28.90
      9      14        -47.49
     10       6        -31.32
     11       9        -72.14
     12       2        -64.82

  Spearman(depth, log_volume) = -0.9555  (n=252)
  -> Negative correlation confirms: specific concepts have smaller boxes.

--- Volume Extremes ---

  Largest boxes (most general):
    entity                       depth=0 log_vol=29.38
    organism                     depth=1 log_vol=23.69
    animal                       depth=2 log_vol=19.29
    artifact                     depth=1 log_vol=17.05
    chordate                     depth=3 log_vol=13.91
  Smallest boxes (most specific):
    cougar                       depth=11 log_vol=-86.14
    lion                         depth=11 log_vol=-86.08
    tabby_cat                    depth=11 log_vol=-82.38
    Persian_cat                  depth=11 log_vol=-81.98
    Egyptian_cat                 depth=11 log_vol=-81.85

  252 total entities trained
```

</details>

<details>
<summary><b>el_training</b> -- EL++ ontology embedding training</summary>

```text
=== EL++ Ontology Embedding Training ===

Ontology: 15 concepts, 4 roles, 21 axioms

Training with dim=30, epochs=500...

epoch 50/500: avg_loss = 0.054586, lr = 0.004960
epoch 100/500: avg_loss = 0.003529, lr = 0.004706
epoch 150/500: avg_loss = 0.000357, lr = 0.004245
epoch 200/500: avg_loss = 0.000117, lr = 0.003625
epoch 250/500: avg_loss = 0.000000, lr = 0.002912
epoch 300/500: avg_loss = 0.000000, lr = 0.002182
epoch 350/500: avg_loss = 0.000000, lr = 0.001512
epoch 400/500: avg_loss = 0.000000, lr = 0.000974
epoch 450/500: avg_loss = 0.000000, lr = 0.000624
epoch 500/500: avg_loss = 0.000000, lr = 0.000500

Loss: 4.5574 (epoch 1) -> 0.0000 (epoch 500)

Subsumption prediction (on training axioms):
  Hits@1:  0.85
  Hits@10: 1.00
  MRR:     0.9231

--- Spot checks (lower = better containment) ---
  Dog ⊑ Mammal: 0.0637  (SHOULD be low)
  Dog ⊑ Animal: 0.0847  (SHOULD be low)
  Dog ⊑ LivingThing: 0.1179  (SHOULD be low)
  Dog ⊑ Cat: 2.4659  (SHOULD be high)
  Mammal ⊑ Fish: 4.6036  (SHOULD be high)
  Animal ⊑ Plant: 6.8601  (SHOULD be high)
  Eagle ⊑ Bird: 0.0840  (SHOULD be low)
  Salmon ⊑ Fish: 0.1320  (SHOULD be low)
```

</details>

## Visualization scripts

The `scripts/` directory contains Python plotting scripts (PEP 723, run with `uv run`):

- `scripts/plot_box_concept.py` -- generates `docs/box_concepts.png` (containment, Gumbel soft boundary, octagon vs box)
- `scripts/plot_training.py` -- generates `docs/training_convergence.png` (loss curve + containment probability convergence)
- `scripts/plot_gumbel_robustness.py` -- generates `docs/gumbel_robustness.png` (gradient landscape: Gumbel membership + gradient magnitude at box boundary)
- `scripts/plot_temperature_sensitivity.py` -- generates `docs/temperature_sensitivity.png` (containment probability vs Gumbel temperature)
- `scripts/plot_fuzzy_tnorms.py` -- generates `docs/fuzzy_tnorms.png` (t-norm contour plots and 1D slices)
