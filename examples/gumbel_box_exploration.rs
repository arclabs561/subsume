//! Gumbel box embeddings: exploring probabilistic containment and temperature effects.
//!
//! Gumbel boxes (Dasgupta et al., 2020) solve the **local identifiability problem**
//! that prevents learning with hard boxes. By modeling box coordinates as Gumbel
//! random variables, every parameter contributes to the loss -- no flat regions,
//! no vanishing gradients.
//!
//! This example explores Gumbel box properties (no training loop):
//! 1. Creating Gumbel boxes (NdarrayGumbelBox)
//! 2. Computing probabilistic containment (soft containment via Gumbel)
//! 3. How temperature controls the softness of containment scores
//! 4. Membership probability for points
//!
//! Reference: Dasgupta et al. (2020), "Improving Local Identifiability in
//! Probabilistic Box Embeddings" (NeurIPS 2020)
//!
//! Run: cargo run -p subsume --example gumbel_box_exploration

use ndarray::array;
use subsume::ndarray_backend::NdarrayGumbelBox;
use subsume::Box as BoxTrait;

fn main() -> Result<(), subsume::BoxError> {
    println!("=== Gumbel Box Embeddings: Probabilistic Containment ===\n");

    // --- Part 1: Create Gumbel boxes for a small taxonomy ---
    //
    // Temperature is the key parameter. It controls "softness":
    //   - Low temperature  (-> 0): sharp, hard-box-like behavior
    //   - High temperature (-> inf): soft, spread-out probability

    let temperature = 1.0;

    let animal = NdarrayGumbelBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], temperature)?;

    let mammal = NdarrayGumbelBox::new(
        array![0.05, 0.05, 0.05],
        array![0.95, 0.95, 0.95],
        temperature,
    )?;

    let dog = NdarrayGumbelBox::new(array![0.1, 0.1, 0.1], array![0.4, 0.4, 0.4], temperature)?;

    let vehicle = NdarrayGumbelBox::new(array![2.0, 2.0, 2.0], array![3.0, 3.0, 3.0], temperature)?;

    // Quick orientation: temperature controls whether containment is intuitive.
    // At low temperature, Gumbel boxes behave like hard boxes.
    println!("--- Temperature effect on membership probability ---\n");
    let center_pt = array![0.5, 0.5, 0.5]; // geometric center of animal
    let low_t = 0.01;
    let high_t = 1.0;
    let animal_low = NdarrayGumbelBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], low_t)?;
    let animal_high = NdarrayGumbelBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], high_t)?;
    println!(
        "  At low temperature  (t={:.2}, hard boxes):  P(center in animal) = {:.4}",
        low_t,
        animal_low.membership_probability(&center_pt)?
    );
    println!(
        "  At high temperature (t={:.2}, soft boxes):  P(center in animal) = {:.4}",
        high_t,
        animal_high.membership_probability(&center_pt)?
    );
    println!("  (Low temp recovers the intuitive ~1.0; high temp spreads probability mass.)\n");

    println!("--- Part 1: Gumbel boxes created ---\n");
    let entities: Vec<(&str, &NdarrayGumbelBox)> = vec![
        ("animal", &animal),
        ("mammal", &mammal),
        ("dog", &dog),
        ("vehicle", &vehicle),
    ];
    for (name, b) in &entities {
        println!(
            "  {:>8}: dim={}, temp={:.2}, volume={:.4}",
            name,
            b.dim(),
            b.temperature(),
            b.volume()?
        );
    }

    // --- Part 2: Probabilistic containment ---
    //
    // P(other inside self) = Vol(self ^ other) / Vol(other)
    //
    // For nested boxes this is ~1.0. For disjoint boxes this is 0.0.
    // Gumbel boxes compute this the same way as hard boxes for the
    // volume ratio, but the key difference is during training: gradients
    // flow even when boxes are disjoint, because membership is soft.

    println!("\n--- Part 2: Soft containment probabilities ---\n");
    // NOTE: At temperature=1.0, Gumbel softening spreads probability mass well
    // beyond the hard box boundaries, so containment probabilities are much lower
    // than 1.0 even for geometrically nested boxes. Use low temperature (e.g. 0.01)
    // to recover hard-box-like scores. The Part 3 table below shows this effect.
    println!(
        "  P(dog inside animal)  = {:.4}   (dog IS-A animal; <1 due to temp=1.0 softening)",
        animal.containment_prob(&dog)?
    );
    println!(
        "  P(dog inside mammal)  = {:.4}   (dog IS-A mammal; <1 due to temp=1.0 softening)",
        mammal.containment_prob(&dog)?
    );
    println!(
        "  P(mammal inside animal) = {:.4} (mammal IS-A animal; <1 due to temp=1.0 softening)",
        animal.containment_prob(&mammal)?
    );
    println!(
        "  P(animal inside dog)  = {:.4}   (should be < P(dog|animal): animal is NOT a dog)",
        dog.containment_prob(&animal)?
    );
    println!(
        "  P(dog inside vehicle) = {:.4}   (should be ~0.0: dog is NOT a vehicle)",
        vehicle.containment_prob(&dog)?
    );

    // --- Part 3: Temperature controls softness ---
    //
    // This is the core insight from Dasgupta et al. (2020).
    //
    // At training start, use high temperature for smooth gradients.
    // Gradually anneal to low temperature for sharp final predictions.
    // This is analogous to simulated annealing or Gumbel-Softmax in
    // categorical distributions (Jang et al., 2017).

    println!("\n--- Part 3: Temperature effect on containment ---\n");
    println!(
        "{:>12} {:>20} {:>20}",
        "temperature", "P(dog|animal)", "P(dog|vehicle)"
    );
    println!("{}", "-".repeat(54));

    for &t in &[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0] {
        // Re-create boxes at this temperature to show membership effects.
        let animal_t = NdarrayGumbelBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], t)?;
        let dog_t = NdarrayGumbelBox::new(array![0.1, 0.1, 0.1], array![0.4, 0.4, 0.4], t)?;
        let vehicle_t = NdarrayGumbelBox::new(array![2.0, 2.0, 2.0], array![3.0, 3.0, 3.0], t)?;

        let p_dog_animal = animal_t.containment_prob(&dog_t)?;
        let p_dog_vehicle = vehicle_t.containment_prob(&dog_t)?;

        println!("{:>12.2} {:>20.6} {:>20.6}", t, p_dog_animal, p_dog_vehicle);
    }

    // --- Part 4: Membership probability for points ---
    //
    // Gumbel boxes can compute P(point in box) using sigmoid-based
    // soft membership. This is differentiable, unlike hard indicator
    // functions, enabling gradient-based learning.

    println!("\n--- Part 4: Point membership probability ---\n");

    let test_points = vec![
        ("center (0.5,0.5,0.5)", array![0.5, 0.5, 0.5]),
        ("inside dog (0.2,0.2,0.2)", array![0.2, 0.2, 0.2]),
        ("boundary (0.0,0.0,0.0)", array![0.0, 0.0, 0.0]),
        ("outside (5.0,5.0,5.0)", array![5.0, 5.0, 5.0]),
    ];

    println!("{:>30} {:>15} {:>15}", "point", "P(in animal)", "P(in dog)");
    println!("{}", "-".repeat(62));

    for (label, pt) in &test_points {
        let p_animal = animal.membership_probability(pt)?;
        let p_dog = dog.membership_probability(pt)?;
        println!("{:>30} {:>15.6} {:>15.6}", label, p_animal, p_dog);
    }

    // --- Part 5: Temperature effect on membership sharpness ---

    println!("\n--- Part 5: Temperature sharpens membership ---\n");
    let boundary_point = array![0.0, 0.0, 0.0]; // on the boundary of animal
    println!("{:>12} {:>25}", "temperature", "P(boundary_pt in animal)");
    println!("{}", "-".repeat(39));

    for &t in &[0.01, 0.1, 0.5, 1.0, 5.0, 100.0] {
        let animal_t = NdarrayGumbelBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], t)?;
        let p = animal_t.membership_probability(&boundary_point)?;
        println!("{:>12.2} {:>25.6}", t, p);
    }

    println!("\nKey takeaways:");
    println!("  - Gumbel boxes compute soft containment: gradients flow even for disjoint boxes");
    println!("  - Low temperature  -> sharp boundaries (good for final predictions)");
    println!("  - High temperature -> smooth gradients (good for early training)");
    println!("  - Temperature annealing: start high, decrease during training");
    println!("  - Reference: Dasgupta et al. (2020), NeurIPS 2020");

    // See scripts/plot_gumbel_robustness.py for a visualization of Gumbel noise robustness
    // compared to Gaussian boxes.

    Ok(())
}
