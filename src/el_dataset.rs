//! EL++ normalized axiom dataset loader.
//!
//! Loads ontologies in a simple text format for use with the EL++ embedding
//! primitives in [`crate::el`].
//!
//! # File Format
//!
//! One axiom per line, tab-separated, type-tagged:
//!
//! ```text
//! NF1  HeartDisease  Genetic  InheritedHeartDisease
//! NF2  Dog  Animal
//! NF3  Heart  partOf  Body
//! NF4  hasParent  Human  Human
//! RI6  hasChild  hasDescendant
//! RI7  hasParent  hasSibling  hasUncle
//! DISJ  Cat  Dog
//! ```
//!
//! (Fields are tab-separated in the actual file.)
//!
//! | Tag | Fields | Meaning |
//! |-----|--------|---------|
//! | `NF1` | C1, C2, D | C1 ⊓ C2 ⊑ D (conjunction subsumption) |
//! | `NF2` | C, D | C ⊑ D (atomic subsumption) |
//! | `NF3` | C, r, D | C ⊑ ∃r.D (existential restriction) |
//! | `NF4` | r, C, D | ∃r.C ⊑ D (inverse existential) |
//! | `RI6` | r, s | r ⊑ s (role inclusion) |
//! | `RI7` | r, s, t | r ∘ s ⊑ t (role chain) |
//! | `DISJ` | C, D | C ⊓ D ⊑ ⊥ (disjointness) |
//!
//! Lines starting with `#` are comments. Empty lines are skipped.
//!
//! # Preprocessing
//!
//! OWL ontologies must be normalized to this format externally. Tools:
//! - [mOWL](https://mowl.readthedocs.io/) (Python + Java, supports EL++ normalization)
//! - [jcel](https://github.com/julianmendez/jcel) (Java EL++ reasoner with normalization)
//!
//! # Example
//!
//! ```rust,no_run
//! use subsume::el_dataset::{load_el_axioms, Axiom};
//!
//! let axioms = load_el_axioms("ontology/go_normalized.tsv").unwrap();
//! for ax in &axioms.nf2 {
//!     println!("{} ⊑ {}", ax.0, ax.1);
//! }
//! println!("Classes: {}, Roles: {}", axioms.classes().len(), axioms.roles().len());
//! ```

use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use crate::BoxError;

/// A parsed EL++ axiom.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Axiom {
    /// NF1: C1 ⊓ C2 ⊑ D (conjunction subsumption).
    Intersection(String, String, String),
    /// NF2: C ⊑ D (atomic concept subsumption).
    Subsumption(String, String),
    /// NF3: C ⊑ ∃r.D (existential restriction).
    Existential(String, String, String),
    /// NF4: ∃r.C ⊑ D (inverse existential).
    InverseExistential(String, String, String),
    /// RI6: r ⊑ s (role inclusion).
    RoleInclusion(String, String),
    /// RI7: r ∘ s ⊑ t (role chain).
    RoleChain(String, String, String),
    /// C ⊓ D ⊑ ⊥ (disjointness).
    Disjointness(String, String),
}

/// A collection of EL++ normalized axioms, grouped by normal form.
#[derive(Debug, Clone, Default)]
pub struct ElDataset {
    /// NF1: (C1, C2, D) -- C1 ⊓ C2 ⊑ D
    pub nf1: Vec<(String, String, String)>,
    /// NF2: (C, D) -- C ⊑ D
    pub nf2: Vec<(String, String)>,
    /// NF3: (C, r, D) -- C ⊑ ∃r.D
    pub nf3: Vec<(String, String, String)>,
    /// NF4: (r, C, D) -- ∃r.C ⊑ D
    pub nf4: Vec<(String, String, String)>,
    /// RI6: (r, s) -- r ⊑ s
    pub ri6: Vec<(String, String)>,
    /// RI7: (r, s, t) -- r ∘ s ⊑ t
    pub ri7: Vec<(String, String, String)>,
    /// Disjointness: (C, D) -- C ⊓ D ⊑ ⊥
    pub disj: Vec<(String, String)>,
}

impl ElDataset {
    /// Total number of axioms across all normal forms.
    pub fn len(&self) -> usize {
        self.nf1.len()
            + self.nf2.len()
            + self.nf3.len()
            + self.nf4.len()
            + self.ri6.len()
            + self.ri7.len()
            + self.disj.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Collect all unique class/concept names.
    pub fn classes(&self) -> HashSet<&str> {
        let mut s = HashSet::new();
        for (a, b, c) in &self.nf1 {
            s.insert(a.as_str());
            s.insert(b.as_str());
            s.insert(c.as_str());
        }
        for (a, b) in &self.nf2 {
            s.insert(a.as_str());
            s.insert(b.as_str());
        }
        for (c, _, d) in &self.nf3 {
            s.insert(c.as_str());
            s.insert(d.as_str());
        }
        for (_, c, d) in &self.nf4 {
            s.insert(c.as_str());
            s.insert(d.as_str());
        }
        for (a, b) in &self.disj {
            s.insert(a.as_str());
            s.insert(b.as_str());
        }
        s
    }

    /// Collect all unique role/relation names.
    pub fn roles(&self) -> HashSet<&str> {
        let mut s = HashSet::new();
        for (_, r, _) in &self.nf3 {
            s.insert(r.as_str());
        }
        for (r, _, _) in &self.nf4 {
            s.insert(r.as_str());
        }
        for (r, s_) in &self.ri6 {
            s.insert(r.as_str());
            s.insert(s_.as_str());
        }
        for (r, s_, t) in &self.ri7 {
            s.insert(r.as_str());
            s.insert(s_.as_str());
            s.insert(t.as_str());
        }
        s
    }

    /// All axioms as a flat iterator.
    pub fn iter(&self) -> impl Iterator<Item = Axiom> + '_ {
        self.nf1
            .iter()
            .map(|(a, b, c)| Axiom::Intersection(a.clone(), b.clone(), c.clone()))
            .chain(
                self.nf2
                    .iter()
                    .map(|(a, b)| Axiom::Subsumption(a.clone(), b.clone())),
            )
            .chain(
                self.nf3
                    .iter()
                    .map(|(c, r, d)| Axiom::Existential(c.clone(), r.clone(), d.clone())),
            )
            .chain(
                self.nf4
                    .iter()
                    .map(|(r, c, d)| Axiom::InverseExistential(r.clone(), c.clone(), d.clone())),
            )
            .chain(
                self.ri6
                    .iter()
                    .map(|(r, s)| Axiom::RoleInclusion(r.clone(), s.clone())),
            )
            .chain(
                self.ri7
                    .iter()
                    .map(|(r, s, t)| Axiom::RoleChain(r.clone(), s.clone(), t.clone())),
            )
            .chain(
                self.disj
                    .iter()
                    .map(|(a, b)| Axiom::Disjointness(a.clone(), b.clone())),
            )
    }
}

/// Load EL++ normalized axioms from a text file.
///
/// See [module docs](self) for the file format.
pub fn load_el_axioms<P: AsRef<Path>>(path: P) -> Result<ElDataset, BoxError> {
    let content =
        std::fs::read_to_string(path.as_ref()).map_err(|e| BoxError::Io(format!("{e}")))?;
    parse_el_axioms(&content)
}

/// Parse EL++ axioms from a string (same format as file loader).
pub fn parse_el_axioms(text: &str) -> Result<ElDataset, BoxError> {
    let mut dataset = ElDataset::default();

    for (line_num, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "NF1" if parts.len() == 4 => {
                dataset
                    .nf1
                    .push((parts[1].into(), parts[2].into(), parts[3].into()));
            }
            "NF2" if parts.len() == 3 => {
                dataset.nf2.push((parts[1].into(), parts[2].into()));
            }
            "NF3" if parts.len() == 4 => {
                dataset
                    .nf3
                    .push((parts[1].into(), parts[2].into(), parts[3].into()));
            }
            "NF4" if parts.len() == 4 => {
                dataset
                    .nf4
                    .push((parts[1].into(), parts[2].into(), parts[3].into()));
            }
            "RI6" if parts.len() == 3 => {
                dataset.ri6.push((parts[1].into(), parts[2].into()));
            }
            "RI7" if parts.len() == 4 => {
                dataset
                    .ri7
                    .push((parts[1].into(), parts[2].into(), parts[3].into()));
            }
            "DISJ" if parts.len() == 3 => {
                dataset.disj.push((parts[1].into(), parts[2].into()));
            }
            _ => {
                return Err(BoxError::Internal(format!(
                    "line {}: invalid axiom '{}'",
                    line_num + 1,
                    line
                )));
            }
        }
    }

    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_all_forms() {
        let text = "\
# Gene Ontology subset (normalized)
NF1\tHeartDisease\tGenetic\tInheritedHeartDisease
NF2\tDog\tAnimal
NF2\tCat\tAnimal
NF3\tHeart\tpartOf\tBody
NF4\thasParent\tHuman\tHuman
RI6\thasChild\thasDescendant
RI7\thasParent\thasSibling\thasUncle
DISJ\tCat\tDog
";
        let ds = parse_el_axioms(text).unwrap();
        assert_eq!(ds.nf1.len(), 1);
        assert_eq!(ds.nf2.len(), 2);
        assert_eq!(ds.nf3.len(), 1);
        assert_eq!(ds.nf4.len(), 1);
        assert_eq!(ds.ri6.len(), 1);
        assert_eq!(ds.ri7.len(), 1);
        assert_eq!(ds.disj.len(), 1);
        assert_eq!(ds.len(), 8);

        let classes = ds.classes();
        assert!(classes.contains("Dog"));
        assert!(classes.contains("Animal"));
        assert!(classes.contains("HeartDisease"));
        assert!(!classes.contains("partOf")); // role, not class

        let roles = ds.roles();
        assert!(roles.contains("partOf"));
        assert!(roles.contains("hasParent"));
        assert!(!roles.contains("Dog")); // class, not role
    }

    #[test]
    fn test_empty_and_comments() {
        let text = "# comment\n\n# another comment\n";
        let ds = parse_el_axioms(text).unwrap();
        assert!(ds.is_empty());
    }

    #[test]
    fn test_invalid_tag() {
        let text = "INVALID\tFoo\tBar";
        assert!(parse_el_axioms(text).is_err());
    }

    #[test]
    fn test_wrong_field_count() {
        let text = "NF2\tDog";
        assert!(parse_el_axioms(text).is_err());
    }

    #[test]
    fn test_iter_roundtrip() {
        let text = "NF2\tA\tB\nNF3\tC\tr\tD\nDISJ\tE\tF\n";
        let ds = parse_el_axioms(text).unwrap();
        let axioms: Vec<Axiom> = ds.iter().collect();
        assert_eq!(axioms.len(), 3);
        assert!(matches!(&axioms[0], Axiom::Subsumption(a, b) if a == "A" && b == "B"));
        assert!(
            matches!(&axioms[1], Axiom::Existential(c, r, d) if c == "C" && r == "r" && d == "D")
        );
        assert!(matches!(&axioms[2], Axiom::Disjointness(e, f) if e == "E" && f == "F"));
    }
}
