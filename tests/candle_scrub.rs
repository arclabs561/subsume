//! Post-migration guard: the candle backend was removed in 0.14.0, so live
//! docs, code, scripts, and CI must not reference it as a present capability.
//! Historical artifacts (changelog, design narrative, lab notebooks,
//! reproducibility provenance comments) are allowlisted by path.

use std::process::Command;

/// Tracked files where the token is legitimate history or provenance.
const ALLOWED: &[&str] = &[
    "CHANGELOG.md",                         // release history
    "docs/SUBSUMPTION_HISTORY.md",          // past-tense design narrative
    ".claude/CLAUDE.md",                    // project memory documents the retirement
    "experiments/",                         // lab notebook + results artifacts
    "scripts/gpu_results_v5.txt",           // historical benchmark output
    "src/trainer/burn_el_trainer.rs",       // "matches CandleElTrainer" provenance
    "src/trainer/burn_taxobell_trainer.rs", // same
    "tests/candle_scrub.rs",                // this file
];

#[test]
fn no_live_candle_references() {
    let out = Command::new("git")
        .args(["ls-files"])
        .output()
        .expect("git ls-files");
    assert!(out.status.success(), "git ls-files failed");
    let files = String::from_utf8(out.stdout).expect("utf8 file list");

    let mut offenders = Vec::new();
    for f in files.lines() {
        let allowed = ALLOWED
            .iter()
            .any(|a| f == *a || (a.ends_with('/') && f.starts_with(a)));
        if allowed {
            continue;
        }
        // Skip binary/unreadable files; the token we police is text.
        let Ok(body) = std::fs::read_to_string(f) else {
            continue;
        };
        if body.to_lowercase().contains("candle") {
            offenders.push(f.to_string());
        }
    }

    assert!(
        offenders.is_empty(),
        "live candle references found (backend removed in 0.14.0; if a hit is \
         genuine history, allowlist it in tests/candle_scrub.rs): {offenders:?}"
    );
}
