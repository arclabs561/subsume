# Contributing to subsume

Thanks for your interest. subsume is geometric region embeddings (boxes, cones, octagons, Gaussians, hyperbolic intervals, sheaf networks) for subsumption, entailment, and logical query answering.

## Before you start

For non-trivial work (new APIs, features, large refactors), open an issue first to align on scope. Drive-by bug fixes and doc patches don't need an issue.

## Setup

- Use the current stable Rust toolchain for development. The library MSRV is
  `1.87`; development dependencies such as Heyting require a newer compiler.
- Optional: `cargo-nextest` for faster test runs (`cargo install cargo-nextest`).

```sh
just check
cargo test --features burn-ndarray --example el_clqa_galen
```

## Style

- Direct, lowercase prose in commits. No marketing words ("powerful", "robust", "elegant"). No em-dashes in prose.
- Use descriptive Conventional Commit subjects, such as `fix: preserve calibration ties`.
  Keep each commit focused on one change.
- Run `just check` before committing.

## Testing

- Test the feature set affected by the change; [CI](.github/workflows/ci.yml)
  defines the supported checks. `--all-features` also enables the optional
  LibTorch backend and requires its external libraries.
- Test names should describe the property under test, not the function under test.

## Pull requests

- Keep PRs scoped to one concern.
- Show before/after for behavior changes.
- Link the related issue.
- CI must be green before requesting review.

## License

Dual-licensed under MIT or Apache-2.0 at your option. By contributing you agree your contributions are licensed under both.
