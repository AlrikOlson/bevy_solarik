# Contributing

Small fixes, bug reports and useful test scenes are welcome. For a larger change, open an issue first so we can talk through it.

## Reporting a problem

Use the [issue tracker](https://github.com/AlrikOlson/bevy_solarik/issues). Include what you expected, what happened, and the smallest scene or steps that reproduce it. For rendering problems, include a screenshot and the relevant log output. GPU model, driver, OS, Bevy version and crate version matter here.

Say whether DLSS is enabled, whether the problem appears in the reference pathtracer, and which mesh or material seems to trigger it.

## Making a change

Keep each PR focused on one change and explain why it helps. Add a regression test when a bug can be checked without a GPU. For shader or rendering changes, include before/after captures from the same scene, camera, resolution and exposure.

Run these from the crate directory:

```text
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo nextest run --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The default checks do not need the DLSS SDK. If you change the optional `dlss` integration, also check and test with `--features dlss`; the [README](README.md#requirements) lists its SDK requirements.

`cargo nextest` is installed separately. If you cannot run a check or a GPU capture, say so in the PR and include what you did run.

Solarik has its own design and release criteria. Changes do not require upstream Solari parity, synchronization, source comparison or regenerated diffs. Evaluate correctness against documented Solarik behavior, analytical tests and its reference pathtracer; evaluate quality and performance with reproducible captures and measurements. Maintain compatibility with the Bevy APIs this crate uses.

After rendering changes, refresh the README screenshots. Hold camera and lighting fixed, compare warm-up lengths to check that GI and temporal history have settled, and inspect the saved full-resolution images before publishing. Record the renderer revision and capture settings; a successful capture command alone does not establish image quality.

Include attribution for copied code and only share test assets you have permission to redistribute. Historical provenance and license notices remain part of the project.
