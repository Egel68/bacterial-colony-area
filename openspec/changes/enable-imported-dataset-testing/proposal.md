## Why

The algorithm-testing window originally used `TestDataset`, which only understood the legacy `source/` + `masks/` layout and required matching image/mask extensions. The GUI now accepts the manifest-backed 22022540 dataset, but a user report describes a failed run when the imported root is under `C:/Users/Егор/...`. A likely cause is Unicode-path decoding in the image-loading path; this still requires a reproducible Windows test. When all records fail to load, the GUI currently exposes only a generic no-results message, which does not distinguish unreadable files from algorithm failures.

## What Changes

- Make the GUI algorithm-testing flow accept the manifest-backed output of `CocoBboxImporter`, including image/mask pairs with different extensions and optional cropped variants.
- Ensure valid manifest images and masks remain readable when their absolute Windows paths contain Unicode characters, and add a regression test for this path case.
- When a run produces no results, distinguish data-pair decoding failures from algorithm execution failures and show actionable pair/path and failure information.
- Update the GUI dataset guidance to describe both the existing four-directory layout and the manifest-backed imported layout, and point to the project's dataset-source catalog.
- Add a central documentation catalog preserving primary dataset links, versioned identifiers, citations, access/licensing notes, and carefully qualified mappings to local dataset directories; cross-link it from the README and relevant guides.
- Keep dataset files read-only, preserve legacy GUI dataset behavior and algorithm selection, and include cropped records associated with selected source samples when a sample limit is set.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `algorithm-testing-gui`: Extend the GUI dataset contract so users can select and run tests on a manifest-backed imported dataset, and discover the dataset-source catalog from the GUI guidance.

## Impact

- Affected areas: `ui/testing_window.py`, the shared testing scheduler/image-loading path and their tests, testing guidance, README, and the `docs/datasets.md` source catalog. The existing dataset-catalog and guide changes remain in scope. Reuse the existing manifest-capable dataset loading; no new dataset format is needed.
- No new dependencies or changes to the importer/data contract are expected. Existing legacy GUI datasets remain supported; imported source images and masks remain untouched.
- User-facing guidance in the testing window must no longer imply that only the four-directory legacy layout is valid and must point to the source catalog. Documentation must distinguish data records from related papers/software and must not claim exact identity between unverified local copies and external archives.
