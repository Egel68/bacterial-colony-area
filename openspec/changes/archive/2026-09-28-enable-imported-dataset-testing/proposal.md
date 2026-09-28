## Why

The algorithm-testing window accepts an arbitrary dataset directory, but its worker uses `TestDataset`, which only understands the legacy `source/` + `masks/` layout and requires matching image/mask extensions. The imported 22022540 dataset stores PNG masks beside JPEG images and describes pairs in `dataset.json`, so choosing `datasets/22022540_imported` currently produces an empty test set instead of allowing an interactive algorithm run.

## What Changes

- Make the GUI algorithm-testing flow accept the manifest-backed output of `CocoBboxImporter`, including image/mask pairs with different extensions and optional cropped variants.
- Update the GUI dataset guidance to describe both the existing four-directory layout and the manifest-backed imported layout, and point to the project's dataset-source catalog.
- Add a central documentation catalog preserving primary dataset links, versioned identifiers, citations, access/licensing notes, and carefully qualified mappings to local dataset directories; cross-link it from the README and relevant guides.
- Keep dataset files read-only, preserve legacy GUI dataset behavior and algorithm selection, and include cropped records associated with selected source samples when a sample limit is set.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `algorithm-testing-gui`: Extend the GUI dataset contract so users can select and run tests on a manifest-backed imported dataset, and discover the dataset-source catalog from the GUI guidance.

## Impact

- Affected areas: `ui/testing_window.py`, its tests, testing guidance, README, and a new `docs/datasets.md` source catalog. Reuse the existing manifest-capable evaluation dataset loading and shared scheduler; no new dataset format is needed.
- No new dependencies or changes to the importer/data contract are expected. Existing legacy GUI datasets remain supported; imported source images and masks remain untouched.
- User-facing guidance in the testing window must no longer imply that only the four-directory legacy layout is valid and must point to the source catalog. Documentation must distinguish data records from related papers/software and must not claim exact identity between unverified local copies and external archives.
