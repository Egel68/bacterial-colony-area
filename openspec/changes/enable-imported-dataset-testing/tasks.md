## 1. Route GUI evaluation through manifest-capable loading

- [x] 1.1 Update the GUI worker to use the existing dataset loader that supports manifest, importer, and legacy structures; verify with tests that a `dataset.json` containing JPEG/PNG pairs loads without modifying dataset files.
- [x] 1.2 Preserve GUI sample-limit, cropped-record, comparison, and summary behavior with the shared scheduler; verify source-only, source-plus-cropped, and limited manifest runs using focused tests.
- [x] 1.3 Preserve existing legacy GUI runs and present clear errors for empty, malformed, or unreadable datasets, including when the scheduler skips every record; verify the corresponding legacy and failure-path tests.

## 2. Explain the supported dataset formats in the GUI

- [x] 2.1 Update the dataset hint to describe both the four-directory legacy format and `dataset.json` manifest format, identify `datasets/22022540_imported` as the imported root, and link to the repository's dataset-source catalog; verify the displayed text and link in GUI tests.

## 3. Verify integrated algorithm testing

- [x] 3.1 Add or update regression coverage for running a selected algorithm on a representative imported manifest dataset and receiving results for the available source/cropped records; verify via `tests/test_testing_window.py` and relevant scheduler tests.
- [x] 3.2 Run the focused testing-window and dataset/scheduler test suites; verify legacy and imported datasets pass together without regressions.

## 4. Preserve dataset sources and citations

- [x] 4.1 Add `docs/datasets.md` with all supplied AGAR, Petri plates, 22022540, MDPI, Nature, and NIST links; distinguish dataset records from articles/software and include versioned identifiers, citations, access/licensing facts only where confirmed, and a verification date; verify each entry against its primary source or DOI metadata.
- [x] 4.2 Record the observed local directory mappings and their limits, including the unverified `Petri_plates` correspondence; verify local paths/counts and ensure the documentation does not assert checksum-level identity.
- [x] 4.3 Link the catalog from README, user/testing guides, and the 22022540 pipeline guide, and allowlist the new docs file in `.gitignore`; verify every relative link resolves and the new file is visible to Git.
