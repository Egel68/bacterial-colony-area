## Context

See `proposal.md` for the problem. `TestingWindow._RunWorker` currently constructs `TestDataset`, whose legacy format uses `source_path`/`source_mask_path`/`cropped_path` fields. The shared scheduler supports both this shape and `BaselineDataset` entries (`image_path`, `mask_path`, `variant`); `BaselineDataset` already reads `dataset.json`, resolves manifest records against the root, and can load the importer output. Its current sample-limit implementation loads source and cropped records, then retains cropped entries matching selected source ids. The imported directory in this checkout contains 369 source and 369 cropped records, with JPEG images, PNG masks, and no explicit subset field.

## Goals / Non-Goals

**Goals:**
- Route GUI runs through a dataset abstraction that already supports `dataset.json` and the shared scheduler.
- Preserve legacy GUI testing and current source/cropped metric behavior.
- Describe both supported layouts in the GUI hint, point users to the imported root directory, and link to the dataset-source catalog.
- Preserve provenance and citation information for the external datasets in one project document, with explicit caveats for unverified local copies.

**Non-Goals:**
- Change `TestDataset` or the CLI `report`/`baseline`/`evaluate` behavior.
- Add dataset downloading, importing, conversion, subset filtering, or new manifest fields.
- Change the annotation semantics: 22022540 masks remain weak labels rasterized from bounding boxes.
- Download, mirror, reconcile, or checksum external datasets as part of this change; the catalog records source links and the current limits of local correspondence only.

## Decisions

1. **Reuse `BaselineDataset` as the GUI worker's dataset loader, with a GUI sample view for limits.** It already handles manifest-based data (including different image/mask extensions and `source`/`cropped` records), and `testing.scheduler._sample_refs` supports its `BaselineSample` representation. Apply a GUI-side selected-sample view after loading so source limits consistently retain the matching cropped records in both manifest and legacy layouts; this avoids changing CLI baseline behavior and avoids the legacy loader's early-break edge case. This is less duplication and lower risk than teaching `TestDataset` a second format or introducing a new adapter.
2. **Leave the worker's execution/report contract in place.** `run_all`, comparison, summaries, telemetry, and HTML export already operate via the shared scheduler and results structure. Only input dataset construction and any dataset-name/path assumptions need to be checked when using `BaselineDataset`. Since the scheduler skips records whose files cannot be decoded, the worker must also detect when every selected sample was skipped and report a failed run rather than displaying an empty success.
3. **Preserve dataset-defined paths and read-only semantics.** Do not copy, rename, or rewrite image/mask files or `dataset.json`; let the existing manifest loader resolve relative paths. Do not interpret `subset` in this GUI change: the current imported data has no subset assignments, and the request is to test the imported dataset as available.
4. **Make GUI help format-neutral.** Retain the existing legacy directory explanation, add the manifest-backed `dataset.json` option, and give `datasets/22022540_imported` as the example path. Do not hard-code auto-selection or trigger an import from the UI.
5. **Keep data provenance in one documentation source.** Add `docs/datasets.md` for the supplied AGAR, Petri plates, 22022540, and NIST links, distinguishing archive/record links from associated publications and software. Use version-specific DOI links for Figshare, cite access and licensing only where confirmed by the source, and cross-link the catalog from the README and relevant guides. Do not claim the local Petri plates folder is an exact copy: the observed local count and filenames do not directly match the published approximate count/example filename, and no checksums were compared.

## Risks / Trade-offs

- **[Risk]** `BaselineDataset` may expose detail differences in sample naming or cropped aggregation compared with `TestDataset`. → **Mitigation:** verify the existing scheduler's mapping for `variant="source"`/`"cropped"`, ensure source and cropped results are distinct, and add GUI-level tests for source, cropped, and limited runs.
- **[Risk]** Large full-resolution dataset runs may take substantial time and memory. → **Mitigation:** preserve the existing batch-size and sample-limit controls; loading remains lazy through `load_images=False`/path-based samples.
- **[Risk]** Manifest parsing may fail for malformed JSON or missing paths, and the scheduler skips undecodable files. → **Mitigation:** surface a clear GUI error and do not proceed with an empty result; cover invalid manifests and all-files-unreadable cases.
- **[Trade-off]** The GUI will share a loader whose name refers to baseline evaluation. → **Mitigation:** treat it as a reusable implementation for now; extracting a neutral common loader is not needed for this narrowly scoped change.
- **[Risk]** Dataset pages, download endpoints, licenses, or access policies can change, and local files may differ from a source release. → **Mitigation:** retain versioned DOIs and verification date in the catalog, link to primary records, and label unverified local associations rather than asserting identity.

## Migration Plan

No data migration is required. Existing four-directory datasets continue to use the legacy fallback. Selecting a valid manifest root makes its records available to GUI testing; rollback consists of reverting the GUI routing and hint, with no changes to user dataset files. The source catalog and its cross-links are documentation-only and do not download or modify datasets.
