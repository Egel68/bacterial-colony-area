## Context

See `proposal.md` for the reported failure. The GUI uses `BaselineDataset` for manifest, legacy, and importer layouts; manifest paths are resolved and checked before the shared scheduler reads file contents. The scheduler currently calls `cv2.imread` directly on those paths and skips pairs when decoding returns no image or mask. A user reports that the all-results-failed message occurs for a dataset below `C:/Users/Егор/...`; Unicode-path incompatibility in this direct decoder is the leading hypothesis, but it has not yet been reproduced on Windows. The imported directory in the original checkout contained 369 source and 369 cropped records, with JPEG images, PNG masks, and no explicit subset field.

## Goals / Non-Goals

**Goals:**
- Decode supported image and mask files when their Windows paths contain Unicode characters.
- Make a zero-result GUI failure identify whether pairs could not be read or algorithm tasks failed, and include actionable path/reason information.
- Preserve manifest, importer, and legacy dataset behavior, including source/cropped metrics and sample limits.

**Non-Goals:**
- Change `TestDataset` or the CLI `report`/`baseline`/`evaluate` behavior.
- Add dataset downloading, importing, conversion, subset filtering, or new manifest fields.
- Change the annotation semantics: 22022540 masks remain weak labels rasterized from bounding boxes.
- Change dataset contents, paths, or the manifest contract, or add a new image-decoding dependency.

## Decisions

1. **Keep manifest parsing separate from byte decoding.** `BaselineDataset` remains responsible for resolving and validating manifest records and the GUI sample view remains responsible for source limits. Fix decoding at the shared scheduler's file-read boundary so manifest, importer, and legacy records all benefit without changing their formats or the CLI `TestDataset` contract.
2. **Use a Unicode-safe, non-GUI decoder for scheduler paths.** Read file bytes through a Unicode-capable filesystem API and pass the byte buffer to OpenCV's in-memory decoder with the existing color/grayscale mode. This avoids relying on Windows `cv2.imread` path handling, adds no dependency, and works in the scheduler's worker threads and non-GUI callers. Reusing `QPixmap`/the GUI image loader was considered but rejected because it introduces GUI-thread/application constraints into shared pipeline code. Requiring users to move datasets to ASCII-only paths is also rejected as an unacceptable workaround.
3. **Keep data-load and algorithm failures separately observable.** The scheduler must retain load-failure diagnostics independently of optional telemetry and track algorithm-task failures separately. If no pairs can be decoded, the worker reports a dataset read/decode error with at least one affected pair path and reason. If pairs were decoded but every algorithm task failed, the worker reports an algorithm-run error with representative task context. Successful partial runs continue to return results under the existing contract.
4. **Preserve dataset-defined paths and read-only semantics.** Do not copy, rename, or rewrite image/mask files or `dataset.json`; keep relative-path resolution and do not interpret `subset` as part of this fix.
5. **Verify the reported platform-specific case directly.** Add a GUI-worker regression using a manifest rooted under a directory with non-ASCII characters, and run it on Windows CI as well as other supported test platforms. Also cover unreadable pairs and algorithm tasks that all fail, asserting that the resulting GUI errors identify the correct failure class and useful context. Existing source/cropped, sample-limit, and legacy tests remain regression coverage.

## Risks / Trade-offs

- **[Risk]** Unicode-safe decoding could change channel/bit-depth behavior or supported formats. → **Mitigation:** preserve OpenCV's existing color/grayscale decode flags and add tests comparing decoded arrays and exercising representative supported formats.
- **[Risk]** Partially unreadable datasets may have many failed records. → **Mitigation:** retain representative paths/reasons and counts rather than flooding the GUI; keep detailed telemetry/logging available where configured.
- **[Risk]** Users may encounter other all-failed runs unrelated to Unicode paths. → **Mitigation:** treat Unicode as a hypothesis until Windows regression coverage proves the path case, and make failure diagnostics distinguish data decoding from algorithm execution.
- **[Risk]** `BaselineDataset` may expose detail differences in sample naming or cropped aggregation compared with `TestDataset`. → **Mitigation:** preserve the existing scheduler mapping for `variant="source"`/`"cropped"` and retain GUI-level source, cropped, sample-limit, and legacy tests.
- **[Trade-off]** The GUI will share a loader whose name refers to baseline evaluation. → **Mitigation:** treat it as a reusable implementation for now; extracting a neutral common loader is not needed for this narrowly scoped change.
- **[Risk]** Dataset pages, download endpoints, licenses, or access policies can change, and local files may differ from a source release. → **Mitigation:** retain versioned DOIs and verification date in the catalog, link to primary records, and label unverified local associations rather than asserting identity.

## Migration Plan

No data migration is required. Existing manifest, importer, and four-directory datasets remain at their current paths and are read-only. After the fix, users should be able to rerun the same manifest from its Unicode-containing Windows path; rollback requires reverting only the shared decoding/diagnostic changes and their tests.
