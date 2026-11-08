# Tasks

## 1. Paired dataset preparation

- [x] 1.1 Extend 22022540 preparation to preserve dish geometry and persist the source-level 70/15/15 split before variants are generated; verify stable group IDs and source immutability with importer/split tests.
- [x] 1.2 Generate one full-size black-background image per source by zeroing only pixels outside the dish circle; verify unchanged interior pixels and GT mask, plus explicit failure for missing geometry in focused tests.
- [x] 1.3 Add augmented manifest records with stable source linkage and inherited subset; verify every source/variant pair remains in exactly one split and repeats produce the same IDs.
- [x] 1.4 Document background transformation, split order, and generated dataset layout in the 22022540 training guide; verify the documented preparation command matches CLI help.

## 2. Scratch training and paired evaluation

- [x] 2.1 Extend `train-colony` with an opt-in black-background experiment and `unet` model selection while preserving current defaults; verify new options and backward compatibility through CLI tests.
- [x] 2.2 Train `unet` from random initialization on train-only original/black pairs, select checkpoints on val, require CUDA with no CPU fallback, and record the actual GPU; verify no pretrained/resume weights are loaded and test records are excluded.
- [x] 2.3 Evaluate the bundled baseline on original and black held-out test variants before optimizer updates, then evaluate the new model on those exact IDs and both backgrounds; verify ID equality and that unrecoverable baseline training membership is reported as unknown.
- [x] 2.4 Aggregate pixel-level TP/FP/FN/TN and derive precision/recall from the totals; verify matrix orientation, formulas, zero-denominator behavior, and per-sample counts with metric tests.
- [x] 2.5 Write the canonical JSON and comparison HTML with split IDs, seed, model/configuration, threshold, provenance, per-sample metrics, and four model/background summaries; verify both reports contain matching IDs and confusion matrices.
- [x] 2.6 Update training documentation with the scratch-U-Net experiment command, mandatory CUDA behavior, prepared-dataset continuation, and report interpretation; verify the documented invocation is accepted by CLI help.

- [x] 2.7 Allow an explicitly requested continuation from a fully prepared dataset; validate the complete source/variant pairing, masks, files and inherited split before baseline evaluation or training.

## 3. End-to-end experiment validation

- [x] 3.1 Run a small end-to-end synthetic COCO experiment and verify import, split, black variants, scratch training, and report generation work together without modifying input files.
- [x] 3.2 Run the full experiment on dataset 22022540 with seed 42 and the 70/15/15 split; verify the complete paired dataset is prepared first, baseline original/black metrics precede training, U-Net trains on CUDA, and post-training metrics use the identical test IDs/backgrounds with unchanged source hash, precision/recall, confusion matrices and baseline-overlap status.
