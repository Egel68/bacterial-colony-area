## 1. Deterministic dataset preparation and split

- [x] 1.1 Implement a seed-based, stable stratified splitter on source COCO image IDs with default 70/15/15 train/val/test proportions; verify that repeated runs with reordered input records produce identical group assignments and that all subsets are non-empty.
- [x] 1.2 Extend `CocoBboxImporter` with an opt-in split mode that writes `subset` to source/cropped records and a `split.json` containing seed, proportions, strategy and source IDs; verify source files remain byte-for-byte unchanged and repeated imports produce the same manifest/split.
- [x] 1.3 Update training dataset selection to honor explicit train/val subsets without ever folding test records into either set, and add a separate test-set loader; verify with a manifest fixture containing all three subsets and retain compatibility for legacy manifests without subsets.
- [x] 1.4 Implement deterministic per-image patch sampling for 512×512 image/mask crops with configurable per-image budget and paired geometric/photometric augmentation; verify image/mask coordinates remain aligned, output tensor sizes are correct, and patch selection is repeatable for a given seed/epoch.

## 2. Compact segmentation architecture and training pipeline

- [x] 2.1 Add and register `mobilenet_v3_small_unet` with optional torchvision ImageNet initialization, lightweight skip decoder and one-channel segmentation logits; verify construction with and without pretrained weights, 512×512 output shape, finite BCE+Dice loss and CPU ONNX export smoke test.
- [x] 2.2 Extend the training loop for patch datasets with deterministic seeds, train-only gradients, fixed validation patches, validation-selected checkpointing, early stopping and per-epoch timing; verify a tiny synthetic run never loads test samples and restores the best validation checkpoint.
- [x] 2.3 Add an end-to-end training CLI for COCO preparation, split seed/proportions, architecture, epoch/batch/patch budget, pretrained opt-out, output directory and explicit model promotion; verify `--help`, invalid configuration errors and a small synthetic end-to-end run.
- [x] 2.4 Implement a held-out evaluator and run report containing per-split IoU/Dice/precision/recall, per-image metrics, class/group counts, seed/split IDs, configuration, best epoch and epoch times; verify test evaluation runs only after checkpoint selection and cannot change selected weights.

## 3. Full-resolution CPU inference and analysis UI

- [x] 3.1 Implement tiled inference for the new ONNX model using only `CPUExecutionProvider`, 512×512 tiles, 384-pixel stride, edge padding, overlap probability averaging and one final threshold; verify mask dimensions/binary values, complete edge coverage, small/non-divisible images, overlap merging and progress callback counts with a fake ONNX session.
- [x] 3.2 Register the new bundled model with the tiled inference adapter while preserving existing classic algorithms and `NN:colony_seg`; verify both old and new names appear in the algorithm registry and the legacy model retains its existing inference path.
- [x] 3.3 Extend `AnalysisController` to calculate area/count/results from the selected detection algorithm's mask while preserving classic defaults and Petri dish geometry behavior; verify classic regression and injected fake-model results in controller tests.
- [x] 3.4 Add algorithm selection and asynchronous `QThread` inference to the photograph analysis window, display per-tile progress in its status bar, handle recoverable errors and avoid conflicting recalculations; verify a GUI test shows progress, stays responsive and displays the completed NN mask/results.

## 4. CPU runtime and binary packaging

- [x] 4.1 Move CPU ONNX Runtime into base runtime dependencies, keep PyTorch/torchvision in the full training extra, refresh `uv.lock` and runtime/build documentation; verify fresh runtime/dev environments install ONNX Runtime but do not install PyTorch.
- [x] 4.2 Update Nuitka packaging to include ONNX Runtime CPU provider libraries and all `models/*.onnx` files without PyTorch; verify generated build flags and a clean Linux build can create an ONNX session with `CPUExecutionProvider`.
- [x] 4.3 Add a packaged-app smoke test for registration and CPU inference using a small test image, and record the measured binary size/range; verify the built application runs the bundled model without CUDA or an external Python installation.

## 5. Train, evaluate and deliver the bundled model

- [x] 5.1 Prepare the real `datasets/22022540` source into a separate output directory with the fixed train/val/test split; verify 369 source images are accounted for, source categories are represented where possible, and source dataset files are unchanged.
- [x] 5.2 Train the pretrained MobileNetV3-Small segmenter, select the checkpoint solely on validation metrics, tune only train/validation patch budget until the measured RTX 2060 epoch target is ≤60 seconds where feasible, and preserve the run/checkpoint/report; verify the saved split seed/IDs and per-epoch timing.
- [x] 5.3 After checkpoint selection is frozen, evaluate test metrics once and export a separate single-file ONNX model without overwriting `models/colony_seg.onnx`; verify ONNX Runtime CPU output agrees with PyTorch output on representative patches within numerical tolerance.
- [x] 5.4 Benchmark the exported model on full-resolution held-out images through the production CPU tiling path and record CPU model, image sizes, warmup, median/p95 latency and model size; verify benchmark includes all tiles and stitching, not only a 512×512 patch.
- [x] 5.5 Promote the reviewed ONNX artifact under a new `models/` name and document training reproduction, split/test metrics, weak-label limitation and measured CPU performance; verify the packaged analysis UI can select the new model and existing model files remain unchanged.

## 6. Integration verification

- [x] 6.1 Run focused dataset, model, tile-inference, controller, GUI and packaging tests; verify all newly specified scenarios have automated coverage.
- [ ] 6.2 Run the full supported test suite and clean Linux/Windows build checks; verify the app launches, the new bundled model registers, CPU tiled inference completes asynchronously with status updates, and the binary excludes PyTorch. Linux verification passed: all 238 tests (including GUI and slow tests) offscreen, clean Nuitka onefile build, and CPU-only headless smoke tests for both bundled models with PyTorch absent from the build environment and payload. Windows build verification remains pending CI.
