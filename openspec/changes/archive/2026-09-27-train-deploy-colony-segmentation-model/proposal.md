## Why

The project can import the downloaded 22022540 dataset and train/export U-Net models, but it has no persisted train/validation/test split, and the current analysis window cannot select a neural model for a photograph. The standalone build also omits ONNX Runtime, so a bundled model is not yet usable there. This change delivers a reproducible, held-out-evaluated, compact colony segmenter that is available from the actual analysis UI and works in the packaged application.

## What Changes

- Add a deterministic, group-aware split for the COCO dataset: all representations of one source dish stay in the same train, validation, or test subset, and repeated preparation with the same inputs and seed produces the same assignments.
- Add a compact PyTorch MobileNetV3-Small encoder with a lightweight segmentation decoder. Fine-tune from torchvision ImageNet weights by default, use validation-only checkpoint selection/early stopping, and evaluate the held-out test subset only after training.
- Provide a reproducible training command that prepares the split dataset, trains the model, reports per-split metrics and CPU inference latency, and exports the selected checkpoint to ONNX. Target at most 60 seconds per training epoch on the available RTX 2060; measure and report CPU timings on full-resolution images rather than promising a fixed latency on arbitrary hardware.
- Add the new ONNX model beside the existing `models/colony_seg.onnx`; retain the existing model as a selectable fallback/comparison.
- Add an algorithm selector to the photograph analysis UI so users can choose the new bundled model or an existing registered algorithm. Neural inference SHALL use CPU only, cover the complete original image at its source resolution using overlapping model-sized tiles, return a mask matching the original dimensions, and run asynchronously so the UI remains responsive. Show per-tile progress/status in the analysis window's status bar.
- Include the ONNX Runtime CPU dependency in the runtime and clean binary-build environments, bundle the model weights, and verify inference from the packaged application without PyTorch.

## Capabilities

### New Capabilities
- `colony-model-training`: Deterministic train/validation/test preparation, compact pretrained segmentation training, held-out evaluation, reproducible artifacts, and CPU performance measurement for the downloaded colony dataset.
- `analysis-model-selection`: Selection and use of a registered classic or bundled neural algorithm in the photograph analysis window.

### Modified Capabilities
- `coco-bbox-importer`: Persist reproducible train/validation/test assignments for source dishes and keep derived cropped samples in the same subset.
- `unet-training`: Add the MobileNetV3-Small segmentation architecture and pretrained initialization to the model registry/training contract.
- `neural-model-algorithms`: Make bundled ONNX inference available through the CPU runtime in the standalone application.
- `development-environments`: Include ONNX Runtime in the application runtime while keeping PyTorch and training dependencies in the full development environment.
- `nu-packaging`: Package ONNX Runtime CPU support together with bundled ONNX weights without adding PyTorch to the binary.

## Impact

- **Training/data:** `train/dataset_adapters.py`, `train/dataset.py`, `train/models/`, the training CLI/config/loop, and new tests. The source dataset remains read-only; prepared files and run artifacts are written to explicit output locations.
- **Desktop UI:** `ui/analysis_window.py` and `ui/controllers/analysis_controller.py`; existing algorithm APIs remain supported. Inference uses ONNX Runtime's CPU provider and an asynchronous worker; arbitrary hardware has no universal five-second guarantee, so full-image latency is measured and reported.
- **Runtime/build:** `pyproject.toml`, `uv.lock`, environment/build documentation and Nuitka packaging. The application runtime and binary grow by the ONNX Runtime CPU package and the new model, but PyTorch remains excluded.
- **Model assets:** add a separately named ONNX artifact under `models/`; keep the existing model unchanged. Training records split seed, dataset/config metadata, validation-selected checkpoint, test metrics, model size, and measured inference time.
