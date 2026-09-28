## 1. Reference profile and correctness baseline

- [ ] 1.1 Add a reproducible benchmark profile record for the implementation host; verify it captures OS, CPU model, affinity/quota, RAM, OpenCV/ONNX Runtime versions, dataset/sample identities, image dimensions, and warm-cache conditions.
- [ ] 1.2 Capture baseline binary masks and segmentation metrics for the selected source/cropped, ordinary, boundary, and largest-image fixtures; verify repeated baseline runs produce identical mask hashes and metrics within `1e-6`.
- [ ] 1.3 Record sequential and current-worker baseline timings, process CPU-time, peak RSS, and existing GUI heartbeat on the reference profile; verify the report distinguishes measured warm-cache behavior from unmeasured cold-storage I/O.

## 2. Pipeline telemetry and resource bounds

- [ ] 2.1 Separate logical file-read and image/mask decode timings and byte counts in pipeline telemetry; verify a warm-cache test still reports logical reads and distinct read/decode latency when physical read counters remain zero or unavailable.
- [ ] 2.2 Measure each task's queue wait from its own submission and distinguish compute-worker wait for prepared input from executor queue wait; verify deterministic slow-loader and queued-task tests report the two waits separately.
- [ ] 2.3 Report process CPU-time, wall-time, CPU core-equivalents, available CPU/affinity information, effective outer/native worker limits, current/peak RSS, and unavailable platform counters as unknown; verify telemetry degrades gracefully when an OS counter is unavailable.
- [ ] 2.4 Derive automatic worker capacity from process-visible CPU affinity/quota with a conservative fallback, while retaining `--workers` as an upper bound; verify mocked restricted-CPU, unknown-quota, and explicit `--workers 1/2` cases choose valid limits.
- [ ] 2.5 Bound active plus prefetched decoded input arrays by `memory_budget`, reserve conservative headroom for concurrent algorithm work, and isolate a single oversized pair; verify tests cover backpressure, no unbounded queue growth, and successful processing of an oversized valid sample.
- [ ] 2.6 Coordinate decode workers, outer algorithm workers, and OpenCV/ONNX native threading under one resource policy; verify benchmark comparisons include classic and tiled ONNX workloads and GUI auto-policy remains automatic while CLI overrides remain supported.
- [ ] 2.7 Add bounded decode/compute overlap with backpressure and deterministic cleanup on failures/cancellation; verify the pinned CPU-bound acceptance run reports input-starvation and stays at or below 5% without exceeding the configured decoded-input target.

## 3. Remove redundant algorithm and comparison work

- [ ] 3.1 Profile dish search, component filtering, and classic preprocessing separately on ordinary and 20–35 MP images; verify the profiling output identifies per-stage wall time and does not change reference masks.
- [ ] 3.2 Replace the confirmed repeated component-label scan with an equivalent vectorized lookup where profiling supports it; verify `tests/test_colony_detector.py` plus reference fixtures produce bitwise-identical masks.
- [ ] 3.3 Reuse only invariant per-sample geometry/preprocessing context across compatible classic runs; verify source/cropped and parameter-invalidation tests match independent execution bitwise and reject incompatible context.
- [ ] 3.4 Combine main-run and pairwise-comparison work so each compatible algorithm/sample/variant is detected once, running only missing comparison algorithms when needed; verify `tests/test_runner.py` and `tests/test_testing_window.py` preserve displayed selection and comparison winners without duplicate detector calls.

## 4. Background-safe image decoding and worker lifecycle

- [ ] 4.1 Add a worker-safe image and grayscale decoder that avoids `QPixmap` off the GUI thread and falls back to OpenCV while preserving supported formats, BGR/grayscale `uint8` contracts, and the existing GUI-thread loading behavior; verify `tests/test_image_loader.py` covers format, row-padding, fallback, and failure cases.
- [ ] 4.2 Define a reusable background-operation lifecycle with immutable input snapshots, queued status/progress/result/error delivery, generation checks, cooperative cancellation, and nonblocking deferred close; verify focused Qt tests cover stale results, native-call completion after close request, and widget access only on the GUI thread.

## 5. Keep photograph analysis responsive

- [ ] 5.1 Move initial image decode, dish search, and first analysis out of `AnalysisWindow` construction; verify a Qt heartbeat and resize/input events continue while a slow loader and detector run, with recoverable load/detection errors.
- [ ] 5.2 Run classic recalculation and selected algorithm inference in background workers with phase/progress state and conflict prevention; verify `tests/test_analysis_model_selection.py` exercises classic and tiled-NN paths, errors, cancellation, and delayed close.
- [ ] 5.3 Move full-resolution view compositing/scaling to a viewport-bounded, stale-result-safe presentation path without downsampling analysis data; verify rapid mode/resize changes show only the latest view while masks and algorithm inputs retain full resolution.
- [ ] 5.4 Save analysis output from an immutable view/mode snapshot outside the GUI thread; verify a large-image save test keeps the heartbeat active and preserves the current `QPixmap.save` content, displayed dimensions, and extension-selected encoding.

## 6. Keep image labeling responsive and pixel-accurate

- [ ] 6.1 Load a selected image and its mask asynchronously and reuse its decoded original for auto-detect/crop; verify selection-generation tests ensure late results cannot overwrite a newer file or mode.
- [ ] 6.2 Enumerate and sort large session file lists outside the GUI thread; verify a thousands-of-files fixture keeps the heartbeat active and returns the same supported files in sorted order.
- [ ] 6.3 Run dish auto-detection, crop generation, and crop-file writing in a worker with status/progress and error recovery; verify controller and Qt tests preserve dish geometry, crop pixels, and the existing saved-file naming contract.
- [ ] 6.4 Copy batches of source files asynchronously with progress while preserving `shutil.copy2` overwrite behavior for matching names; verify tests cover success, copy failure, and the documented conflict behavior.
- [ ] 6.5 Replace whole-frame canvas reconstruction on every brush event with viewport/local-region rendering while keeping full-resolution mask coordinates and zoom semantics; verify brush tests compare exact mask pixels at multiple zoom levels and a large-image heartbeat remains within the reference limits.
- [ ] 6.6 Save masks from immutable snapshots and export ZIP files through a temporary destination that is published only after success; verify large save/export tests keep the GUI responsive, cancellation/failure preserves any previous target file, and no partial archive is published.
- [ ] 6.7 Apply the shared deferred-close lifecycle and safe cancellation to labeling operations; verify closing during an uninterruptible detector/export shows a completion state, keeps the event loop active, stops subsequent work, and closes at the next safe point.

## 7. Keep algorithm testing and reports responsive

- [ ] 7.1 Emit distinct dataset-scan, read/decode, algorithm, comparison, and result-preparation progress plus contextual per-task errors; verify `tests/test_testing_window.py` observes phase transitions and successful tasks remain available after an individual failure.
- [ ] 7.2 Present large result/comparison sets incrementally or through a model/view instead of building every table cell in one GUI-thread pass; verify a hundreds/thousands-row fixture renders while heartbeat and user input continue.
- [ ] 7.3 Generate and save HTML reports in the background from a stable result snapshot, publishing the final path only on success; verify the report contents remain compatible and export failure/cancellation neither publishes a partial report nor modifies a pre-existing target file.
- [ ] 7.4 Add a cancel action and deferred close to algorithm-testing runs and report exports; verify new work stops at safe boundaries, non-interruptible native work never blocks the GUI event loop, final report paths are not partially overwritten, and late results are not applied after closure.

## 8. Cross-cutting acceptance

- [ ] 8.1 Add automated Qt heartbeat coverage for analysis, labeling selection/brush/copy/save/ZIP, testing run/result presentation/report, and CPU-bound background work; verify the pinned reference profile meets p95 heartbeat ≤100 ms and maximum interval ≤250 ms at a 50 ms timer period.
- [ ] 8.2 Run the same deterministic reference workload in sequential and conservative-auto modes and publish read/decode/detect latency, input-starvation, throughput, CPU cores consumed, effective concurrency, and peak RSS; verify correctness checks pass and no acceleration claim is based on worker count alone.
- [ ] 8.3 Run the focused pipeline, classic-algorithm, decoder, controller, and GUI regression suites followed by `UV_PROJECT_ENVIRONMENT=.venv-dev uv run pytest`; verify legacy CLI flags, dataset/report formats, result semantics, and labeling precision remain compatible.
