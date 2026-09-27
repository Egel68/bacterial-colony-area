## 1. Existing baseline (completed before this revision)

- [x] 1.1 Existing process worker and path-based cache work remains available as comparison baseline; verify current behavior on a 5-object smoke dataset
- [x] 1.2 Existing CLI `--workers` and progress reporting remain available during migration; verify old path is still callable

## 2. Batch loading and sample limiting

- [x] 2.1 Add deterministic `sample_limit` selection to `TestDataset`/`BaselineDataset` and CLI/config; verify `--sample-limit 5` loads exactly 5 source objects in sorted order and does not scan the full execution set
- [x] 2.2 Implement bounded batch loader that reads each image and GT mask once per variant and releases the batch after completion; verify process read counters and loader instrumentation show no repeated image reads per algorithm
- [x] 2.3 Add `batch_size` and `memory_budget` controls with peak-RSS guard; verify oversized batches are split and a 5-object run completes without exceeding the configured guard

## 3. A3 threaded in-memory scheduler

- [x] 3.1 Replace flat `ProcessPool` task submission in `run_all()` with batch tasks over in-memory arrays; verify selected algorithms run for source/cropped variants and no worker calls `imread()`
- [x] 3.2 Apply the same scheduler to `run_baseline()` and evaluate mode; verify cache and report output remain compatible
- [x] 3.3 Implement conservative worker selection and `--workers` upper bound using physical-core, memory and task limits; verify explicit `--workers 1/2` and auto mode report the effective value
- [x] 3.4 Add thread-local algorithm instances for class registrations and safe serial-lane/lock handling for registered singleton instances; verify concurrent class and ONNX/instance algorithms produce correct masks without races
- [x] 3.5 Keep canonical result aggregation order `algorithm -> sample -> variant`; verify A3 and sequential mode have identical binary masks and metrics within `1e-6` on 5 objects
- [x] 3.6 Add adaptive concurrency guard based on initial batch telemetry; verify it can reduce concurrency on queue/RSS pressure and never exceeds the configured worker limit

## 4. Cache without repeated I/O

- [x] 4.1 Extend `PredictionCache` to use the digest of already-read image bytes/array and algorithm parameters; verify cache lookup does not reopen the source image
- [x] 4.2 Integrate cache hit/miss handling into the batch scheduler and baseline path; verify first and second 5-object runs return identical results and the second run skips detection
- [x] 4.3 Preserve invalidation when image content or algorithm parameters change; verify changed content causes a miss

## 5. Telemetry and structured logging

- [x] 5.1 Add `psutil` runtime dependency and telemetry collector with capability flags for unavailable counters; verify the runtime still starts when optional counters are unavailable
- [x] 5.2 Record CPU time/utilization, RSS/peak RSS, process read/write counts and bytes, and bytes/s/operations/s; verify `performance.json` contains these fields or explicit `null` values
- [x] 5.3 Record stage latency for load/cache/queue_wait/detect/metrics/report and calculate p50/p95/p99; verify values are non-negative and task counts reconcile
- [x] 5.4 Record throughput, queue depth, active/completed/failed/retried tasks, cache hit/miss, serial-lane calls, batch parameters and effective workers; verify completed + failed equals submitted
- [x] 5.5 Emit periodic structured events to `performance.jsonl` and final summary to `performance.json`; keep per-task events opt-in through debug mode and verify default logs do not create per-task output
- [x] 5.6 Expose telemetry options `--telemetry`, `--telemetry-interval`, `--performance-output` in report, baseline, evaluate and GUI execution paths; verify output path and disabled mode

## 6. Verification on the 5-object dataset

- [x] 6.1 Run sequential and A3 modes on exactly 5 objects across selected classic algorithms and compare masks/metrics; verify equality and record telemetry
- [x] 6.2 Compare path/process baseline with A3 on the same 5 objects; verify source read operations/bytes and wall time are reported, without requiring a speedup claim when the dataset is too small
- [x] 6.3 Verify cache-enabled first/second runs on 5 objects and compare cache hit/miss and detect latency
- [x] 6.4 Run CLI report, baseline, evaluate and GUI smoke path with `--sample-limit 5`; verify reports are generated and the GUI remains responsive
- [x] 6.5 Run the focused test suite and verify no full-dataset command is used by change acceptance tests
