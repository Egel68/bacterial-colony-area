# Соответствие сценариев спек тестам

> Назначение: каждый `#### Scenario:` из capability-спек соответствует минимум одному тесту.
> Формат: `спека / требование / сценарий` → тест(ы).
> Легенда: `[✓]` — покрыто, `[—]` — не покрыто (GUI/device), `[~]` — покрытие частичное.

## robust-image-loading

| Требование | Сценарий | Тест |
|---|---|---|
| Fallback image loading | Image loads successfully via QPixmap | `tests/test_image_loader.py::TestLoadImage::test_load_color` [✓] |
| Fallback image loading | Image fails QPixmap but loads via OpenCV fallback | `...::TestLoadImageRowPadding::test_padded_width_matches_opencv` [~] (row-padding-патч, ~CMYK) |
| Fallback image loading | Image fails due to QImage row-alignment padding, OpenCV succeeds | `...::test_padded_width_matches_opencv` [✓] |
| Fallback image loading | Any exception from the QPixmap path triggers fallback | `...::TestLoadImageErrorMessage::test_message_mentions_both_loaders` [~] |
| Fallback image loading | Both QPixmap and OpenCV fail to load | `...::test_nonexistent_path` [✓] |
| Robust numpy conversion | QImage with row padding converts correctly | `...::test_padded_width_matches_opencv` [✓] |
| Fallback grayscale | Grayscale fallback loads successfully | `...::TestLoadImageGrayscale::test_grayscale_shape` [✓] |
| Fallback grayscale | Grayscale QImage with row padding falls back | `...::TestLoadImageRowPadding::test_padded_width_matches_grayscale` [✓] |
| User-visible error | Auto-detect with unloadable image | `labeling_window.py` QMessageBox [—] |
| User-visible error | Crop with unloadable image | `labeling_window.py` QMessageBox [—] |
| Supported image formats | Open unsupported format | `...::TestSupportedExtensions::test_contains_common_formats` [✓] + `test_labeling_controller.py::test_filters_unsupported` [✓] |

## colony-analysis

| Требование | Сценарий | Тест |
|---|---|---|
| Petri dish detection | Image loads with specular highlights | `test_colony_detector.py::test_returns_info_for_real_image` [~] |
| Petri dish detection | No highlight contour, Hough fallback | — [—] |
| Petri dish detection | No dish detected at all | `test_colony_detector.py::test_returns_none_for_blank_image` [✓] |
| Petri dish detection | PetriInfo invariant validation | `test_colony_detector.py::test_margin_reduces_radius` / `test_zero_margin_clamped_to_one` [~] |
| Analysis parameter validation | Default params construct without error | `test_analysis_params.py::test_default_values_ok` [✓] |
| Analysis parameter validation | Out-of-range parameter is rejected | `test_analysis_params.py::test_out_of_range_rejected` [✓] |
| Colony detection pipeline | ROI restricts detection to inner dish | — [—] |
| Colony detection pipeline | Empty ROI yields empty result | `test_colony_detector.py::test_empty_roi_returns_zero_mask` [✓] |
| Colony detection pipeline | Solid fill fills enclosed annular colonies | — [—] |
| Colony detection pipeline | Small components are removed by size filter | `test_colony_detector.py::test_removes_small_components` [✓] |
| Area calculations | Areas with inner margin | `test_calculations.py::test_margin_reduces_working_area` / `test_margin_zero_same_as_default` [✓] |
| Area calculations | No Petri info given | `test_calculations.py::test_without_petri_info` [✓] |
| Area calculations | Coverage percent computed | `test_calculations.py::test_coverage_bounds_property` / `test_calculations.py::test_colony_count_one` [~] |
| Analysis controller | analyzed returns typed result | `test_analysis_controller.py::test_analyze_returns_typed_result` [✓] |

## image-labeling

| Требование | Сценарий | Тест |
|---|---|---|
| File list filtered by supported extensions | Unsupported formats are hidden | `test_labeling_controller.py::test_filters_unsupported` [✓] |
| File list filtered by supported extensions | Only unsupported files present | `...::test_only_unsupported_returns_empty` [✓] |
| File list filtered by supported extensions | Missing directory | `...::test_missing_dir_returns_empty` [✓] |
| Session directory selection | Source dir has images | `...::test_source_dir_preferred` [✓] |
| Session directory selection | Empty source falls back to session root | `...::test_source_empty_falls_back_to_session` [✓] |
| Session directory selection | Mask directory selection | `...::test_source_mask_dir` / `test_cropped_mask_dir` [✓] |
| Crop by Petri dish | Crop inside image | `...::test_crop_shape_and_black_background` / `test_crop_center_preserved` [✓] |
| Crop by Petri dish | Crop clamp at image edge | `...::test_crop_near_edge_clamps` [✓] |
| Mask save/load | Save mask creates parent directory | — (см. note) [—] |
| Mask save/load | Load mask shape mismatch | — (см. note) [—] |
| Session export to ZIP | Export includes all files | `...::test_zip_contains_all_files` [✓] |

## algorithm-testing

| Требование | Сценарий | Тест |
|---|---|---|
| Algorithm registry | Registered algorithm is listed | `test_registry.py::test_list_contains_expected` / `test_get_algorithm_returns_instance` [✓] |
| Algorithm registry | Unknown algorithm raises | `test_registry.py::test_unknown_algorithm_raises` [✓] |
| Algorithm interface | Detect returns mask | `test_classic_algorithms.py::TestAlgorithmsRun::test_each_returns_mask` [✓] |
| Algorithm interface | Cropped input detected | `...::test_each_returns_cropped_mask` [✓] |
| Classic variants | Default variant runs | `test_classic_algorithms.py::TestAlgorithmsRun::test_each_returns_mask` [✓] |
| Classic variants | Fallback when dish not detected | — [—] |
| Test dataset loading | Pair with cropped variant | `test_classic_algorithms.py::TestAlgorithmsRun::test_each_returns_cropped_mask` (dataset fixture) [~] |
| Segmentation metrics | Perfect prediction | `test_metrics.py::test_perfect_match` [✓] |
| Segmentation metrics | Disjoint predictions | `test_metrics.py::test_no_match` [✓] |
| Runner and aggregation | Empty metrics list | `test_classic_algorithms.py::TestMeanMetrics::test_empty_list` [✓] |
| Runner and aggregation | Run algorithm on dataset | `test_classic_algorithms.py::TestAlgorithmsOnDataset::test_metrics_nonzero` [✓] |
| HTML report | Report generated | `test_classic_algorithms.py::TestGenerateReport::test_report_created` [✓] |

## data-augmentation

| Требование | Сценарий | Тест |
|---|---|---|
| Augmentation pipeline | 15 пар на исходник, синхронные пути | — [—] (требует albumentations; обычно dev-окружение) |
| Augmentation pipeline | Маска без пары пропускается | — [—] |
| Augmentation seed | Reproducibility with same seed | — [—] |

> Покрытие не тестами: `train/augment.py` требует albumentations; критерий — код-ревью в `openspec/changes/archive/2026-08-08-ml-pipeline/tasks.md`.

## unet-training

| Требование | Сценарий | Тест |
|---|---|---|
| Training configuration | Default config constructs | — [—] (torch) |
| Dataset loading and split | No pairs raises | — [—] (torch) |
| Model registry and U-Net | Unknown model raises | — [—] (torch) |
| Loss and metrics | Perfect prediction on identity | — [—] (torch) |
| Training loop | Train and validation history recorded | — [—] (torch) |
| ONNX export/load | ONNX export produces valid session | — [—] (torch) |

> Требует dev-окружения (torch, ~5 ГБ); не включено в обычный прогон. Мотивация — «Риски» в `docs/implementation-plan.md` (§1 torch).