## Context

См. `proposal.md` — Why/What Changes. В проекте уже есть `CocoBboxImporter`, `DatasetManifest` с subset `train|val|test`, BCE+Dice `BaseSegmenter`, CLI обучения и регистрация bundled `models/*.onnx` в тестовом алгоритм-реестре. Текущий trainer делит только на train/val случайной записью, адаптер импортёра не сохраняет subset, а UI анализа непосредственно вызывает классический `AnalysisController`. Runtime-зависимости сейчас исключают ONNX Runtime, включая clean Nuitka build env. COCO содержит 369 снимков, сгруппированных по одному из 24 классов, с 56 865 bboxes; загруженные источники нельзя изменять. Для разработки доступна RTX 2060 6 GB; приложение должно выполнять инференс на CPU на целевых устройствах.

## Goals / Non-Goals

**Goals:**
- Один воспроизводимый путь от неизменяемого COCO-источника к материализованному датасету с групповой train/val/test-разбивкой, обученному checkpoint, test-отчёту и packaged CPU-совместимой ONNX-моделью.
- Не допустить утечки изображений/производных патчей между выборками и не использовать test при выборе checkpoint.
- Обрабатывать крупные изображения полноразмерными patch-тайлами как при обучении, так и в UI; сообщать прогресс по тайлам без блокирования UI.
- Сохранить существующие классические алгоритмы и bundled ONNX-модель.

**Non-Goals:**
- Многоклассовая классификация бактерий; выход — бинарная маска наличия колоний.
- Гарантия абсолютного времени инференса для произвольного CPU или обучения ≤60 секунд/эпоха вне зафиксированного RTX 2060 benchmark; фактические измерения фиксируются.
- Гарантия семантически точных попиксельных масок из исходных bbox; это weak supervision, с возможным последующим улучшением на ручной разметке.
- Использование GPU в production-инференсе или включение PyTorch в пользовательский бинарник.

## Decisions

### D1. Детерминированный group split до создания патчей

Добавить опциональную подготовку split в COCO-импортёр. Сначала сгруппировать записи по исходному COCO image id, выполнить детерминированную стратифицированную разбивку изображений по единственному `category_id`. Для имеющегося формата, где каждый снимок относится к одному виду, это сохраняет представленность 24 классов. Группы сортируются по стабильному ID, а seed-based перемешивание внутри каждой страты использует seed 42; целевые доли по изображениям — 70/15/15. Для малой страты стараться направлять минимум один снимок в test и, если возможно, один в val; валидировать, что все три выборки непусты и выводить их размеры. К cropped-вариантам назначать subset исходной фотографии. Записывать subset в dataset manifest и отдельный `split.json` с seed, пропорциями, алгоритмом и исходными sample IDs. Повторный запуск с теми же данными/параметрами идемпотентен.

Patch Dataset получает записи только своей выборки, а координаты тайлов генерирует после split. Train coordinates можно варьировать по эпохам детерминированно от `(seed, epoch, sample_id)`, а validation/test coordinates фиксированы. Это не добавляет тайлы в manifest и исключает межвыборочную утечку. Обычный `import-22022540` без флага split сохраняет прежнее поведение.

**Альтернативы:** случайно делить 369 записей без стратификации или делить патчи. Разбивка на уровне исходного снимка с сохранёнными assignments воспроизводима, проще для аудита и не допускает leakage.

### D2. Три выборки; test-only после обучения

Train даёт градиенты, val управляет best-checkpoint и early stopping, test читается отдельным evaluator после завершения выбора. Test метрики не участвуют в подборе модели, эпохи, порога или гиперпараметров. Seed, group IDs, proportions, лучшая эпоха и конфигурация сохраняются в отчёте.

**Альтернатива:** только train/test не предоставляет независимый сигнал для ранней остановки и повышает риск переобучения; пользователь согласовал train/validation/test.

### D3. MobileNetV3-Small encoder и лёгкий сегментационный decoder

Добавить архитектуру `mobilenet_v3_small_unet`, зарегистрированную через существующий `BaseSegmenter` registry. Использовать `torchvision.models.mobilenet_v3_small` с ImageNet-pretrained весами по умолчанию (флаг отключения нужен для offline/reproduction); извлекать несколько spatial feature levels и применять компактные skip/fusion блоки с bilinear upsampling к одному выходному logit-каналу размером 512×512. Архитектура сохраняет существующий BCE+Dice контракт. До упаковки зафиксировать inference output contract (logits или probabilities) и привести его к единому порогу в ONNX adapter.

Pretrained weights улучшают sample efficiency на 369 исходниках; против переобучения применяются group holdout, аугментации, weight decay, validation-only checkpoint selection и early stopping. В пользовательский пакет попадает только обученная ONNX-модель, не PyTorch/pretrained checkpoints. Benchmark обучения на RTX 2060 должен синхронизировать CUDA для корректных замеров, отметить warmup и считать один полный проход обучающих patches за эпоху.

**Альтернативы:** полный U-Net (медленнее/крупнее), UNetSmall с нуля (нет transfer learning), классификационный MobileNet без spatial skip features (теряются мелкие объекты), новая тяжелая dependency для decoder. Выбран torchvision encoder и небольшой decoder.

### D4. Patch-level обучение на оригинальном разрешении

Для новой модели не использовать текущий путь, который уменьшает весь кадр до квадрата 512: на снимках 2800–5600 px колонии размером около 50 px теряются. Добавить patch dataset: 512×512 окна исходного изображения и маски, лимитируемый конфигом `patches_per_image`, budget патчей на снимок и контролируемая смесь foreground/background locations. Train anchor/random координаты должны варьироваться детерминированно по эпохам; validation/test patches — фиксированные, без augment. Глобальные метрики агрегировать на уровне исходного изображения, не смешивать patch leakage между subsets. Парные augmentations: flips/rotations для изображения и маски; photometric transforms — только для изображения. Установить настраиваемый batch/patch budget и измерить цель ≤60 секунд/эпоха на доступной RTX 2060 6 GB; дефолты подтвердить реальным прогоном, не считать цель гарантией для другой GPU.

**Альтернативы:** downscale кадра быстрее, но уменьшает колонию до нескольких пикселей; полные кадры в батче не помещаются в 6 GB; on-the-fly patches сохраняют масштаб и ограничивают память.

### D5. CPU full-resolution tiled inference и асинхронный UI

Не ломать существующий синхронный контракт `OnnxModelAlgorithm.detect() -> uint8 mask`. Реализовать tiled inference в совместимом алгоритме либо отдельном интерфейсе с `detect_with_progress`, оставив обычный `detect` для runner. Создавать ONNX Runtime session с `providers=["CPUExecutionProvider"]`; никакой PyTorch dependency в runtime. Для bundled compact model обрабатывать оригинальный кадр тайлами 512×512 с шагом 384 (25% overlap). Кадры меньше окна дополнять до 512; сетка должна обеспечивать дополнительный правый/нижний тайл на границе. Вход: BGR→RGB, resize не применяется, normalization должна совпадать с train. После инференса каждого тайла обрезать padding, накапливать float probabilities и coverage counts, усреднять вероятности перекрытия и бинаризовать единожды с порогом 0.5. Вернуть бинарную маску исходного H×W.

В `AnalysisController` добавить выполнение выбранного алгоритма и расчёт `AnalysisResult` по его маске; в classic режиме оставить текущие настройки и результаты. Геометрия чашки остаётся нужна для определения рабочей площади. В analysis UI добавить selector доступных алгоритмов, перерасчёт, disable конфликтующих действий на время задачи и встроенный status bar «Обработано тайлов i/N». ONNX worker живёт в `QThread`, передаёт progress/result/error сигналами; UI не вызывает синхронный inference. При ошибке worker очистить, показать понятное сообщение и оставить приложение/классический режим доступными.

**Альтернативы:** запуск детекции в Qt main thread замораживает UI; asyncio без executor не уводит нативный inference из GUI; resize всего кадра теряет разрешение; QThread уже используется в приложении и дополнительных GUI зависимостей не требует.

### D6. ONNX Runtime CPU — runtime dependency, PyTorch — только full

Перенести `onnxruntime` в базовые runtime dependencies как CPU package; оставить `torch`, `torchvision` и training libs в `full`. Nuitka `.venv-build` содержит ORT CPU, не содержит PyTorch, но включает `.onnx`. Новый артефакт получает отдельное имя, например `models/colony_mobilenet_v3_small.onnx`; текущий `models/colony_seg.onnx` не менять. Проверить путь к моделям в onefile-сборке: filesystem путь `Path(__file__).../models` может не совпадать с Nuitka extraction directory. Убедиться, что CPU provider и его динамические библиотеки упакованы на Linux и Windows.

**Trade-off:** runtime/binary вырастет на ONNX Runtime CPU и новую модель, зато packaged inference самодостаточен и не требует CUDA/PyTorch.

### D7. Один end-to-end CLI и независимый full-frame CPU benchmark

Добавить training entrypoint в full extra с явными параметрами `--data-root`, `--output`, `--seed`, split proportions, `--epochs`, `--batch-size`, `--patches-per-image`, `--img-size`, `--pretrained/--no-pretrained` и `--cpu-benchmark`. CLI готовит split/import, обучает, выбирает checkpoint по val, оценивает test после заморозки выбора, экспортирует ONNX в output/run directory. Копирование в `models/` требует явного promotion пути и не перезаписывает файл без явного согласия.

CPU benchmark должен запускать экспортированную ONNX через CPU provider на исходных test кадрах, включая preprocessing, все tile calls и stitch; записать warmup, median/p95, исходный размер изображений, CPU model/thread count, ONNX размер. Run report фиксирует hash/signature источника, args, seed/split IDs/counts по классам, train/val/test метрики и per-image metrics, лучшую эпоху, времена эпох, latency и git commit. Не считать latency только на 512×512 dummy или patch.

**Альтернатива:** `train-compare` сейчас сравнивает несколько архитектур и отдельный `TestDataset`; он не готовит воспроизводимый split внутри COCO-набора и не делает full-frame CPU tiled benchmark.

## Risks / Trade-offs

- **Bbox — шумная weak-label, не граница колонии** → Чётко обозначить природу ground truth в отчёте; сравнивать с bbox-derived masks и рассматривать ручную pixel-разметку как будущий quality шаг.
- **369 групп; редкие категории имеют 3–5 фото** → Фиксировать class/group counts, гарантировать непустые splits и предупреждать, что test estimate редких классов статистически слаб.
- **Патчи могут переобучиться на локальные текстуры** → Ограниченный patch budget, spatial/photometric augmentation, pretrained encoder, validation early stopping и per-image отчёт; test не использовать для настройки.
- **Tile overlap может дать швы или края без покрытия** → Усреднять вероятности, тестировать разные размеры, tiny frames и крайние регионы; CPU benchmark использовать на полном кадре.
- **Очень медленный CPU и большие изображения** → Не обещать ≤5 секунд для произвольного устройства; QThread/status bar сохраняют отзывчивость, отчёт публикует measured latency на фактическом CPU.
- **ONNX Runtime усложнит/увеличит Nuitka bundle** → Точечно включать необходимые ORT dynamic libraries, не включать PyTorch, проверить clean build и выполнение модели в onefile на обеих платформах.
- **Epoch latency зависит от IO и GPU contention** → Зафиксировать конфигурацию benchmark, отдельно логировать warmup/epoch time и калибровать patch/batch defaults на RTX 2060.

## Migration Plan

1. Оставить COCO source и текущую модель нетронутыми; все подготовленные данные/checkpoints писать в отдельные output каталоги. Старые манифесты без subset продолжают работать существующим train/val кодом.
2. Добавить ONNX Runtime CPU в runtime/build dependencies, обновить lockfile, пересоздать окружения; проверить, что PyTorch есть только в full environment.
3. Выполнить end-to-end training с фиксированным seed; review split manifest, validation selection, test metrics и full-frame CPU report до promotion.
4. Продвигать новый ONNX под отдельным именем только явной командой/путём; проверить UI и чистую onefile-сборку без PyTorch.
5. Rollback: удалить только новый ONNX артефакт и model selector wiring, вернуть runtime/build dependency changes согласованно; существующие `colony_seg.onnx`, классические алгоритмы и сохранённые training runs не удалять.

## Open Questions

Нет блокирующих вопросов. Batch size/patch budget и CPU thread count должны быть настроены по измерениям; они не меняют согласованные инварианты воспроизводимого split, test isolation и full-resolution asynchronous CPU inference.
