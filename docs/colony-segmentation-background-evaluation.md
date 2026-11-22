# Эксперимент: сегментация colony на исходном и чёрном фоне

## Краткий вывод

Scratch U-Net улучшила pixel-level IoU и Dice на одном и том же наборе из 54 held-out source ID для обоих вариантов изображения. Для original-background IoU/Dice составили 0.4812/0.6497, для black-background — 0.4721/0.6414. Историческое пересечение baseline test ID с его обучающей выборкой неизвестно; поэтому baseline приведён для парного сравнения и не считается доказанной независимой оценкой.

## Условия эксперимента

- Источник: 22022540, 369 исходных изображений; COCO bbox растеризованы в приблизительные вписанные эллипсы, а не точные контуры колоний.
- Split зафиксирован на уровне source ID: seed 42, train/val/test = 70/15/15; фактически 253/62/54 исходника. Два фоновых представления одного source ID остаются в одной выборке.
- Вариант `black`: чёрным заменялись только пиксели вне найденной окружности чашки; пиксели внутри круга и GT-маска сохранялись.
- Обучение: U-Net со случайной инициализацией, без pretrained и resume checkpoint; CUDA, NVIDIA GeForce RTX 2060 with Max-Q Design; максимум 100 эпох, batch size 2, 4 обучающих patches на изображение, patch 512×512. Лучший checkpoint выбран по validation: epoch 40, val IoU 0.5600.
- Каждая модель проверялась на одних и тех же 54 source ID и GT-масках. Оба checkpoint оценивались одинаковым способом: CPU ONNX, полное разрешение, тайлы 512×512, stride 384, sigmoid для logits, усреднение вероятностей перекрытий, threshold 0.5.
- Табличные метрики агрегированы по пикселям всех 54 test images из суммарных TP/FP/FN/TN; это не среднее по изображениям.

## Качество по фону

| Checkpoint | Фон | IoU | Dice | Precision | Recall | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| Дообучение не выполнялось (baseline) | Original | 0.1857 | 0.3132 | 0.4722 | 0.2343 | 0.9162 |
| Дообучение не выполнялось (baseline) | Black | 0.2032 | 0.3378 | 0.6545 | 0.2276 | 0.9273 |
| Scratch U-Net, best epoch 40 | Original | **0.4812** | **0.6497** | **0.8340** | **0.5322** | 0.9532 |
| Scratch U-Net, best epoch 40 | Black | **0.4721** | **0.6414** | **0.8375** | **0.5197** | 0.9526 |

Относительно baseline, scratch U-Net увеличила IoU на 0.2955 для original и на 0.2689 для black; Dice — на 0.3365 и 0.3036 соответственно. Разница между двумя входными фонами у новой модели составила 0.0091 IoU и 0.0083 Dice.

## Суммарная pixel-level confusion matrix

Порядок в таблице — `TN / FP / FN / TP`; единица измерения — пиксели по всему test-набору.

| Checkpoint | Фон | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|
| Baseline | Original | 322,400,272 | 7,669,698 | 22,428,468 | 6,862,982 |
| Baseline | Black | 326,550,608 | 3,519,362 | 22,624,054 | 6,667,396 |
| Scratch U-Net | Original | 326,968,101 | 3,101,869 | 13,703,952 | 15,587,498 |
| Scratch U-Net | Black | 327,116,858 | 2,953,112 | 14,068,830 | 15,222,620 |

В переносимой копии paired report сохранены метрики и confusion counts по каждому source ID, hashes checkpoint-ов, split ID и provenance: [comparison.json](colony-segmentation-background-comparison.json). Абсолютные локальные пути нормализованы; метрики, source ID и hashes не изменены. Источник набора остался неизменным; fingerprint из отчёта: `9f13e8dd…b0b37f76`.

## Интерпретация и ограничения

1. Новый checkpoint дал более высокий IoU/Dice на обеих версиях фона в сравнении на парном held-out наборе.
2. Baseline training membership неизвестен (`unknown`). Не следует трактовать его test-метрики как независимую оценку обобщения старой модели.
3. COCO bbox → вписанный эллипс — weak label; границы не являются ручной точной сегментацией. Метрики измеряют совпадение именно с этими масками.
4. У baseline и scratch checkpoint в этом сравнении одинаковый tiled CPU inference, хотя предыдущая production-регистрация baseline использовала resize всего кадра. Поэтому baseline-строка здесь — сравнение весов при фиксированном новом inference contract, а не воспроизведение прежнего production-пути.
5. После явного решения о promotion `models/colony_seg.onnx` заменён теми же весами, что и новый checkpoint. SHA-256 production-файла: `5856addc678d4accc61fac028099bd16dfe88dac22686f1364fb9293004b873c`; прежний baseline SHA-256: `b909384a3cdaf7728b0c1d7a6e03d5a807dc98734f0444fdce9d5da20d6a4c5b`.

Post-promotion smoke test на `sp01_img06` (2886×2905, 64 тайла) совпал по TP/FP/FN/TN с per-sample записью в paired report; это проверка соответствия production loader протоколу отчёта, не дополнительный агрегатный benchmark.

## Артефакты

- Production weights: [`models/colony_seg.onnx`](../models/colony_seg.onnx).
- Полный paired report: [comparison.json](colony-segmentation-background-comparison.json).
- Подготовка и обучение воспроизводятся командой `train-colony`; её параметры описаны в [dataset-training-pipeline.md](dataset-training-pipeline.md) и [README.md](../README.md).
