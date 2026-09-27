# Внешние датасеты и источники

Каталог сохраняет ссылки на наборы данных и связанные публикации, нужные для
возобновления экспериментов. **Страница статьи, запись набора данных и ссылка на
отдельный файл — разные ресурсы**; ниже они отмечены отдельно. Локальные имена
папок фиксируют состояние этого checkout, но сами по себе не доказывают, что файлы
побайтно совпадают с опубликованным архивом.

Метаданные и ссылки проверены **2026-09-27**. Перед повторным использованием сверяйте
актуальные условия доступа и лицензии на странице источника и указывайте в публикации
версию датасета и дату получения.

## AGAR

- **Что это:** Annotated Germs for Automated Recognition — датасет изображений
  бактериальных колоний. Сайт описывает 18 000 фотографий пяти микроорганизмов и
  336 442 размеченные колонии.
- **Официальная страница:** [AGAR dataset / NeuroSYS](https://agar.neurosys.com/).
- **Демоархив:** [AGAR_demo.zip](https://api.data.neurosys.com:4443/agar-public/AGAR_demo.zip)
  — доступен по прямой ссылке со страницы AGAR.
- **Полный набор:** сайт предлагает запросить доступ через форму. На сайте указано
  использование для академических исследований по лицензии
  [CC BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/).
- **Публикация для цитирования:** S. Majchrowska et al., “AGAR a microbial colony
  dataset for deep learning detection” (2021), [arXiv:2108.01234](https://arxiv.org/abs/2108.01234).
  Сайт AGAR приводит эту ссылку и BibTeX-цитату.
- **Связь с локальными файлами:** в checkout есть `datasets/AGAR_demo/` (40 JPG и
  40 JSON по текущему файловому подсчёту). Контрольная сумма архива и точное
  соответствие локальной папки текущей загрузке официального демо не проверялись.
  Полный AGAR в этом каталоге не обнаружен.

## Petri plates — *E. coli*, *S. aureus*, *P. aeruginosa*

- **Датасет, версия 2:** [Figshare, версия 2, файл 35973995](https://figshare.com/articles/dataset/Dataset_bioengineering_17489364/20109377/2?file=35973995).
  Для устойчивого цитирования используйте DOI версии
  [10.6084/m9.figshare.20109377.v2](https://doi.org/10.6084/m9.figshare.20109377.v2).
- **Описание записи:** Pedro Miguel Rodrigues, Jorge Luís, Freni Kekhasharú Tavaria,
  “Petri dishes digital images dataset of e.coli, s.aureus and p.aeruginosa.” (2022).
  В описании Figshare указано около 1 150 размеченных изображений трёх видов;
  DataCite указывает размер записи 2 261 518 909 bytes и лицензию
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Связанная статья:** P. M. Rodrigues, J. Luís, F. K. Tavaria, “Image Analysis
  Semi-Automatic System for Colony-Forming-Unit Counting,” *Bioengineering* 9(7),
  271 (2022), [DOI 10.3390/bioengineering9070271](https://doi.org/10.3390/bioengineering9070271).
  Это статья, а не сама запись данных; статья также опубликована по CC BY 4.0.
- **Связь с локальными файлами:** локальная папка `datasets/Petri_plates/` содержит
  1 252 JPG. Сопоставление этой папки с Figshare — рабочая подсказка из проекта, а
  не подтверждённая побайтовая проверка: опубликованное описание говорит «около
  1 150» изображений и приводит пример имени `CEMTimages_0278_EC_C76`, тогда как
  локальные файлы имеют имена вида `IMG_7708ecoli_T4_10^-5__300.JPG`. До сверки
  манифеста, состава и/или контрольных сумм не считайте эти копии идентичными.

## Annotated dataset for deep-learning-based bacterial colony detection (22022540)

- **Датасет, версия 3:** [Figshare, версия 3](https://figshare.com/articles/dataset/Annotated_dataset_for_deep-learning-based_bacterial_colony_detection/22022540/3);
  постоянный DOI версии —
  [10.6084/m9.figshare.22022540.v3](https://doi.org/10.6084/m9.figshare.22022540.v3).
- **Состав по статье:** 369 цифровых изображений культур 24 видов бактерий и
  56 865 аннотаций колоний в формате bounding boxes. Запись Figshare также содержит
  metadata и файлы аннотаций COCO, TSV/CSV, VOC и YOLO.
- **Лицензия данных:** Figshare/DataCite указывает
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); DataCite указывает
  размер записи 641 992 356 bytes.
- **Статья-описание:** L. Makrai et al., “Annotated dataset for deep-learning-based
  bacterial colony detection,” *Scientific Data* 10, Article 497 (2023),
  [DOI 10.1038/s41597-023-02404-8](https://doi.org/10.1038/s41597-023-02404-8).
  Это статья в открытом доступе, а не отдельный архив данных. В её разделе
  Code availability указано, что `bbox_placement_test.py` и `TSV_to_COCO.py`
  размещены вместе с датасетом на Figshare.
- **Связь с локальными файлами:** исходные файлы проекта находятся в
  `datasets/22022540/`; по текущей локальной инвентаризации там 369 JPG и файлы
  разметки/метаданных. Импортированное представление находится в
  `datasets/22022540_imported/`: `dataset.json` описывает 369 `source` и 369
  производных `cropped` записей. Это производный результат импорта, а не исходный
  Figshare-архив; importer растрирует bbox в приблизительные эллиптические маски
  (weak labels). Контрольные суммы локальных файлов с Figshare не сверялись.
- **Повторный импорт:** полная инструкция и ограничения описаны в
  [dataset-training-pipeline.md](dataset-training-pipeline.md). Для нового выхода
  используйте отдельную папку, чтобы не перезаписывать нужную локальную копию:

  ```bash
  uv run import-22022540 \
    --data-root datasets/22022540 \
    --output /path/to/22022540_imported \
    --crop --verify
  ```

## NIST's Integrated Colony Enumerator (NICE)

- **Что это:** запись NIST о программном обеспечении для подсчёта колоний, не
  отдельный опубликованный исследовательский корпус снимков.
- **Запись NIST:** [NIST Public Data Resource, NICE 1.1.0](https://data.nist.gov/od/id/89811452D29952B5E05324570681C0AF2073),
  [DOI 10.18434/M32073](https://doi.org/10.18434/M32073). В записи указаны
  `NICE_Manual_April_2020.zip`, `NICE_1.4_Essential.zip` и `Demo Images.zip`;
  исходный код доступен в репозитории [usnistgov/NICE-Public](https://github.com/usnistgov/NICE-Public).
- **Цитирование:** Jeeseong C. Hwang (2018), *NIST's Integrated Colony Enumerator*,
  Version 1.1.0, National Institute of Standards and Technology, DOI выше; запись
  просит указать дату доступа.
- **Условия:** NIST ссылается на свои [copyright, fair use and licensing
  statements](https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software).
  Не приписывайте этим файлам лицензию Creative Commons без отдельного подтверждения.
- **Связь с локальными файлами:** отдельная локальная папка NICE в перечисленных
  наборах проекта не обнаружена. Демоархив следует считать примером для NICE, а не
  подтверждённым источником локальных датасетов AGAR, Petri plates или 22022540.

## Как возобновить эксперимент

1. Скачивайте нужный набор с его первичной страницы или DOI; для версионированных
   Figshare-наборов фиксируйте DOI версии (`.v2`, `.v3`), а не только ссылку на статью.
2. Сохраняйте дату получения, список файлов и контрольные суммы рядом с локальной
   копией или в журнале эксперимента. В репозитории эти проверки не выполнялись,
   поэтому приведённые пути не являются доказательством идентичности архивам.
3. Для 22022540 цитируйте и статью *Scientific Data*, и Figshare dataset DOI;
   отдельно описывайте импорт и преобразование bbox в маски. Для Petri plates
   цитируйте Figshare и связанную статью *Bioengineering*.
4. Перед повторным использованием проверьте условия доступа и лицензии на первичных
   страницах. AGAR full dataset требует запроса доступа; NIST NICE не следует
   путать с отдельным открытым корпусом фотографий.
