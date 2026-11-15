# Логистическая регрессия на окнах вокруг словоформ

Модуль `lexical.morphology_logistic_regression` можно загрузить через `load_pipeline`
или указать как вариант `subcategory_extraction` в конфиге. Артефакт
`artifacts/<имя модели>.json.gz` содержит индекс словоформ, словарь признаков,
коэффициенты и пороги. Морфологический анализатор при инференсе не нужен.

Точное совпадение нормализованного токена с индексом открывает окно до 11
слов в пределах предложения. Из объединения окон извлекаются бинарные
символьные, словесные и парные признаки. Категории оцениваются пятью
отдельными регрессиями.

## Воспроизведение

Сохраните таблицы PostgreSQL `case_train`, `case_test` и `manual_test` через
`database.read_records` как одноимённые JSONL в порядке `row_id`.
Используйте [индекс словоформ](dictionary/word-index.json) из модуля.
Его [леммы](dictionary/lemmas.json) выбраны по 92 описаниям с Яндекса и Юлы.
Для диагностики на объединённом наборе нужен локальный снимок
`docs/fih/igh_deduped_no_conflict.jsonl`. Метаданные весов содержат хеши
входных снимков и индекса.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python -m \
  subcategory_extraction.lexical.morphology_logistic_regression.train \
  out/morph-logreg-reproduction \
  src/subcategory_extraction/lexical/morphology_logistic_regression/dictionary/word-index.json \
  docs/fih/igh_deduped_no_conflict.jsonl --artifact-dir out/morph-logreg-reproduction/artifacts
```

Перед запуском положите снимки БД в `out/morph-logreg-reproduction/`.
Все варианты одного описания группируются по символьному TF–IDF
(косинус 0,85). Группы, связанные с `case_test` и `manual_test`,
исключаются из обучения и калибровки. Отчёт `out/morph-logreg-reproduction/reports/avito-case-train.json`
показывает весь локальный набор и подвыборку, чьи группы не встречались
в fit или tune. Хеши входов и веса зафиксированы в `artifacts/avito-case-train.metadata.json`.

## Варианты и воспроизведение переноса

По умолчанию загружается `avito-case-train`. Для варианта на Яндексе и Юле
установите `MORPH_LOGREG_MODEL=yandex-youla-decision1` при запуске пайплайна.

Для обучения второго варианта используйте размеченные локальные снимки
Decision 1. Группы провайдеров, связанные с целевым Avito, исключаются
до разбиения на обучение и калибровку. Строки без разметки пропускаются.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python -m \
  subcategory_extraction.lexical.morphology_logistic_regression.train \
  out/morph-logreg-reproduction \
  src/subcategory_extraction/lexical/morphology_logistic_regression/dictionary/word-index.json \
  docs/fih/igh_deduped_no_conflict.jsonl --artifact-dir out/morph-logreg-reproduction/artifacts \
  --provider-sources docs/provider_collections/yandex-1492-decision-1.jsonl \
    docs/provider_collections/youla-1469-decision-1.jsonl
```

Файлы весов и метаданных имеют одинаковый префикс. Поставляемые веса
лежат в `artifacts/`. Построчные отчёты тренера пишутся в `reports/` внутри
каталога снимков.
Команда обучения останавливается, если имя уже занято.

## Записка о сохранённых моделях

Оба артефакта используют один индекс из 7 008 форм. Веса, словарь признаков
и пороги обучены отдельно.

| Имя артефакта | Источник | Fit / tune | Выборка | N | Micro-F1 |
| --- | --- | ---: | --- | ---: | ---: |
| avito-case-train | PostgreSQL `case_train` | 7 128 / 1 331 | PostgreSQL `manual_test` | 139 | 0.723849 |
| avito-case-train | PostgreSQL `case_train` | 7 128 / 1 331 | Локальный дедуп, группы вне fit/tune | 660 | 0.925690 |
| avito-case-train | PostgreSQL `case_train` | 7 128 / 1 331 | Локальный дедуп целиком | 4 966 | 0.983556 |
| yandex-youla-decision1 | Локальные Яндекс и Юла, Decision 1 | 2 516 / 439 | Локальный дедуп целиком | 4 966 | 0.926953 |
| yandex-youla-decision1 | Локальные Яндекс и Юла, Decision 1 | 2 516 / 439 | Те же 660 строк вне fit/tune Avito | 660 | 0.907516 |

В полном локальном дедупе для `avito-case-train` 4 306 строк относятся к
группам, использованным при обучении или калибровке. Для
`yandex-youla-decision1` исключены пять размеченных строк провайдеров из групп
целевой выборки и одна строка Юлы без меток. Провайдерскому варианту
`manual_test` в этой серии не запускали.
