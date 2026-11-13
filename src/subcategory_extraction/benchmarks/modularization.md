# Бенчмарк: модульный перенос Google Decision и TF-IDF

Сервис: subcategory_extraction  
Дата: 2026-11-13   Автор: fih

## Что проверяем и почему

Совпадение предсказаний Google после переноса и новый локальный вариант TF-IDF.
Google проверяется без сети на сохранённых вероятностях: это проверка адаптера,
а не новое измерение качества API или промптов.

```json
{
  "google_cached_samples": [
    "src/subcategory_extraction/benchmarks/historical/2026-11-13_decision-1_seed43.json",
    "src/subcategory_extraction/benchmarks/historical/2026-11-13_decision-1_fresh32.json"
  ],
  "tfidf_dataset": "docs/innoglobalhack_avito/public-train-annotated.jsonl",
  "dataset_sha256": "b7118d68086624fbeab43839f22a635d4e325e4ee339f22d4aa0ad3c2e986615",
  "normalization": "NFKC, casefold, ё→е, whitespace",
  "train_split_seed": 42,
  "validation_holdout_split_seed": 43,
  "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7]
}
```

Запуск из корня:

```sh
PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.replay_google
PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.train_tfidf
uv run python src/subcategory_extraction/benchmarks/historical/benchmark_checks.py
```

TF-IDF: 4 854 уникальных нормализованных описания, train/validation/holdout
3 397 / 728 / 729. Векторизатор и регрессии обучаются только на train,
порог 0.5 выбран на validation. Точные нормализованные дубликаты объединены
до разбиения; близкие шаблоны отдельно не группировались.
JSON-веса и предсказания через `load_pipeline()` сверяются с обученной моделью.
`public-test-dataset.parquet` не использовался.

## Результаты по версиям

| Версия / модуль | Дата | Выборка | micro-F1 | TP / FP / FN |
| --- | --- | --- | ---: | --- |
| 2d5e02b, исторический Google baseline (сохранённый результат) | 2026-11-13 | 32 | 0.677686 | 41 / 38 / 1 |
| Рабочая ветка, google_decision baseline (офлайн-replay) | 2026-11-13 | те же 32 | 0.677686 | 41 / 38 / 1 |
| 2d5e02b, исторический Google guided (сохранённый результат) | 2026-11-13 | 32 | 0.863158 | 41 / 12 / 1 |
| Рабочая ветка, google_decision guided (офлайн-replay) | 2026-11-13 | те же 32 | 0.863158 | 41 / 12 / 1 |
| 2d5e02b, исторический Google fresh32 (сохранённый результат) | 2026-11-13 | свежие 32 | 0.800000 | 54 / 22 / 5 |
| Рабочая ветка, google_decision (офлайн-replay fresh32) | 2026-11-13 | те же свежие 32 | 0.800000 | 54 / 22 / 5 |
| Рабочая ветка, tfidf | 2026-11-13 | validation, 728 | 0.931900 | 1690 / 104 / 143 |
| Рабочая ветка, tfidf | 2026-11-13 | holdout, 729 | 0.925414 | 1675 / 122 / 148 |
| `88712ac` + рабочие правки, tfidf | 2026-11-14 | holdout, 729; повтор обучения | 0.925414 | 1675 / 122 / 148 |

Повтор 2026-11-14 после перехода на два шага: Google replay — baseline
0.677686, guided 0.863158, fresh32 0.800000; 10/10 некорректных ответов
отклонены. Перенесённые исторические проверки — 4/4.

На main `0643454` модуль TF-IDF отсутствует: сравнимого нового прогона нет.
Исторические смешанные датасеты не сравниваем с новым holdout как равные.

## Выводы

Предсказания baseline, guided и fresh32 на сохранённых выборках совпадают с исходными.
10/10 некорректных ответов модели отклонены. Пустой запрос не обращается к API;
отсутствие ключа обнаруживается явно. Payload и учёт токенов проверяются
подменённым HTTP-транспортом, платных запросов нет.
4/4 проверок исторических метрик, выборки, бюджета и маршрутизации проходят.
На `data/input.json` TF-IDF возвращает `[104]`.
Обученные веса и полные отчёты локально в `out/`, в Git не добавляются.
Исходная категория в train всегда 101: обобщение на другие источники и
выделение самой категории 101 не подтверждены.
