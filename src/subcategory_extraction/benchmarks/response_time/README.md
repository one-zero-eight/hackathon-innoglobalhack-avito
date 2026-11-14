# Бенчмарк: время работы классификатора
Сервис: subcategory_extraction

## Что проверяем

Последовательные настоящие `run()` на фиксированной выборке из 10
validation-групп официального train (`seed=44`).
Датасет SHA-256 `b7118d68086624fbeab43839f22a635d4e325e4ee339f22d4aa0ad3c2e986615`.
Загрузка модели и чтение данных не входят во время; первого вызова
не исключаем. API-ожидание и повторы входят, если выбран внешний модуль.

## Запуск

```bash
TFIDF_ARTIFACT_PATH=out/classifier-methods/tfidf.json uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module tfidf
```

Выберите другой модуль флагом `--module`; для API нужен действующий ключ,
для локального encoder — веса. Отчёт: `results.json`.

## Сравнение решений

| Сценарий | Версия / дата | Модуль / модель | Промпт | Качество | N | Среднее / медиана / p95, с | Стоимость |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Свежие CPU `run()` | 1c2a6f5 + main 3144889 / 2026-11-14 | semantic_encoder_f1 / distilled RuBERT-tiny2 | — | Здесь не оценивается | 10 | 0.040450 / 0.036370 / 0.081191 | бесплатно |
| Предыдущие CPU `run()` | e814b58 + рабочие изменения / 2026-11-14 | tfidf | — | Здесь не оценивается | 10 | 0.001142 / 0.000819 / 0.002920 | бесплатно |
| Исторический CPU `run()` на 119 диагностических входах | e814b58 / 2026-11-14 | semantic_encoder_f1 | — | Здесь не оценивается | 119 | 0.004776 / 0.003196 / 0.011021 | бесплатно |
| Новый API-замер | та же версия | google_decision | guided | Не запускался | 0 | — | новых расходов нет |

## Выводы

Это время работы конкретного локального запуска, не SLA. Для
интерпретации API-цены см. `../cpu_costs/README.md`.

Исторический замер semantic_encoder_f1 сохранён в `recorded-results.json`
со ссылкой на исходный commit/отчёт. Загрузка исключена; первый вызов включён;
p95 посчитан nearest rank. Его не сравниваем напрямую с выборкой из 10 входов.

## Свежий прогон опубликованных весов Leo

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RUBERT_F1_ARTIFACT_PATH=src/subcategory_extraction/semantic_encoder_f1/artifacts RUBERT_F1_DEVICE=cpu uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module semantic_encoder_f1 --size 10
```

Apple Silicon arm64, CPU, один поток PyTorch; версии и платформа — в `results.json`.
Загрузка модели и данных исключена, первый вызов включён, прогрева и кэша
предсказаний нет. Квота и повторы отсутствуют. p95 — линейный percentile NumPy.
Это другая выборка, чем исторические 119 диагностических входов.

## learned_keywords: новый прогон PR #9

2026-11-14, код `3768d76`, база main `e04c97d`; веса и конфигурация
совпадают с двумя F1-сценариями.

| Модуль / модель | N | Среднее, с | Медиана, с | p95, с | Стоимость API |
| --- | ---: | ---: | ---: | ---: | --- |
| learned_keywords / combined | 10 | 0.001730 | 0.001522 | 0.002868 | бесплатно |

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module learned_keywords --size 10
```

Apple Silicon arm64, CPU; один поток OpenMP/OpenBLAS. Загрузка модели и данных
исключена, первый run() включён; прогрева, кэша предсказаний, квоты и повторов нет.
p95 — линейный percentile NumPy. Выборка: 10 validation-групп официального train
с seed=44; для этой модели это замер задержки, а не оценка на отложенных данных.
Сохранён прежний замер semantic_encoder_f1 с исходным происхождением.

## semantic_evidence: выбранный OOF-ансамбль, 2026-11-14

Реально выполнены 10 последовательных `run()` на прежней фиксированной
выборке seed44, без прогрева и кэша предсказаний. Загрузка моделей, проверка
SHA, чтение данных и генерация черновиков исключены; первый вызов включён.
CPU arm64/macOS, PyTorch2.15.0, один поток. P95 — линейный percentile NumPy.

| Модель | N | Среднее, с | Медиана, с | p95, с | API |
| --- | ---: | ---: | ---: | ---: | --- |
| semantic_evidence: document0.25 + evidence0.75 | 10 | 0.020982 | 0.018053 | 0.046685 | не используется |

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module semantic_evidence --size 10
```

Отпечаток кода `4f0af99e008c`, точные времена и происхождение двух компонентов
сохранены в `results.json`; прежние измерения имеют свои measurement_metadata.
Это локальная задержка, не SLA и не время полного решения с внешней генерацией.

Отдельно выполнен настоящий `src/main.py` с новым default `config.json`:
электромонтажное объявление → один черновик105 с непустыми title/text.
Использован настоящий Gemini, не replay: 1 API-попытка, 0 ошибок.
Вход/выход и usage — `out/semantic-evidence-cli-{input,output}.json`
и `out/semantic-evidence-cli-output.json.costs.json`. Для этой проверки
`SAMPLE_LLM_MAX_ATTEMPTS=1`; модель генератора не подменялась.

