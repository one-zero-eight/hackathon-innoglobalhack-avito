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
