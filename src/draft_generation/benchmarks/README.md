# Бенчмарки генерации

Формат актуального main: `benchmarks/<name>/run.py`, `README.md` и JSON.

| Сценарий | Текущий статус |
| --- | --- |
| [sample_llm](sample_llm/README.md) | Офлайн-контракт и реальное сравнение подтверждённости фактов / сохранения деталей (`compare.py`) |
| [full_pipeline_evaluation](full_pipeline_evaluation/README.md) | Оценщик/Parquet/аудит проверены; новый live не запускался |
| [response_time](response_time/README.md) | API отключён, свежая latency отсутствует |

Исторические live-ответы и времена сохранены отдельно и не выдаются
за результаты текущей версии или за новые модельные вызовы.
