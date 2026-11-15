# Выделение дополнительных микрокатегорий

Вход и выход раздела не изменены:
`SubcategoryExtractionPipeline.run(PipelineInput) -> list[int]`.
Возвращаются дополнительные категории 101–106, исходная категория исключается.

| Группа | Модули |
| --- | --- |
| Лексические — `lexical` | `regex`, `tfidf`, `learned_keywords`, `snippet_logistic_regression` |
| Семантические — `semantic` | `semantic_encoder`, `semantic_encoder_calibrated`, `semantic_encoder_f1`, `semantic_evidence` |
| Decision-модели — `decisions` | `google_decision`, `qwen_decision`, `jev`, `local_jev`, `jev_calibrated`, `jev_boundaries`, `jev_retrieval` |
| Гибриды и каскады — `hybrid` | `jev_tfidf`, `evidence_jev`, `jev_arbiter` |

Выбор в корневом `config.json` — полный путь внутри раздела:

```json
{
  "subcategory_extraction": "decisions.jev",
  "draft_generation": "sample_llm"
}
```

Старые короткие значения не поддерживаются: обновите локальные конфиги.
Каждый модуль хранит `pipeline.py`, обучение и ресурсы рядом с реализацией.
В каждом модуле `decisions` есть `prompts/` и `PROMPTS.md` с анализом вариантов.
TinyBERT исключён из поддерживаемых реализаций.

Заранее подготовленные веса и другие зависимости перечислены в
[ARTIFACTS.md](ARTIFACTS.md). Перенос не меняет веса, пороги и протоколы обучения.
Исходные хеши и пути в исторических метаданных артефактов сохранены как
происхождение обучения; текущие пути загрузки задаются кодом и конфигурацией.

## Общие проверки

[Четыре существующих сценария](benchmarks/README.md) общие для всех групп.
Исторические ключи строк результатов сохранены; новые пути импорта указаны
в `benchmarks/configurations.py`. Сохранённые ответы не изменены.

```bash
uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods lexical.regex
uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --methods lexical.regex
uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module lexical.regex
```

Перенос файлов не является новым замером F1 или времени. Для воспроизведения
существующих конфигураций используйте прежние ключи `--configuration`, например
`--configuration jev::baseline`.
