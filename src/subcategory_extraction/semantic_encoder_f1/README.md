# RuBERT-tiny2: воспроизводимое обучение и CPU-инференс

Итоговая модель хранится в `artifacts/` рядом с `pipeline.py`:
encoder, multilabel-голова, токенизатор, метаданные и `calibration.json`.
Это путь загрузки по умолчанию. `RUBERT_F1_ARTIFACT_PATH` позволяет явно
указать другой каталог; production inference работает только на CPU.

## Воспроизведение

Из корня репозитория:

```bash
uv sync
HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false uv run python src/subcategory_extraction/semantic_encoder_f1/reproduce_weights.py --train
```

Скрипт повторяет полный протокол Leo: исходные baseline, групповое разбиение,
LR/checkpoint selection, attention-вариант, review-веса, локальный учитель,
дистилляция, пороговый crossfit и проверки seed. Решения принимаются только
по validation; затем выполняется CPU-оценка на control и диагностических
119 объявлениях. Архивные результаты не подменяют свежий F1.

Нужны доступ к Hugging Face для исходных pretrained-моделей, локальное
дисковое пространство и время обучения. На Apple Silicon используется MPS,
при его отсутствии — CPU. Внешние LLM API не вызываются.

Промежуточные модели остаются в `out/`; итоговый выбранный артефакт копируется
в `artifacts/`. Существующие итоговые результаты не перезаписываются.
Полный отчёт: `out/rubert-f1-plan/reproduced-results.json`;
F1, SHA-256 и точные размеры файлов:
`out/rubert-f1-plan/summary-results.json`.

## Публикация

Папка `artifacts/` включается в Git. Encoder в FP32 может превысить лимит
GitHub 100 MiB на файл: такие веса необходимо добавлять через Git LFS.
Перед коммитом установите Git LFS и настройте отслеживание:

```bash
git lfs install
git lfs track "src/subcategory_extraction/semantic_encoder_f1/artifacts/encoder/*.safetensors"
```

Добавьте `.gitattributes` вместе с артефактом. Для скачивания LFS-весов
после клонирования используйте `git lfs pull`.
