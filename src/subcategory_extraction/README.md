# subcategory_extraction

Шаг 1 пайплайна: по тексту объявления (`description`) выбранный модуль выделяет услуги
и определяет микрокатегории (101–106), к которым они относятся.

- Интерфейс: `SubcategoryExtractionPipeline.run(PipelineInput) -> list[int]`
- Варианты: `google_decision` (Google API, промпты baseline/guided),
  `tfidf` (локальная логистическая регрессия, JSON-артефакт),
  `jev` (нативный Jev через BestAPI), `jev_tfidf` (среднее вероятностей),
  `jev_retrieval` (размеченные соседи), `evidence_jev` (цитаты → Jev),
  `jev_arbiter` (сильный арбитр), `semantic_encoder` (fine-tuned RuBERT-tiny2),
  `semantic_encoder_calibrated` (validation-пороги),
  `semantic_encoder_f1` (выбор checkpoint, окна/attention и дистилляция; CPU),
  `semantic_evidence` (категорийное внимание к токенам + глобальный контекст,
  train-only внешний review, дистилляция и групповой OOF-ансамбль; CPU),
  `jev_boundaries` (точные границы категорий).
- Выбор: ключ `subcategory_extraction` в корневом `config.json`
- `null` — модуль не выбран.
- Результат содержит только дополнительные mcId, исходная категория исключена.

## Четыре проверки

[Общий список и команды](benchmarks/README.md). Новые бенчмарки и папки запрещены.

- [Стоимость токенов](benchmarks/cpu_costs/README.md) — ручная запись.
- [F1 improved_dataset](benchmarks/improved_dataset/README.md) — 119 строк.
- [F1 InnoGlobalHack](benchmarks/classifier_methods/README.md) — 208 строк.
- [Время работы](benchmarks/response_time/README.md) — настоящие вызовы `run()`.

Оба F1-runner по умолчанию проверяют сохранённые ответы без API и обучения.
Для оценки текущего модуля передайте `--methods <имя>`; параметры и пути
к весам берутся из `.env`. Подготовка весов описана в README модулей,
код обучения хранится внутри модуля. Старые эксперименты доступны в истории Git.

[dataset_annotation: время и шесть примеров с mcId](../dataset_annotation/rubert_review/README.md).

## semantic_evidence: фрагменты, учитель и ансамбль

Общий encoder RuBERT-tiny2 обрабатывает весь текст без пропуска хвоста.
Document-вариант усредняет токены; evidence-вариант обучает отдельное внимание
каждой категории к токенам всех перекрывающихся фрагментов и объединяет
свидетельства с глобальным контекстом через learned gate. Это не прежнее
взвешивание больших окон. Третий студент обучается с проверенными
train-предложениями Gemini и soft targets локального ансамбля.

```bash
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false EVIDENCE_TEACHER_TOKEN_BUDGET=1000000 PYTHONPATH=src uv run python src/subcategory_extraction/semantic_evidence/train_plan.py --device auto
```
Дневной лимит выбранного учителя останавливает обучение явно. Продолжить
только review другой доступной моделью, сохранив уже проверенные ответы:

```bash
EVIDENCE_TEACHER_MODEL=gemini-3.1-flash-lite PYTHONPATH=src uv run python src/subcategory_extraction/semantic_evidence/train_plan.py --stage review --resume-review out/semantic-evidence-f1/teacher-review.json
```

Затем повторяется обычная команда `--stage all` (по умолчанию). У каждого
ответа сохраняется реальная модель; старые ответы не приписываются новому
учителю. Невалидные цитаты отклоняются, исходные метки таких строк сохраняются.
По умолчанию для дистилляции требуется завершённая очередь всех96 примеров.
`--allow-partial-review` разрешён только при явном согласии на неполный review;
статус API-ошибки и фактический объём сохраняются в происхождении весов.
В текущем опыте пользователь согласовал44 проверенных примера после
исчерпания квот/недоступности моделей. Снимок реальных ответов сохранён в
`../dataset_annotation/semantic_evidence_reviewed.json`. Повторное обучение
на том же снимке, без новых API-запросов:

```bash
mkdir -p out/reproduce-semantic-evidence
cp src/dataset_annotation/semantic_evidence_reviewed.json out/reproduce-semantic-evidence/teacher-review.json
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONPATH=src uv run python src/subcategory_extraction/semantic_evidence/train_plan.py --output out/reproduce-semantic-evidence --artifact out/reproduce-semantic-evidence/artifacts --allow-partial-review --device auto
```

Для нового API-review интервал запросов по умолчанию12 секунд;
`EVIDENCE_TEACHER_MIN_INTERVAL` позволяет соблюдать более строгую квоту.


Обучение сохраняет прежний control612, уже просмотренный в прошлых опытах.
Development4239 разбивается по близким текстам на три encoder OOF fold.
Каждый encoder обучается фиксированные пять эпох; внешний fold не выбирает
checkpoint. Учитель для дистилляции обучен только на train-части того же fold.
Контрольные и исходные test-метки в review и подбор не передаются.
Индивидуальные пользовательские ответы защищены от машинного исправления.
Предложения Gemini проверяются по полной схеме и дословным цитатам; это
не независимо подтверждённый gold. Исходные файлы датасетов не переписываются.

Пороги и небольшая фиксированная сетка весов композиции выбираются по
development OOF. Это настоящие OOF encoder, но не полностью вложенная
независимая оценка выбора ансамбля. Выбор фиксируется до control и test в
`out/semantic-evidence-f1/frozen-selection.json`. Трудная очередь и usage —
`teacher-review.json`, применения с before/after — `applied-review.json`.
Промежуточные OOF-модели и отдельные композиции остаются в `out/`;
выбранные runtime-веса экспортируются в `semantic_evidence/artifacts/`.

Production загружает только локальные safetensors и проверяет SHA-256
компонентов по `selection.json`; работает на CPU, без Gemini и API-ключей.
Веса encoder распространяются через Git LFS. Путь bundle можно изменить
через `SEMANTIC_EVIDENCE_ARTIFACT_PATH`; исходная категория исключается.
У класса101 нет положительной обучающей поддержки.

Оценка — только существующими сценариями:

```bash
uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --methods semantic_evidence --offline
uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods semantic_evidence --offline
uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module semantic_evidence
```

`--variant <метка>` у F1-runner позволяет сохранить отдельные заранее
зафиксированные варианты одного модуля в общей таблице, не смешивая их
артефакты. Метка не меняет веса; путь bundle задаётся окружением.
Отчёт содержит реальные предсказания, происхождение весов и проверку
пересечений именно с фактическими обучающими строками нового артефакта.

### Измеренные результаты обучения

2026-11-14, рабочая ветка `leo/feature/semantic-evidence-f1`.
Ни один public test не использовался для выбора checkpoint, порогов или весов.
OOF ниже — метрика выбора с групповым crossfit порогов, не независимый
результат после адаптивного review/поиска конфигурации. Control612 уже
просматривался в прошлых опытах и не является новым независимым holdout.

| Вариант | OOF selection,4239 | Reused control,612 |
| --- | ---: | ---: |
| TF-IDF, фиксированный0.5 | 0.883064 | 0.894027 |
| document | 0.914600 | 0.924785 |
| evidence | 0.918569 | 0.925875 |
| distilled | 0.922231 | 0.928293 |
| **ensemble, document0.25 + evidence0.75** | **0.923378** | **0.930258** |

Веса ансамбля не включают TF-IDF или учителя. Дистиллированный одиночный
студент тоже обучен и измерен, но OOF предпочёл композицию двух исходных
студентов. Итоговый `config.json` выбирает `semantic_evidence`.
Пороги выбранного ансамбля102–106:
0.5673034042 /0.7674361467 /0.6714528464 /0.6081880629 /0.5047261342.
Выбор зафиксирован `2026-11-14T16:13:18Z`, до обоих public test.

Протокол, индексы fold, замороженный выбор, все16 композиций, контрольные
результаты, фактический объём review и среда сохранены вместе с весами:
[`semantic_evidence/artifacts/experiment.json`](semantic_evidence/artifacts/experiment.json).
Runtime-композиция и SHA компонентов —
[`selection.json`](semantic_evidence/artifacts/selection.json).
Основной источник содержит4851 описание, но все новые модели/TF-IDF обучены
на одних4239 development-текстах; у distilled30 train-наборов меток изменены
машинными предложениями,10 примеров перевзвешены. Это совмещённый эффект
review и дистилляции, не изолированная абляция одного из них.

Результаты существующих public test приведены раздельно в
[improved_dataset](benchmarks/improved_dataset/README.md) и
[classifier_methods](benchmarks/classifier_methods/README.md).
**Внутренние0.930258 не являются F1 public test или скрытого лидерборда.**

