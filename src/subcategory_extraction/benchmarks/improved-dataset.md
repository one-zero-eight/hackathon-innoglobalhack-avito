# Evaluation на исправленном InnoGlobalHack

Сервис: `subcategory_extraction`. Дата: 2026-11-14.  
Ветка: `leo/feature/improved-dataset-evaluation`.

## Данные и ограничения

Использован [коммит e3d0e0e36aeddef637c6bc63cd8db4165f24005f](https://github.com/one-zero-eight/hackathon-innoglobalhack-avito/commit/e3d0e0e36aeddef637c6bc63cd8db4165f24005f), а не прежний train или смешанный датасет.
В `docs/improved_dataset/` перенесены основные файлы; train, test, manifest и policy побайтно соответствуют этому коммиту.

- Train: **4851** уникальное нормализованное описание, включая 778 пустых наборов меток.
- Test: **119** уникальных описаний, 209 положительных назначений категорий, 44 пустых набора.
- Все 119 описаний дословно присутствовали в исходном public test; у 55 изменены метки относительно хотя бы одной исходной копии.
- Вход моделей — только `description`, без `title`. Исходный `mcId` train — 101; для test поле отсутствует, поэтому **101 — явное допущение**, а не восстановленные данные.
- Сохранено готовое разделение из коммита. Новых внутренних разбиений, validation, поиска параметров или порогов нет. Все 4851 train-строка использованы обеими обучаемыми моделями.
- Нормализованных точных пересечений train/test — **0**. Дополнительный поиск ближайшего train по символьному TF-IDF дал максимум cosine **0.892593**, пар с cosine ≥0.92 — **0**. Метод: совместный unsupervised `char_wb`, ngrams `(3,5)`, `min_df=1`, `max_features=60000`, `sublinear_tf=True`. Это проверка известных видов сходства, не гарантия отсутствия всех шаблонных утечек.

**Это не независимо подтверждённый ручной эталон.** Manifest помечает данные как `provisional_annotation_not_independently_verified`, а test — как `reviewed_diagnostic_subset_not_independent_holdout`. Автор сообщает автоматический пересчёт и 1574 группы с диагностическими сомнениями во всём исходном корпусе. Оценка измеряет соответствие этой версии разметки; независимое обобщение на скрытый test не доказано.

SHA-256:

```text
train.jsonl:             d365141906cd7e2a1d66792aa2e37661d8bcdcd0a4bbaf3781649a9108238b76
reviewed_test.parquet:   0af183508c1501425a1330675ceb8923dfe130fff3f86e418ebe3ec4baba34a2
actual training JSONL:  d152c7c5e80a53ddd6258a1cee6f61bf797cdaef54348857b2ccda2be0f26e70
```

Фактический training JSONL сериализован общим trainer без title/item_id; порядок, полные описания и множества меток всех 4851 строк сохранены. Его SHA совпадает в метаданных TF-IDF и RuBERT.

## Результаты: один test, пять методов

Метрика — общий `run.calculate_metrics`: sklearn micro-F1 по дополнительным `mcId`, `zero_division=0`. Баллы классификации — `60 × F1`, **не полный результат из 100 баллов**. Генерация черновиков не оценивалась.

| Метод | Источник предсказаний | micro-F1 | Баллы / 60 | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Google Decisions guided | Реальные сохранённые API-ответы, пересчёт по новым меткам | 0.801061 | 48.063660 | 151 | 17 | 58 |
| Jev guided | Реальные ответы native-кэша | 0.812665 | 48.759894 | 154 | 16 | 55 |
| TF-IDF + LogisticRegression | Заново обучен на исправленном train | 0.762332 | 45.739910 | 170 | 67 | 39 |
| **Jev + TF-IDF, среднее** | Те же ответы Jev + переобученный TF-IDF | **0.854962** | **51.297710** | **168** | **16** | **41** |
| RuBERT-tiny2 | Заново дообучены encoder и multilabel-голова; inference CPU | 0.811530 | 48.691796 | 183 | 59 | 26 |

Дополнительная диагностика важна из-за большого числа отрицательных объявлений; micro-F1 не учитывает true negatives.

| Метод | Полностью верный набор / 119 | Непустой ответ на отрицательном объявлении / 44 |
| --- | ---: | ---: |
| Google guided | 76 | 4 |
| Jev guided | 80 | 3 |
| TF-IDF | 53 | 29 |
| Jev + TF-IDF | 89 | 3 |
| RuBERT-tiny2 | 65 | 21 |

Лучший измеренный вариант здесь — **Jev + TF-IDF**. Он сохраняет FP Jev и уменьшает FN. RuBERT имеет полноту 0.875598, но precision 0.756198: частые ложноположительные ответы мешают превзойти Jev и ансамбль. Для ансамбля precision 0.913043 и recall 0.803828.

Старые 0.839394 / 0.846154 / 0.778462 / 0.842262 / 0.786704 из [исторического отчёта](classifier-methods.md) относятся к другим 208 строкам/меткам и старым обученным весам. Их разность с этой таблицей нельзя трактовать как чистое улучшение или регрессию модели.

## Что именно обучено и что переиспользовано

**TF-IDF:** word ngrams `(1,2)`, `min_df=2`, `max_features=30000`, `sublinear_tf=True`; пять независимых `LogisticRegression(C=4, max_iter=500, random_state=42)`. Порог 0.5. Новый `--prepared-dataset` сохраняет весь уже подготовленный train без дополнительного удаления компонент при cosine 0.85. Старое поведение trainer по умолчанию не изменено. Обучение заняло **1.251 с**.

**RuBERT:** `cointegrated/rubert-tiny2`, фиксированная ревизия `e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae`. Новое дообучение исходного pretrained checkpoint, не продолжение старых task-весов: все 29 193 768 параметров encoder и 1878 параметров головы обучались на MPS. AdamW `lr=3e-5`, `weight_decay=0.01`, seed42, batch16 документов, BCEWithLogitsLoss, ровно три эпохи, последний checkpoint. Начало/середина/конец, до трёх окон по 512 токенов; между окнами очень длинного текста возможны пропуски. Положительных train-примеров класса101 нет.

| Эпоха RuBERT | Train BCE | Время, с |
| --- | ---: | ---: |
| 1 | 0.362617 | 51.289 |
| 2 | 0.216042 | 51.001 |
| 3 | 0.169416 | 50.778 |

Полная команда обучения RuBERT заняла **156.018 с**, включая загрузку и сохранение. Ни checkpoint, ни параметры после просмотра test не менялись.

**Google/Jev:** не переобучались; guided-промпт сохранён прежним, SHA-256 `f05534059aa8ab3e29a281d39d65bfc3e5cbc887330787aba267abd8891728b7`. Google — `gemini-decision-1`, Jev — `jev-1.13.0`. Фиксированный порог 0.5. У ансамбля фиксированные веса 0.5/0.5 и порог 0.5. Промпт не подстраивался под новую политику разметки.

Google replay проверяет хеш исходного датасета записи, модель, промпт, порог, исходную категорию, порядок и хеши всех записанных описаний, полный набор и диапазон вероятностей. Новые строки сопоставляются по точному SHA описания и source mcId; новые эталоны не участвуют в выборе ответов. Отсутствующий вход или противоречивые вероятности копий — ошибка. Jev использует существующий точный native-кэш, общий для baseline и ансамбля.

## Инференс и расходы

CPU arm64, Darwin25.5.0, Python3.13.5, PyTorch2.15.0, Transformers4.57.6, sklearn1.9.1. Один worker; по одному настоящему `run()` на каждый из 119 входов. Ниже локальные модели, без загрузки пайплайна, но с первым inference:

| Модель | Среднее, мс | Медиана, мс | P95 nearest rank, мс | Загрузка, с |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF | 0.354 | 0.282 | 0.514 | 0.050 |
| RuBERT CPU | 5.888 | 3.535 | 14.539 | 1.274 |

Новых API-попыток во всех пяти сценариях — **0**, внешние API-расходы этого запуска — **$0**. У Google/Jev/ансамбля по 119 replay/cache hits. Время чтения кэша не является временем ответа внешней модели; fresh latency здесь не измерялась. Цена первоначального получения ответов и cold-run без известного тарифа — **неизвестна**, а не бесплатно. Локальные модели не требуют API-платежей; электричество и оборудование в эту стоимость не входят.

## Воспроизведение из корня `hackathon-avito-latest`

```bash
uv sync
PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.train_jev_hybrid --dataset docs/improved_dataset/train.jsonl --exclude-dataset docs/improved_dataset/reviewed_test.parquet --prepared-dataset --artifact out/improved-dataset/tfidf.json --report out/improved-dataset/tfidf-training.json
HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.train_semantic_encoder --dataset out/improved-dataset/hybrid-training.jsonl --artifact out/improved-dataset/semantic-encoder --device auto
TFIDF_ARTIFACT_PATH=out/improved-dataset/tfidf.json SEMANTIC_ENCODER_ARTIFACT_PATH=out/improved-dataset/semantic-encoder SEMANTIC_ENCODER_DEVICE=cpu TOKENIZERS_PARALLELISM=false PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.compare_methods --dataset docs/improved_dataset/reviewed_test.parquet --training-dataset docs/improved_dataset/train.jsonl --methods google_decision jev tfidf jev_tfidf semantic_encoder --workers 1 --report out/improved-dataset/reproduced-methods.json
```

Для идентичного offline-повтора Jev нужен локальный `out/jev-cache` с настоящими ответами. При отсутствии совпавших записей клиент обращается к API и требует действующий ключ; фиктивного offline fallback нет. Google record хранится в `results/google-guided-public.json`. Веса и Jev-кэш в Git не включены.

Новые веса: `out/improved-dataset/tfidf.json` и `out/improved-dataset/semantic-encoder/`. Старые артефакты и `config.json` сохранены: смена default по диагностическому test не выполнялась. Для использования новых весов явно задавайте соответствующие переменные `*_ARTIFACT_PATH`.

## Проверка и сохранённые доказательства

- Реально выполнены оба обучения и inference всех пяти методов на новых119 эталонах.
- При публикации совпали тестовые строки и протоколы всех запусков; для каждого метода F1 и TP/FP/FN пересчитаны из сохранённых per-row предсказаний двумя реализациями метрики. Все пять статусов — `completed`.
- Проверены SHA из manifest и сохранность всех4851 обучающих описаний/наборов меток между исходником и сериализованным train; оба артефакта ссылаются на один фактический train.
- Дополнительно выполнены in-memory проверки replay: исходные208 строк, переупорядоченная/переразмеченная подвыборка, отказ при изменённом/отсутствующем входе, неверном промпте/хеше/порядке/вероятностях и противоречивых копиях. Отметка времени отчёта вынесена из сравниваемого протокола, чтобы раздельные запуски оставались совместимы с существующим publisher.

[Сводный JSON](results/improved-dataset.json) содержит per-row эталоны и предсказания, метрики по категориям, снимки source hashes каждого запуска, API/cache-счётчики, расходы, training reports и аудит данных. Исходные отчёты отдельных запусков и логи: `out/improved-dataset/{api-methods,local-hybrid-methods,encoder-methods,tfidf-training,training-verification,dataset-audit}.json`, `out/improved-dataset/semantic-encoder/train-report.json`.
