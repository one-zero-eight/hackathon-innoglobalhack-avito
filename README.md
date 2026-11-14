# Услуги из объявлений Авито → черновики

По тексту объявления и его микрокатегории находим услуги из других
микрокатегорий «Ремонт и отделка» (101–106) и генерируем черновики объявлений.

## Структура

| Путь | Что это |
| --- | --- |
| `src/main.py` | CLI + всё общее: контракт, микрокатегории, `BasePipeline`, `LLMClient` |
| `config.json` | выбор реализации каждого шага |
| `src/subcategory_extraction/` | шаг 1: выделение подкатегорий услуг |
| `src/draft_generation/` | шаг 2: генерация черновиков |
| `data/` | пример входа |
| `docs/` | кейс, критерии, справочник, датасеты |
| `backlog/` | локальный бэклог (в `.gitignore`) |

В каждом разделе — несколько вариантов: `<module>/pipeline.py` и, при
необходимости, `<module>/prompts/`. Общие `benchmarks/` (`.md` по файлу на
бенчмарк) остаются в корне раздела. Экспериментальные решения тоже оформляем
как модули, без отдельной папки `experiments/`.

## Выбор реализации

Корневой `config.json` задаёт имя папки модуля для каждого шага:

```json
{
  "subcategory_extraction": "semantic_encoder_f1",
  "draft_generation": "sample_llm"
}
```

Доступны `subcategory_extraction/google_decision`, `tfidf`, `jev`, `jev_tfidf`,
`jev_retrieval`, `evidence_jev`, `jev_arbiter`, `semantic_encoder`,
`semantic_encoder_calibrated`, `semantic_encoder_f1`, `jev_boundaries`.
Доступен генератор `draft_generation/sample_llm`. `null` означает, что модуль
не выбран. CLI выдаёт явную ошибку.
Варианты (например, `jev`, `embeddings` или `llm`) добавляются в
`src/<service>/<module>/`; затем в конфиге указывается имя папки вместо `null`.
Каждый вариант содержит `__init__.py` и `pipeline.py` с классом и интерфейсом
своего раздела (см. [AGENTS.md](AGENTS.md)). Переключение — только через конфиг,
без изменения оркестрации. Несуществующий модуль вызывает ошибку.

Для сравнения вариантов используем общие бенчмарки раздела и фиксируем
модуль в результатах. Скрипты переноса и обучения описаны в
[subcategory_extraction/README.md](src/subcategory_extraction/README.md);
исторические результаты с внешним SplitShift сохранены в
`src/subcategory_extraction/benchmarks/historical/`.

### Актуальная оценка на исправленном датасете

RuBERT-tiny2 и TF-IDF заново обучены на `improved_dataset` из коммита
`e3d0e0e`; Google guided, Jev guided, TF-IDF, Jev + TF-IDF и RuBERT
сравниваются на одних 119 диагностических test-описаниях.
[Результаты, ограничения разметки и команды](src/subcategory_extraction/benchmarks/improved-dataset.md).
Прежние метрики на исходных 208 строках ниже и в старых отчётах —
исторические, с новой оценкой напрямую не сопоставимы.

### Подбор порогов RuBERT на CPU

`semantic_encoder_calibrated` использует тот же RuBERT-tiny2 с порогами,
выбранными только на внутренней validation. Для этого отдельные веса
обучаются на train-компонентах; близкие тексты не переходят между
train/validation/control. Диагностический public test в подбор не входит.
Прежние веса и default `config.json` не меняются.
[Протокол, результаты и запуск](src/subcategory_extraction/benchmarks/rubert-thresholds.md).
В этом опыте подобранные пороги не улучшили micro-F1 на control;
экспериментальный вариант не включён по умолчанию.

### Полный план улучшения RuBERT-tiny2

`semantic_encoder_f1` добавляет выбор checkpoint/LR и warmup, проверенные
трудные train-примеры, label-wise attention окон, дистилляцию локального
RuBERT-base и group-crossfit калибровку порогов. Итоговый классификатор —
один компактный RuBERT-tiny2 на CPU, без учителя и внешнего API при inference.
Варианты сравниваются на одних119 диагностических описаниях; это локальная
оценка предварительных меток, не прогноз скрытого лидерборда.
[Все результаты, ограничения и воспроизведение](src/subcategory_extraction/benchmarks/rubert-f1-plan.md).
На тех же119 описаниях выбранный по validation студент получил micro-F1
**0.857143** против0.811530 у прежнего RuBERT. В этой ветке он выбран в
`config.json`. Готовые веса, токенизатор и пороги опубликованы в
`out/rubert-f1-plan/selected`; encoder хранится в Git LFS. Переобучение для
запуска не требуется. После установки Git LFS выполните:

```bash
git lfs install
git lfs pull
uv sync
```

Результат на скрытом test неизвестен; устойчивость улучшения по разным seed
не подтверждена.

### Jev + локальный TF-IDF без validation

`jev` использует нативный System One API BestAPI и существующий guided-промпт.
`jev_tfidf` усредняет вероятности Jev и обученных локальных голов с весами
0.5/0.5 и порогом 0.5. Для категории без обученной локальной головы остаётся
решение Jev. Это заранее фиксированные параметры, не подбор по public test.

В `.env` нужны `BESTAPI_API_KEY` для классификатора и `LLM_API_KEY` для
существующего генератора Gemini. При наличии `GOOGLE_API_KEY` допустима запись
`LLM_API_KEY=${GOOGLE_API_KEY}`. Ключи не коммитим.

```bash
PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.train_jev_hybrid
PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.compare_jev_hybrid
uv run python src/main.py --input data/input.json --output out/jev-hybrid-live.json
```

Новый trainer использует официальный train без validation/holdout и без поиска
порогов. Из обучения исключаются компоненты точных и близких к public test
описаний, только по текстам. Оценочные строки и метки не удаляются.
Артефакт — `out/tfidf.json`, фактический train — `out/hybrid-training.jsonl`.
Общий бенчмарк сравнивает три классификатора на одинаковых 208 строках
через конкурсную `calculate_metrics`; генерация и остальные баллы в эту
классификационную оценку не входят. Фактические ответы Jev кэшируются в
`out/jev-cache`, чтобы гибрид использовал те же ответы, что и baseline.


Результаты, инвентарь промптов и ограничения:
[бенчмарк Jev/TF-IDF](src/subcategory_extraction/benchmarks/jev-tfidf.md).

## Пять новых методов и общий отчёт

Реализованы RAG с размеченными train-примерами, извлечение цитат перед Jev,
сильный арбитр спорных решений, реально дообучаемый семантический encoder
и отдельный промпт границ микрокатегорий. Все выбираются через обычный
`config.json`; код моделей и промпты находятся в `src/subcategory_extraction/`.

**Полное API-сравнение не завершено:** предоставленный BestAPI-ключ отклонён
с HTTP 401. В отчёте четыре метода имеют явный статус ошибки без выдуманного F1.
Encoder обучен на MPS и измерен; базовые варианты посчитаны на тех же 208 строках.
Порог 0.5 и остальные настройки фиксированы до запуска, validation не создана.

[Описание каждого метода, таблица результатов и воспроизведение](src/subcategory_extraction/benchmarks/classifier-methods.md).
[Исходные предсказания и статусы JSON](src/subcategory_extraction/benchmarks/results/classifier-methods.json).

```bash
uv sync
PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.train_jev_hybrid
PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.train_semantic_encoder --dataset out/hybrid-training.jsonl --device auto
PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.compare_methods --methods google_decision tfidf semantic_encoder
```

Последняя команда не требует BestAPI после обучения локальных моделей:
Google здесь — проверяемый replay реальных ранее полученных ответов на этом
наборе. Для нового API-инференса нужен рабочий ключ; полные кэши и веса
в Git не добавляются.

## Пайплайн

```
description + mcId
   │
   ▼
subcategory_extraction    список mcId найденных услуг (101–106)
   │
   ▼
draft_generation          черновик на каждую целевую категорию
   │
   ▼
{"drafts": [{"mcId": 104, "title": "...", "text": "..."}]}
```

Запуск:

```bash
uv sync
cp .env.example .env          # заполнить LLM_API_KEY

uv run python src/main.py --input data/input.json --output out/output.json
```

Для полного запуска выберите реализации обоих шагов и настройте `.env`
(для генератора см. `src/draft_generation/README.md`).
Общий `LLMClient` — абстрактный интерфейс для конкретных клиентов.

## Оценка полного решения

```bash
uv sync
uv run python src/run.py --config run.config.json
```

`run.config.json` задаёт датасет (`.parquet` или официальный `.jsonl`),
путь к конфигу модулей, исходную категорию (101 для официальных данных),
путь отчёта, необязательный лимит строк и частоту логирования.
Все относительные пути считаются от файла конфигурации.
`pipeline_config` ссылается на обычный `config.json`: необходимо выбрать
оба рабочих модуля, включая генератор. При `null` оценка завершается
ошибкой, без подмены полного решения классификатором. Параметры моделей
и секреты выбранных модулей по-прежнему берутся из `.env`.

Метрика — `sklearn.metrics.f1_score(average="micro", zero_division=0)`
по множествам `mcId` из **итоговых drafts**, баллы за классификацию —
`60 × micro-F1` (не итоговый балл за все критерии хакатона).
Повторы не учитываются, ошибочный вывод исходной категории считается FP.
Если все эталоны и ответы пусты, micro-F1 равен 0.
Отчёт `out/evaluation.json` содержит метрику, TP/FP/FN, конфигурацию,
время, эталоны и полные ответы для каждой строки. Ошибка обработки
не превращается в пустой ответ: прогон прекращается без новой метрики.

`training_dataset` включает проверку нормализованных точных совпадений
описаний с train; пересечения отражаются в отчёте и предупреждении,
но не удаляются из оценки. Это не поиск всех близких дубликатов
и не гарантия отсутствия утечки. `null` отключает проверку.
Официальный test следует использовать для итоговой оценки после
фиксации модели и порогов, не для их подбора.

## Контрибьюшн

1. Ветка на каждую фичу: `<username>/feature/<slug>`. В `main` — только через PR.
2. Бэклог локальный: `backlog/{TODO,PROCESS,REVIEW,DONE}`,
   задача перемещается по папкам по мере работы.
3. Новый бенчмарк — новый `.md` в `src/<service>/benchmarks/`. Перед коммитом
   прогнать на всех версиях кода и обновить таблицу результатов в `.md`.
4. Правки в чужих сервисах — через свою ветку и с согласия автора.
5. Контракт входа/выхода не меняем без согласования со всей командой.

Подробнее — в [AGENTS.md](AGENTS.md).

## CPU и стоимость внешних моделей

Исторический замер на исходных 208 строках: RuBERT-tiny2 на CPU дал micro-F1
0.786704, медиану 3.58 мс на уникальный вход на Apple M4 Pro. Это дообученная
компактная модель, не TinyBERT. Замер не относится к новым весам `improved_dataset`.

Все API-клиенты теперь учитывают usage, повторы и локальный/provider cache.
Обычный CLI дополнительно сохраняет `<output>.costs.json`, включая ошибки;
`run.py` и общий бенчмарк также сохраняют расходы. Тарифы задаются через
`LLM_PRICING_FILE` по точному endpoint/model; неизвестная цена — `null`, не ноль.

[CPU-замер, стоимость, ограничения и формат тарифов](src/subcategory_extraction/benchmarks/cpu-costs.md).
