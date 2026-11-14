# Локальный jevos-v5 (feder-cr/jev)

Модуль `local_jev` использует официальный [feder-cr/jev, релиз jevos-v5](https://github.com/feder-cr/jev/releases/tag/jevos-v5)
через локальный System One HTTP API. Это отдельная открытая модель, отличная от
облачного `subcategory_extraction/jev` (`jev-1.13.0`). Внешний API для классификации
не используется. Исполнительный код адаптера — `pipeline.py` и `client.py`.

`SubcategoryExtractionPipeline(BasePipeline).run(inp: PipelineInput) -> list[int]`
посылает один запрос с пятью Noul-вопросами, исключает исходный mcId и проверяет
точный набор ответов, тип и диапазон вероятностей, имя обслуживаемой модели.
Клиент не кэширует ответы и не повторяет запросы. Ошибки сервера/контекста
не заменяются другим классификатором и прерывают прогон.

Первый прогон: официальный OpenVINO INT8 jevos-v5, общий русский `guided`-промпт
из `../google_decision/prompts/google_decision_guided.md`, порог 0.5. Тот же промпт
использовал облачный Jev в историческом сравнении. Русский вариант сохранил
F1 0.000000 / 0.009302 на исходном / диагностическом test.

Затем язык вопросов проверен только на 12 непустых примерах официального train
(`validation_sample(12, seed=44, nonempty=True)`): русский guided дал F1 0,
английский — 0.452830. Это небольшая диагностическая выборка, не независимая
оценка качества. По ней выбран `english`, порог остался 0.5, без поиска порога.
Английский промпт и описания категорий сохранены в `prompts/`, отчёт выбора —
в существующем сценарии [improved_dataset](../benchmarks/improved_dataset/local-jev-language-selection.json).
После этого оба test оценены заново, русский прогон сохранён отдельно в JSON.
Начальный test не использовался для расчёта порога или выбора английских формулировок;
проверка языка последовала после наблюдения провала русского варианта.

Веса не обучались на данных задачи. Состав upstream train неизвестен, его пересечения с
test проверить невозможно. Это сравнение готовых конфигураций, не доказательство
превосходства архитектуры. Русские тексты передаются без перевода и усечения.

## Веса и запуск

Все файлы официального model-архива хранятся рядом с модулем, в `artifacts/`;
`openvino_model.bin` отслеживается через Git LFS. SHA-256 файлов и исходных архивов
зафиксированы в `artifact-manifest.json`. Для checkout используйте `git lfs pull`.
Для коммита весов нужен установленный Git LFS; правило для `.bin`
добавлено в корневой `.gitattributes`.
Runtime с библиотеками и лицензиями загружается отдельно в игнорируемый `out/`.
На другой ОС скачайте соответствующий архив из того же релиза.

Воспроизведение на Linux x86_64 (glibc 2.35+), из корня репозитория:

```bash
mkdir -p out/local-jev-download out/local-jev-runtime
curl -fL https://github.com/feder-cr/jev/releases/download/jevos-v5/jev-linux-x64.tar.gz -o out/local-jev-download/jev-linux-x64.tar.gz
curl -fL https://github.com/feder-cr/jev/releases/download/jevos-v5/jevos-v5-openvino-int8.zip -o out/local-jev-download/jevos-v5-openvino-int8.zip
```

Перед распаковкой проверьте `sha256sum` скачанных файлов по манифесту:

| Файл | SHA-256 |
| --- | --- |
| jev-linux-x64.tar.gz | `1ce77566405514ed2f20d64f8222cf9eba9b714d332f05f6296e0cad84dd7b04` |
| jevos-v5-openvino-int8.zip | `0a1159c31b9c2abc484af9c866cdcfe5279fdf592b368636625c5a795f970acc` |

Если `artifacts/` уже загружены Git LFS, повторно распаковывать model-архив не нужно.
При восстановлении весов из архива:

```bash
mkdir -p src/subcategory_extraction/local_jev/artifacts
unzip -j out/local-jev-download/jevos-v5-openvino-int8.zip 'model/*' -d src/subcategory_extraction/local_jev/artifacts
tar -xzf out/local-jev-download/jev-linux-x64.tar.gz -C out/local-jev-runtime
out/local-jev-runtime/jev/jev serve --model-dir src/subcategory_extraction/local_jev/artifacts --threads 4 --warmup 0 --state-cache 0 --batch-tokens 0
```

`--state-cache 0` исключает повторное использование состояния текста;
`--warmup 0` отключает фоновый прогрев форм графа; загрузочный прогрев runtime
остаётся частью старта сервера. `--batch-tokens 0` исключает объединение запросов.
Параметры повторяют измеренный запуск. HTTP слушает `127.0.0.1:8017`.
`GET /health` возвращает имя, хеши весов и параметры движка; адаптер сохраняет
их в каждом бенчмарке. Старт сервера и health не входят в latency `run()`.

Настройки `.env`: `LOCAL_JEV_BASE_URL`, `LOCAL_JEV_MODEL`, `LOCAL_JEV_PROMPT`
(`english` по умолчанию или `guided_ru`), `LOCAL_JEV_THRESHOLD`,
`LOCAL_JEV_TIMEOUT`, опционально `LOCAL_JEV_PROMPT_PATH` и `LOCAL_JEV_API_KEY`.
При аутентификации одинаковый ключ задаётся серверу как `JEV_API_KEY`, клиенту
как `LOCAL_JEV_API_KEY`; ключи облачного Jev не передаются localhost.

Для выбора в основном приложении задайте в `config.json`
`"subcategory_extraction": "local_jev"`; генератор остаётся `sample_llm` и требует
свой ключ. Программный выбор — через `PipelineConfig` без изменения оркестрации.

## Проверка

Проверка языка на той же выборке train (test не читается):

```bash
PYTHONPATH=src uv run python - <<'PY'
import logging
import os
from benchmark_support import validation_sample
from run import calculate_metrics
from subcategory_extraction.local_jev.pipeline import SubcategoryExtractionPipeline

logging.basicConfig(level=logging.INFO)
rows = validation_sample(12, seed=44, nonempty=True)
for variant in ("guided_ru", "english"):
    os.environ["LOCAL_JEV_PROMPT"] = variant
    pipeline = SubcategoryExtractionPipeline()
    predicted = [set(pipeline.run(row["inp"])) for row in rows]
    logging.info("%s: %s", variant, calculate_metrics([row["targets"] for row in rows], predicted))
PY
```

Оценка готового модуля:

```bash
uv sync --frozen
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods local_jev learned_keywords --append
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --methods local_jev learned_keywords --append
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module local_jev --size 10 --append
```

Оба F1-сценария сохраняют предсказания и вероятности, хеш данных, код и
происхождение модели в существующие `results.json`, сохраняя историю других
модулей. `--offline` общего runner запрещает все HTTP-модули, включая localhost;
для `local_jev` запускайте без этого флага. При сравнении исторических API-моделей
используется штатный `--replay --report out/<имя>.json`, а не новые платные вызовы.
Ссылки на результаты: [исходный test](../benchmarks/classifier_methods/README.md),
[диагностический test](../benchmarks/improved_dataset/README.md),
[время](../benchmarks/response_time/README.md).

В текущем окружении полный `uv sync --frozen` остановился на сетевом таймауте
CUDA-зависимости PyTorch. Для Jev и learned_keywords PyTorch не нужен:
проверки выполнены с доступными pydantic, python-dotenv, numpy, scikit-learn и
pyarrow; Python и версии фиксируются рядом с результатами. Lock-файл не менялся.
Свежий RuBERT не запускался: в checkout только LFS-указатель, его результаты явно
помечены историческими. Исходные `recorded-results.json` не перезаписываются.

CLI проверен с временным выбором `local_jev`, исходный config восстановлен.
На объявлении своей категории 105 успешно записан `{"drafts": []}`.
На `data/input.json` извлечение прошло, затем существующий `sample_llm`
остановился на DNS-ошибке внешнего генератора; непустые тексты черновиков
в этом окружении не проверены. Код генератора не менялся.
