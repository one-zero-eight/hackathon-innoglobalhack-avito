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

## Первая загрузка и запуск

Выберите `"subcategory_extraction": "local_jev"` в `config.json` и запустите
приложение обычной командой:

```bash
uv run python src/main.py --input data/input.json --output out/output.json
```

Если локальный сервер ещё не работает, модуль при создании клиента автоматически:

1. Скачивает закреплённые официальные архивы модели jevos-v5 и Linux runtime.
2. Проверяет SHA-256 архивов, затем размеры и SHA-256 каждого файла модели
   по `artifact-manifest.json`.
3. Распаковывает их в игнорируемый `out/local-jev/` и запускает сервер на localhost.
4. Ждёт готовности модели перед обработкой объявления.

Модель и runtime **не хранятся в Git и не требуют Git LFS**. В Git остаются
код, промпты, лицензии и манифест. Первая загрузка требует сети и примерно
630 МБ для model-архива плюс 27 МБ для runtime. Повторные запуски используют
кэш без сети; повреждённая модель восстанавливается из проверенного архива.
Временный файл скачивания удаляется при ошибке, непроверенный архив не используется.
Параллельные процессы сериализуют загрузку и старт через файловую блокировку.

Кэш по умолчанию определяется относительно модуля, независимо от текущего
каталога. Путь можно переопределить через `LOCAL_JEV_CACHE_DIR`.
Файлы модели находятся в `<cache>/model/`, runtime — `<cache>/runtime/`,
журнал процесса — `<cache>/server.log`. Собственный дочерний сервер завершается
при штатном завершении приложения. Уже работающий сервер используется напрямую;
приложение не останавливает его и ничего для него не скачивает.

Автозагрузка runtime поддерживает Linux x86_64 (glibc 2.35+). На другой ОС
запустите соответствующий официальный runtime из того же релиза отдельно.
Для работы с отдельно управляемым сервером задайте `LOCAL_JEV_AUTOSTART=0`.
Имя модели в `/health` должно совпадать с `LOCAL_JEV_MODEL`; для другой модели
автозагрузка jevos-v5 не подставляется.

Автоматический запуск использует те же параметры, что исходное сравнение:
`--threads 4 --warmup 0 --state-cache 0 --batch-tokens 0`. Число потоков
настраивается через `LOCAL_JEV_THREADS`. Клиент не кэширует ответы, state-cache
сервера отключён. `GET /health` возвращает имя, хеши весов и параметры движка;
адаптер сохраняет их в бенчмарках.

Настройки `.env`: `LOCAL_JEV_BASE_URL`, `LOCAL_JEV_MODEL`, `LOCAL_JEV_PROMPT`
(`english` по умолчанию или `guided_ru`), `LOCAL_JEV_THRESHOLD`,
`LOCAL_JEV_TIMEOUT`, `LOCAL_JEV_AUTOSTART`, `LOCAL_JEV_THREADS`, опционально
`LOCAL_JEV_CACHE_DIR`, `LOCAL_JEV_PROMPT_PATH` и `LOCAL_JEV_API_KEY`.
При автозапуске ключ клиента передаётся дочернему серверу как `JEV_API_KEY`.
Для отдельно запущенного сервера задайте оба ключа самостоятельно.
Генератор `sample_llm` использует свой ключ.

Исторические F1 и latency ниже относятся к прежним измерениям готовой модели.
Загрузка, распаковка и старт сервера в них не входили. Новый загрузчик использует
ровно те же веса по SHA-256; новый F1 или скорость загрузки этими числами
не заявляются. Общий runner времени исключает создание клиента, включая
первичную загрузку и запуск. При воспроизведении холодный и готовый сервер
следует различать.

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
