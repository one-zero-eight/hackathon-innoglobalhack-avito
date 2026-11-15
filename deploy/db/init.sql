-- Инициализация базы данных веб-сервиса черновиков Avito.
-- Справочник микрокатегорий категории «Ремонт и отделка».
-- Названия точно совпадают с MICROCATEGORIES в src/main.py.

CREATE TABLE IF NOT EXISTS microcategories (
    mc_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL
);

-- Идемпотентно: повторный запуск не меняет уже вставленные строки.
INSERT INTO microcategories (mc_id, title) VALUES
    (101, 'Ремонт под ключ'),
    (102, 'Штукатурные работы'),
    (103, 'Плиточные работы'),
    (104, 'Сантехника'),
    (105, 'Электрика'),
    (106, 'Поклейка обоев и малярные работы')
ON CONFLICT (mc_id) DO NOTHING;

-- Датасет кейса InnoGlobalHack × Avito: объявления public-train-annotated.jsonl
-- (split=train) и public-test-dataset.parquet (split=test).
-- Загружает лоадер: PYTHONPATH=src uv run python -m api.dataset_loader (WP9).
-- row_key — ключ идемпотентности: train:{item_id} или test:{номер}:{хэш текста}
-- (в test есть полные дубликаты текстов, хэш одного текста их схлопнул бы).
-- Тот же DDL в src/api/dataset_loader.py — для существующих баз без init.sql.
CREATE TABLE IF NOT EXISTS case_ads (
    id            BIGSERIAL PRIMARY KEY,
    split         TEXT NOT NULL CHECK (split IN ('train', 'test')),
    item_id       BIGINT,
    title         TEXT NOT NULL,
    description   TEXT NOT NULL,
    source_mc_id  SMALLINT NOT NULL DEFAULT 101,
    found_mc_ids  JSONB,
    categories    JSONB,
    row_key       TEXT NOT NULL UNIQUE,
    CHECK (found_mc_ids IS NULL OR jsonb_typeof(found_mc_ids) = 'array'),
    CHECK (categories  IS NULL OR jsonb_typeof(categories)  = 'array')
);

CREATE INDEX IF NOT EXISTS case_ads_split_idx ON case_ads (split, id);
