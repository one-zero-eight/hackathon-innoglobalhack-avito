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
