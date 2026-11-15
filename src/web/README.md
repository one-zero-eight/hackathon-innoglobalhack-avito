# Frontend · Сервис черновиков Avito

Одностраничный интерфейс подбора дополнительных черновиков объявлений в
микрокатегориях раздела «Ремонт и отделка». Стек: Vite 8 + React 19 +
TypeScript (strict) + Mantine 9 + Tailwind CSS v4.

## Запуск

```bash
npm install
npm run dev          # http://localhost:5173
```

Dev-сервер проксирует `/api/*` на `http://localhost:8000` (см. `vite.config.ts`),
поэтому для работы нужен запущенный бэкенд пайплайна (WP2/WP3). Адрес
бэкенда переопределяется переменной окружения `VITE_BACKEND_ORIGIN`
(например, `VITE_BACKEND_ORIGIN=http://127.0.0.1:8787 npm run dev` —
для проверки на мок-SSE).

## Команды

| Команда         | Что делает                                            |
| --------------- | ----------------------------------------------------- |
| `npm run dev`   | dev-сервер Vite с прокси на бэкенд                    |
| `npm run build` | `tsc -b && vite build` — проверка типов и сборка     |
| `npm run lint`  | `oxlint` по конфигу `.oxlintrc.json`                  |
| `npm run preview` | раздача собранного `dist/`                          |

## Контракт с бэкендом

- `GET /api/microcategories` → `[{ "mcId": 101, "title": "Ремонт под ключ" }, …]`
- `POST /api/drafts` с телом `{ "description": "…", "mcId": 101 }` →
  `{ "drafts": [{ "mcId": 104, "title": "…", "text": "…" }] }`
  (у черновика `title`/`text` могут быть `null` — UI показывает плашку
  «Текст не сгенерирован»).
- Ответ с HTTP-ошибкой: `{ "detail": "…" }` — текст показывается в `Alert`.
- `POST /api/drafts/batch` с телом `{ "ads": [{ "description", "mcId" }], "concurrency": 4 | null }`
  → поток `text/event-stream` (SSE), кадры `event: <type>\ndata: <json>\n\n`:
  - `start` → `{ total, concurrency }` — начало обработки;
  - `result` → `{ index, mcId, drafts: [...] }` — готовая строка (`index` —
    0-based позиция во входном массиве, порядок может отличаться от входного);
  - `error` → `{ index, detail }` — ошибка строки;
  - `done` → `{ processed, succeeded, failed }` — итог.
  Клиент `streamBatchDrafts()` (`api.ts`) читает `response.body.getReader()`,
  разбирает кадры накопительно (`parseSseFrames`) и вызывает обработчики
  onStart/onResult/onError/onDone; не-2xx и сетевые сбои → `ApiError` с
  русским сообщением, отмена — через `AbortSignal`.

## Пакетная обработка (режим «Загрузка таблицы»)

После импорта файла кнопка «Обработать все» отправляет весь массив на
`POST /api/drafts/batch` и прогрессивно показывает результаты по строкам
файла: прогресс-бар done/total, статусы «ожидание / в работе / готово /
ошибка», раскрывающиеся карточки черновиков (переиспользуют `DraftCard`),
detail ошибки. Доступны выбор параллелизма 1–8 (по умолчанию 4), отмена
(«Остановить», `AbortController`) и экспорт «Скачать JSON» / «Скачать CSV»
(`bulkExport.ts`, Blob + download, без зависимостей). CSV-колонки:
`index, description, source_mcId, target_mc_ids, draft_title, draft_text`
(по строке на черновик). Одиночный ручной поток («Ручной ввод») не зависит
от пакетного.

## Структура

```
src/
├── api.ts                  # типы контракта, fetch-клиент, SSE-разборщик потока batch
├── bulkTypes.ts            # состояния строк/фазы пакетной обработки
├── bulkExport.ts           # экспорт результатов в JSON/CSV (Blob + download)
├── theme.ts                # тема Mantine: палитры Avito + цветовая кодировка категорий
├── main.tsx                # MantineProvider + App
├── App.tsx                 # корневой компонент (рендерит HomePage)
├── index.css               # Mantine styles + Tailwind theme/utilities (без preflight)
├── components/
│   ├── AvitoLogo.tsx       # официальный логотип/знак из public/
│   ├── AppHeader.tsx       # шапка: лого, подпись, радужная полоса
│   ├── AppFooter.tsx       # футер: лого, дисклеймер, копирайт
│   ├── icons.tsx           # inline-SVG иконки (стиль Lucide, без зависимостей)
│   ├── AdsDropzone.tsx     # импорт таблицы + «Обработать все» и параллелизм
│   ├── DraftsForm.tsx      # Textarea + Select + кнопка «Подобрать черновики»
│   ├── DraftCard.tsx       # карточка черновика: топ-полоска и бейдж категории
│   ├── ResultsList.tsx     # сетка карточек черновиков
│   ├── BulkResults.tsx     # таблица пакетных результатов, прогресс и экспорт
│   └── EmptyState.tsx      # «Подходящих микрокатегорий не найдено» + ватермарк
└── pages/
    └── HomePage.tsx        # заголовок, форма, ручной режим и пакетная задача
```

## Брендинг

- В `public/avito-logo.svg` — официальный логотип Avito (знак из четырёх
  точек + вордмарк), геометрия и цвета сняты с инлайнового SVG шапки
  avito.ru. В `public/favicon.svg` — тот же знак (используется как favicon
  и как компактный mark в скелетоне/ватермарке).
- Логотип — товарный знак Avito и используется в учебном проекте
  исключительно в демонстрационных целях (см. дисклеймер в футере).

## Замечания по стилю

- Все фирменные цвета объявлены палитрами в `theme.ts` (индекс 5 — точный
  брендовый hex: `avito.5` = `#00AAFF`, `brandGreen.5` = `#04E061`,
  `brandCoral.5` = `#FF4053`, `brandViolet.5` = `#965EEB`, `brandYellow.5` =
  `#FFD400`, `brandSea.5` = `#0074B3`); компоненты берут цвета только через
  CSS-переменные Mantine, hex в коде компонентов не встречается.
- Микрокатегории 101–106 закодированы фирменными парами цветов
  (`microcategoryPalette` в `theme.ts`): 101 — синий, 102 — фиолетовый,
  103 — зелёный, 104 — глубокий синий, 105 — жёлтый, 106 — коралловый.
  Пары «фон бейджа (оттенок 0) / текст (оттенок 8)» дают контраст не ниже
  WCAG AA (от 5.5:1 до 9.9:1 на белом).
- Радужная градиентная полоса под шапкой, hover карточек и переходы
  (180 мс) описаны в `index.css` через CSS-переменные палитр.
- Тач-цели: поля ввода и кнопки не ниже 44 px (переопределение высот
  Mantine в `index.css` и `theme.ts`).
- Tailwind подключён **только слоями `theme` и `utilities`** (`index.css`),
  чтобы preflight Tailwind не конфликтовал с reset'ом Mantine.
- Цвет текста (`#2B2B2B`) и фон страницы (`#F4F5F6`) зафиксированы в
  не-layered CSS, поэтому имеют приоритет над Mantine-слоем.
