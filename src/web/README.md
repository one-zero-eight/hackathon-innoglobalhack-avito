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
поэтому для работы нужен `uv run uvicorn api.app:app --app-dir src --port 8000`,
PostgreSQL и `SELECTEL_API_KEY` для неизменённого классификатора Jev.
`python -m src` запускает другой HTTP API (`/drafts`), не React-бэкенд.
Генератор один с CLI: `config.json.draft_generation=local_llm`.
Runtime и веса Qwen3.5-9B устанавливаются и запускаются отдельно
([инструкция](../../README.md#веб-сервис)); для dev endpoint
`http://127.0.0.1:8080/v1/chat/completions`, `LOCAL_LLM_MODEL_ID=default_model`.
Compose использует `host.docker.internal:8080`. Модель в образы не входит.

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
  Генератор возвращает вступление, пункты `•` и исходные эмодзи.
  При сбое генерации сервер отвечает HTTP 502, а не черновиком с `null`.
- Ответ с HTTP-ошибкой: `{ "detail": "…" }` — текст показывается в `Alert`.

## Структура

```
src/
├── api.ts                  # типы контракта и fetch-клиент
├── theme.ts                # тема Mantine: палитры Avito + цветовая кодировка категорий
├── main.tsx                # MantineProvider + App
├── App.tsx                 # корневой компонент (рендерит HomePage)
├── index.css               # Mantine styles + Tailwind theme/utilities (без preflight)
├── components/
│   ├── AvitoLogo.tsx       # официальный логотип/знак из public/
│   ├── AppHeader.tsx       # шапка: лого, подпись, радужная полоса
│   ├── AppFooter.tsx       # футер: лого, дисклеймер, копирайт
│   ├── icons.tsx           # inline-SVG иконки (стиль Lucide, без зависимостей)
│   ├── DraftsForm.tsx      # Textarea + Select + кнопка «Подобрать черновики»
│   ├── DraftCard.tsx       # карточка черновика: топ-полоска и бейдж категории
│   ├── ResultsList.tsx     # сетка карточек черновиков
│   └── EmptyState.tsx      # «Подходящих микрокатегорий не найдено» + ватермарк
└── pages/
    └── HomePage.tsx        # заголовок, форма, состояния загрузки/ошибки/результата
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
