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
поэтому для работы нужен запущенный бэкенд пайплайна (WP2/WP3).

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

## Структура

```
src/
├── api.ts                  # типы контракта и fetch-клиент
├── theme.ts                # тема Mantine в стиле avito.ru
├── main.tsx                # MantineProvider + App
├── App.tsx                 # корневой компонент (рендерит HomePage)
├── index.css               # Mantine styles + Tailwind theme/utilities (без preflight)
├── components/
│   ├── DraftsForm.tsx      # Textarea + Select + кнопка «Подобрать черновики»
│   ├── DraftCard.tsx       # карточка одного черновика
│   ├── ResultsList.tsx     # сетка карточек черновиков
│   └── EmptyState.tsx      # «Подходящих микрокатегорий не найдено»
└── pages/
    └── HomePage.tsx        # заголовок, форма, состояния загрузки/ошибки/результата
```

## Замечания по стилю

- Фирменный синий `#00AAFF` и оттенки вокруг него объявлены в `theme.ts`
  (`createTheme`), это `primaryColor: 'avito'`.
- Tailwind подключён **только слоями `theme` и `utilities`** (`index.css`),
  чтобы preflight Tailwind не конфликтовал с reset'ом Mantine.
- Цвет текста (`#2B2B2B`) и фон страницы (`#F4F5F6`) зафиксированы в
  не-layered CSS, поэтому имеют приоритет над Mantine-слоем.
