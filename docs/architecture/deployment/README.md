# Deployment topology: hackathon solution

Скилл: `deployment-topology-analyzer` + Mermaid.
Срез: `projacktor/main` @ `8d90028`, 2026-11-14.

## Ответ на вопрос

Решение **не развёрнуто как сервис**. Это локальный Python-CLI: один процесс
`python src/main.py` читает JSON, вызывает два выбранных модуля, обращается
по HTTPS к BestAPI (Jev) и Google Generative Language API (черновики) и пишет
JSON-файл. Локальные модели (`tfidf`, `semantic_encoder`) доступны как
альтернативные классификаторы, но их артефакты лежат в `out/` и не коммитятся.
Средств развёртывания — Docker, Kubernetes, IaC, CI/CD, хостинга — в репозитории
нет. Поставка = исходники через Git, запуск = ручная настройка `.env` и `uv sync`.

## Файлы

| Файл                    | Что внутри                                                                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `deployment-topology.mmd` | текущая топология: рабочая машина, файловое состояние, внешние API, границы доверия |
| `release-path.mmd`        | путь от ветки/PR до запуска CLI; явно отмечены отсутствующие стадии                                 |
| `runtime-evidence.md`     | инвентарь доказательств: код, конфиги, endpoint'ы, артефакты, переменные                           |
| `deployment-risks.md`     | реестр операционных рисков, проверки и рекомендации                                                       |

## Как читать

1. `deployment-topology.mmd`: сплошные рёбра — активная конфигурация
   (`config.json`: `jev` + `sample_llm`); пунктир — альтернативы и опциональные
   зависимости. Поставка, `uv sync` и обучение вынесены в заметку
   «Outside the runtime path», чтобы не заслонять сам runtime.
2. `release-path.mmd`: красные пунктирные узлы — стадии, которых нет
   (CI, образ, деплой, мониторинг).
3. Подробности и цитаты путей — в `runtime-evidence.md`.
4. Приоритеты исправлений — в `deployment-risks.md` §4.

## Рендер диаграмм

Любой Mermaid-рендерер (GitHub/GitLab, IDE-плагин, `mermaid-cli`):

```bash
npx -y @mermaid-js/mermaid-cli \
  -i docs/architecture/deployment/deployment-topology.mmd \
  -o /tmp/deployment-topology.svg
npx -y @mermaid-js/mermaid-cli \
  -i docs/architecture/deployment/release-path.mmd \
  -o /tmp/release-path.svg
```

Оба источника проверены рендером `@mermaid-js/mermaid-cli` (headless Chrome).
Если `mermaid-cli` недоступен, `.mmd` можно открыть в любом Mermaid-превью
или вставить содержимое в Markdown-блок `` ```mermaid ``.

## Допущения и уверенность

Все узлы и рёбра — **current state** по коду, конфигам и документации;
runtime-наблюдений нет.

Endpoint'ы и версии моделей — значения по умолчанию из кода и `.env.example`.

Хостинг, площадка live-демо и владелец runtime — **unknown**.

Уверенность: высокая для локального CLI, внешних вызовов и локальных
артефактов; низкая/unknown для всего, что за пределами рабочей машины.

## Обслуживание

Обновлять при изменении: `config.json`, `.env.example`, состава модулей,
endpoint'ов, путей артефактов, способа поставки или появлении CI/Docker/IaC.
Рендер и проверку Mermaid выполнять после каждой правки.
