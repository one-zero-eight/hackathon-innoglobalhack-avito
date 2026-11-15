# Бенчмарки выделения микрокатегорий

[Открыть единый автономный HTML-отчёт](response_time/benchmark-report.html):
шесть раздельных dataset-scope, графики время–F1, сортировка/фильтр таблиц, provenance и аудит.
Графики и сводный JSON можно скачать прямо из HTML. Внешние ресурсы не используются.

| Сценарий | Что измеряет |
| --- | --- |
| [classifier_methods](classifier_methods/README.md) | case_test: 208 строк; в этой же существующей папке — запрошенная диагностика provider / IGH |
| [improved_dataset](improved_dataset/README.md) | manual_test: 139 строк ручной смысловой разметки |
| [response_time](response_time/README.md) | Настоящие run(), N=10, среднее / медиана / p95 |
| [cpu_costs](cpu_costs/README.md) | Ручной учёт подтверждённой API-стоимости; локальные методы без API-расходов |

Сравнение 15 ноября 2026: все 24 модуля; 18 прежних и шесть новых из
fih/feature/logistic-regression-modules. Для шести новых модулей проверены
девять опубликованных конфигураций весов, без переобучения и подбора порогов.
Три прежних модуля недоступны из-за отсутствующих артефактов; неуспешным запускам F1 не присваивается.

Датасеты не смешиваются. IGH содержит 4840/4966 train-совпадений; provider
использует машинные метки, а provider-trained веса не имеют здесь независимого holdout.
Автор order_format_joint описал использование official-test примеров при разработке.
Все ограничения, даты и хеши сохранены в JSON и HTML.

Для воспроизведения отчёта без моделей, БД и API:

```bash
uv run --with matplotlib python src/subcategory_extraction/benchmarks/response_time/report.py
```

Для новых вызовов использовать существующие run.py и export_diagnostic.py;
команды и протоколы приведены в README соответствующих сценариев.
Канонические JSON и recorded-results.json сохранены; лишние дубли изображений,
CSV и отдельных диагностических Markdown удалены. Новых benchmark-папок нет.


Private-test: [все шесть категорий без исходной mcId](response_time/benchmark-report.html#private).
262 строки и отдельный frozen shortlist; детали/команды в classifier_methods/README.md.
Пересечение с train 40 строк; F1 также пересчитана на 222 строках без точного совпадения.
Опциональная таблица private_test не входит в default loader-набор пяти датасетов.


Новые semantic selectors: `decisions.jev_semantic` и `hybrid.jev_semantic_arbiter`.
Private improvement study (211 development /51 grouped confirmation) и ограничения
включены в ту же HTML-таблицу private. Arbiter full0.978947, confirmation0.972222;
на официальном/manual tests есть регрессии, поэтому default config сохранён.

Фиксированный вариант `jev_semantic_arbiter::kimi-k3.1-pro` сравнивает BestAPI Kimi
с GPT на тех же private/official/manual данных и 10 свежих validation-вызовах.
Категорийные правила и Jev prior не меняются; для Kimi схема также включена в system
message. Предыдущие private-разбиения уже просмотрены: это повторное сравнение,
не новая независимая confirmation. Все результаты находятся в существующих таблицах.
