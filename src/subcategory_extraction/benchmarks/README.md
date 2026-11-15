# Бенчмарки выделения микрокатегорий

[Открыть единый автономный HTML-отчёт](response_time/benchmark-report.html):
пять раздельных dataset-scope, графики время–F1, сортировка/фильтр таблиц, provenance и аудит.
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
