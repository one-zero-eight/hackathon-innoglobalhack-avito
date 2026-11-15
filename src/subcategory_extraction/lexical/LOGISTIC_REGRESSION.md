# Логистическая регрессия для услуг и формата заказа

Классификаторы услуг оценивают микрокатегории 102–106 по признакам из
локальных окон текста. Модули формата заказа учитывают, какие работы
исполнитель готов брать в отдельный заказ. Все варианты работают локально
на CPU и подключаются через `subcategory_extraction` в `config.json`.

| Модуль в `lexical` | Назначение |
| --- | --- |
| [`snippet_logistic_regression`](snippet_logistic_regression/README.md) | Базовый классификатор услуг. Открывает окна вокруг выученных ключевых слов и биграмм. |
| [`morphology_logistic_regression`](morphology_logistic_regression/README.md) | Открывает окна до 11 слов вокруг форм из словаря. По умолчанию веса Avito. `MORPH_LOGREG_MODEL=yandex-youla-decision1` выбирает веса Яндекса и Юлы. |
| [`order_format_joint`](order_format_joint/README.md) | Убирает поисковые хвосты, выделяет услуги snippet-логрегом, проверяет упоминания для 102, 103 и 105 и применяет запреты. |
| [`order_format_veto`](order_format_veto/README.md) | Снимает готовые категории по общим и адресным запретам. Хранит дословные цитаты для снятий. |
| [`order_format_scope`](order_format_scope/README.md) | Сопоставляет действие и область запрета или разрешения в пределах фразы. |
| [`order_format_logistic_regression`](order_format_logistic_regression/README.md) | Бинарный логрег на уровне объявления. Снимает все категории при общем отказе от частичных заказов. |
| [`order_format_phrase_logreg`](order_format_phrase_logreg/README.md) | Фразовый логрег. Различает общий и адресный отказ, разрешение и нейтральный текст. |

Для цепочки с фильтром укажите `lexical.order_format_joint` в конфиге.
По умолчанию она загружает веса базового snippet-логрега. Для весов дедупа
Avito или Яндекса и Юлы задайте в `SNIPPET_LOGISTIC_ARTIFACT` абсолютный путь
к `dedup.json.gz` или `yandex_youla.json.gz` из
[`order_format_joint/artifacts/`](order_format_joint/artifacts/).

На [`manual_test`](../benchmarks/improved_dataset/README.md) из 139 строк
micro-F1 базового логрега составил 0.759295, морфологического с весами Avito
— 0.723849. Для цепочки с фильтром предыдущий замер на 699 отложенных
объявлениях локального дедупа дал 0.925476 с весами Avito и 0.920479
с весами Яндекса и Юлы. В этой выборке фильтр сработал лишь в одном и трёх
объявлениях соответственно. Разбиения и команды воспроизведения описаны
в README модулей.
