# Jev semantic arbiter

Selector: `hybrid.jev_semantic_arbiter`. Один native Jev prior и один structured-LLM
полный смысловой арбитраж, а не распознавание ID или поиск сохранённых эталонов.
Использует calibration/artifacts соседнего decisions.jev_semantic. Модель арбитра
и API-ключи задаются через существующие STRUCTURED_LLM_*/BESTAPI_API_KEY.
В измерении: Selectel typesafe/jev-1.13 + BestAPI gpt-6.1-terra.

Для Kimi K3.1 Pro задать `STRUCTURED_LLM_MODEL=kimi-k3.1-pro` и
`STRUCTURED_LLM_SCHEMA_IN_PROMPT=1`. Второй параметр повторяет ту же JSON Schema
в system message: BestAPI принимал response_format, но в синтетических проверках
Kimi возвращал другую структуру. Строгая Pydantic-валидация и проверка всех категорий
сохраняются; невалидные ответы не кэшируются. Без флага GPT-запросы не меняются.
Готовая фиксированная benchmark-конфигурация: `jev_semantic_arbiter::kimi-k3.1-pro`.
Категорийные правила, исходный prompt и native Jev/calibration прежние. Сравнение
использует уже просмотренный private split, поэтому не является новым holdout.

Каждая категория проверяется, включая низкие native prior; схема требует ровно одно
бинарное offered для каждого eligible mcId. Source категория исключается обычным
контрактом; all-six/no-source режим существует только в benchmark-адаптере.
Выходные0/1 — бинарные решения, не калиброванные вероятности.
Промпт общий для всех объявлений, входной полный текст трактуется только как данные.
Дополнительных локальных весов или скачивания ресурсов арбитру не нужно.

Победитель выбран на211 development перед51 grouped confirmation. Private full F1
0.978947; confirmation 0.972222, N=51; no-exact-train-match 0.977561, N=222.
95% group-bootstrap confirmation interval 0.928–1.000; baseline aggregate ранее известен.
На case_test 0.826709 и manual_test 0.715556 ниже прежнего calibrated baseline;
это явный domain-specific experimental selector, default config не меняется.
Fresh validation latency mean 5.320647 с, N=10; два API-этапа, usage сохранён, тариф неизвестен.

Kimi-сравнение: full262 0.971684, no-exact-train222 0.969990; прежние development211
0.974176 /confirmation51 0.958904. Case_test208 0.821705, manual_test139 0.738197.
Fresh validation N10: mean12.138819 с, median12.033032, p9517.568494, кэш-попаданий0.
Первый timing-прогон остановился на schema-invalid ответе после6 валидных вызовов;
повтор фиксированного N10 завершён полностью, первый сбой сохранён в JSON.
В F1-проверках также были3 разных schema-invalid ответа и1 обрыв соединения;
валидные кэши сохранены, каждый невалидный ответ повторён один раз без настройки правил.
Kimi улучшает manual относительно GPT, но уступает на private/official и по latency;
глобальные настройки по умолчанию не переключены.

Протоколы/все ответы: benchmarks/classifier_methods/jev-semantic-*.json и
private-test-results.json; общий HTML содержит ту же единую таблицу.
Новые параметры после confirmation не подбирались. Для восстановления использовать
проверенные API-кэши и исходные per-partition out/semantic-*.json, не менять split.
