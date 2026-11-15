# Заранее подготовленные веса и ресурсы

Перенос структуры не переобучает модели, не меняет содержимое весов и не
переносит внешние `out/`-артефакты в Git. Решение об их хранении принимается
отдельно. Все пути ниже указаны относительно корня репозитория.

## Собственные обученные модели

| Подход | Что нужно до запуска | Текущее расположение по умолчанию |
| --- | --- | --- |
| `lexical.tfidf` | Словарь, IDF, коэффициенты и intercept логистических голов | `out/tfidf.json`; матрица бенчмарков также использует `out/classifier-methods/tfidf.json` |
| `lexical.learned_keywords` | TF-IDF, выученные фразы и коэффициенты классификаторов | `src/subcategory_extraction/lexical/learned_keywords/artifacts/learned-keywords.json` |
| `lexical.snippet_logistic_regression` | Словарь ключей, признаки и коэффициенты оконного классификатора | `src/subcategory_extraction/lexical/snippet_logistic_regression/artifacts/model.json.gz` |
| `semantic.semantic_encoder` | Дообученный RuBERT, классификационная голова, токенизатор и метаданные | `out/semantic-encoder/` |
| `semantic.semantic_encoder_calibrated` | RuBERT, голова, токенизатор и калибровка порогов | `out/rubert-thresholds/encoder/` |
| `semantic.semantic_encoder_f1` | Дообученный/дистиллированный encoder, голова, токенизатор, метаданные и пороги | `src/subcategory_extraction/semantic/semantic_encoder_f1/artifacts/` |
| `semantic.semantic_evidence` | Два encoder/head/tokenizer-комплекта document/evidence, веса ансамбля и пороги | `src/subcategory_extraction/semantic/semantic_evidence/artifacts/` |

`semantic_encoder_f1` и `semantic_evidence` содержат опубликованные веса;
крупные encoder-файлы используют Git LFS. При клонировании нужен LFS checkout.
`learned_keywords` и `snippet_logistic_regression` содержат свои обученные
артефакты в Git. Наличие внешних `out/`-моделей зависит от локальной среды.

## Гибриды, использующие уже обученные веса

| Подход | Зависимость |
| --- | --- |
| `hybrid.jev_tfidf` | TF-IDF-модель из `lexical.tfidf`; собственной нейросети нет |
| `hybrid.jev_arbiter` | TF-IDF-модель для маршрутизации спорных категорий; арбитр — внешняя LLM |

## Другие заранее подготовленные артефакты

| Подход | Ресурс | Это веса нашей модели? |
| --- | --- | --- |
| `decisions.local_jev` | Upstream pretrained jevos-v5: OpenVINO INT8 веса, токенизатор и runtime; скачиваются при первом запуске | Нет, готовая внешняя модель без нашего дообучения |
| `decisions.jev_calibrated` | `artifacts/calibration.json`: выбранный промпт и пороги по категориям, привязанные к провайдеру/модели/endpoint | Нет, калибровка |
| `decisions.qwen_decision` | `artifacts/calibration.json` для auto; явные guided/boundaries работают без калибровки с порогом 0.5 | Нет, калибровка |
| `decisions.jev_retrieval` | Размеченная train-коллекция `out/hybrid-training.jsonl`; TF-IDF-индекс соседей строится при загрузке | Нет, справочная коллекция и индекс |

`regex` не требует обучения. `google_decision`, `jev`, `jev_boundaries` и
`hybrid.evidence_jev` не требуют локальных обученных весов: им нужны промпты
и доступ к соответствующим API. Веса внешних API-моделей в репозитории нет.

## Что сохранено при переносе

Содержимое обученных артефактов и их контрольные суммы не изменены.
Пути внутри старых metadata/calibration/selection JSON — исходное происхождение,
а не новые результаты. В `semantic_encoder_f1/artifact-manifest.json` обновлён
только путь к перемещённой папке, хеши файлов сохранены.
Протоколы обучения и исходные split-манифесты не менялись; изменены импорты
и вычисление путей из-за дополнительного уровня вложенности.
