# dataset_annotation

Служебная папка для всей работы с разметкой датасета InnoGlobalHack × Avito:
аудита исходников, дедупликации, интерактивного разбора, полной разметки,
смысловых уточнений и сборки производного набора.

Это **не бенчмарк** и **не пайплайн классификатора**. Здесь нет оценки моделей,
подбора промптов и измерения micro-F1. Код переработки данных хранится отдельно
от `subcategory_extraction/` и `draft_generation/`.

Итоговый производный набор по-прежнему лежит в `docs/improved_dataset/`.
Исходные файлы в `docs/innoglobalhack_avito/` не изменяются.

## Состав

| Путь | Назначение |
| --- | --- |
| `audit_dataset.py` | аудит структуры, повторов, пересечений и лексических сигналов |
| `build_improved_dataset.py` | сборка и проверка `docs/improved_dataset/` по решениям |
| `review_dataset.py` | интерактивный разбор спорных групп в терминале |
| `annotate_dataset.py` | офлайн-предложения полной разметки по пользовательским принципам |
| `finalize_dataset_annotations.py` | проверка и применение полной разметки только с `--apply` |
| `merge_reviews.py` | проверка полноты и объединение параллельных смысловых разборов |
| `repair_dataset.py` | полный повторный проход без непроверенных решений шардов, с журналом изменений |
| `render_dataset_audit.py` | подробный общий отчёт для ручной проверки |
| `render_train_audit.py` | отдельный подробный отчёт по train |
| `check_review_cli.py` | служебная проверка CLI на временной копии решений |
| `check_annotations.py` | служебная проверка инвариантов разметки без записи данных |
| `reports/` | Markdown-отчёты; существующие файлы исторические |
| `work/` | кеш и временные файлы; исключён из Git |

`work/dataset-audit.json` — кеш аудита. Временные копии решений, блокировки и
прочие служебные файлы также создаются в `work/`, а не в корневом `out/`.

## Команды

Все команды запускаются из корня репозитория с `PYTHONPATH=src`.

```sh
# 1. Аудит исходников; результат: work/dataset-audit.json
PYTHONPATH=src uv run python -m dataset_annotation.audit_dataset

# 2. Сборка и проверка docs/improved_dataset по текущим decisions.json
PYTHONPATH=src uv run python -m dataset_annotation.build_improved_dataset

# 3. Интерактивный разбор; изменения сохраняются после каждого ответа
PYTHONPATH=src uv run python -m dataset_annotation.review_dataset
PYTHONPATH=src uv run python -m dataset_annotation.review_dataset --preview

# 4. Предложения полной разметки; результат: work/annotation-proposals.jsonl
PYTHONPATH=src uv run python -m dataset_annotation.annotate_dataset

# 5. Проверка полной разметки, затем применение только с --apply
PYTHONPATH=src uv run python -m dataset_annotation.merge_reviews --apply
PYTHONPATH=src uv run python -m dataset_annotation.finalize_dataset_annotations
PYTHONPATH=src uv run python -m dataset_annotation.finalize_dataset_annotations --apply
```

Финализатор отказывается применять предложения с неразобранными сомнениями.
Принципы и ответы пользователя сохранены в
`docs/improved_dataset/annotation_policy.json`, отдельные смысловые уточнения —
в `annotation_overrides.json` рядом с итоговыми данными.

По просьбе пользователя итоговую смысловую проверку можно отложить:

```sh
PYTHONPATH=src uv run python -m dataset_annotation.merge_reviews --allow-partial --apply
PYTHONPATH=src uv run python -m dataset_annotation.finalize_dataset_annotations --allow-unreviewed --apply
```

Такая сборка помечается предварительной. Сомнения не удаляются:
`docs/improved_dataset/pending_review.jsonl` содержит полные исходные записи
для последующего разбора. Это не подтверждённая человеком разметка.

### Отчёты и служебная проверка

```sh
PYTHONPATH=src uv run python -m dataset_annotation.render_dataset_audit
PYTHONPATH=src uv run python -m dataset_annotation.render_train_audit
PYTHONPATH=src uv run python -m dataset_annotation.check_review_cli
PYTHONPATH=src uv run python -m dataset_annotation.check_annotations
```

`check_review_cli` — не бенчмарк, а короткая служебная проверка ввода,
архивирования, атомарной записи и продолжения разбора. Она работает только
если доступны `work/dataset-audit.json` и `decisions.json`; настоящие ответы
не изменяются.

`check_annotations` — тоже не бенчмарк качества: она только читает реальные
исходники, решения и текущие выходы. Проверяет
покрытие групп, строгие метки, приоритет пользовательских ответов,
`policy_id`/SHA-256 политики и смысловых уточнений, а также неизменность
рабочих файлов. Без `decisions.json` проверка не запускается; кеш аудита
для неё не обязателен.

## Порядок работы

Актуальный исправленный набор пересобирается командой:

```sh
PYTHONPATH=src uv run python -m dataset_annotation.repair_dataset
```

Она сохраняет историю, исключает непроверенные правила шардов и заново
вычисляет метки всех 4993 групп. `repair_changes.jsonl` содержит отличия
от первоначального предварительного набора. Ручная проверка всех описаний
не заявляется; диагностические флаги сохраняются.

1. `audit_dataset` строит кеш в `work/`.
2. `build_improved_dataset` собирает текущий набор и проверяет инварианты.
3. `review_dataset` разбирает спорные случаи; решения попадают в
   `docs/improved_dataset/decisions.json`.
4. `annotate_dataset` формирует предложения для всех уникальных групп.
5. `finalize_dataset_annotations` проверяет покрытие и пользовательские
   исключения, а с `--apply` записывает аннотации и пересобирает набор.

Пользовательские ответы имеют приоритет. Финализатор не должен менять
сохранённые `user_chat_review` и `user_cli_review`, а `--apply` выполняется
только после успешной проверки без записи.

## Исторические отчёты

Файлы в `reports/` — история служебных проверок исходной разметки и старых
версий сборки. Они **не являются результатами новой политики полной разметки**
и не должны использоваться как её подтверждение. Даты и версии указаны внутри
отчётов; старые цифры не переписываются под текущее состояние.
