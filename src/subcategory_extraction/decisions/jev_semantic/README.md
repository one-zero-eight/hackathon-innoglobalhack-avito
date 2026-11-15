# Jev semantic

Selector: `decisions.jev_semantic`. Сохраняет BasePipeline/run(PipelineInput)->list[int].
Новый общий prompt различает предложение и упоминание, актуальное и архив,
область отказа и границы101–106. Не содержит исключений для отдельных объявлений.

Default auto загружает `artifacts/calibration.json`, проверяет provider/model/endpoint и
hash prompts+definitions. Выбран task на96 manual_train fit-группах, group-crossfit3,
контроль48 отдельных групп; private/public-test группы исключены. 101 фиксирован0.5.
JEV_SEMANTIC_PROMPT=task/strict включает заранее зафиксированный prompt при threshold0.5
для воспроизведения train-эксперимента. Нативные Jev веса не дообучаются.

Ресурсы калибровки включены в модуль, дополнительных скачиваний нет.
Для воспроизведения использовать сохранённый grouped manifest и PostgreSQL:
`uv run python src/subcategory_extraction/decisions/jev_semantic/train.py`.
Private labels не подбирают веса/пороги. Эта реализация не выбрана private-победителем;
на development211 micro-F1 0.935264, на manual_test (139) — 0.790698, case_test (208) — 0.755208.
Все сравнения и происхождение — в существующих benchmarks/classifier_methods и improved_dataset.
