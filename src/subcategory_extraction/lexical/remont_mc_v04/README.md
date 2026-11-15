# Классификатор remont_mc v0.4

Модуль `lexical.remont_mc_v04` объединяет правила и RuModernBERT-base.
Он возвращает дополнительные микрокатегории 102–106 и исключает исходный `mcId`.
Устройство модели и ограничения проверки описаны в [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Установка весов

Веса ONNX распространяются отдельно от Git в архиве `remont-mc-v04-weights.tar.gz`.
Внутри архива находятся токенизатор, метаданные и `SETUP.md`. Из корня проекта:

```bash
tar -xzf /ПУТЬ/К/remont-mc-v04-weights.tar.gz -C src/subcategory_extraction/lexical/remont_mc_v04
uv sync
```

В `config.json` выберите `"subcategory_extraction": "lexical.remont_mc_v04"`.
После изменения конфига или установки весов перезапустите API. Без ONNX-файла
классификатор завершится ошибкой; другую модель он не выбирает.

SHA-256 `artifacts/model_int8.onnx`:
`33f9ab6dbbb6705bafdcf999df7e7814f22eec2397a344e4ec72632e2af68de1`.

Веса и код получены из предоставленного архива `remont_mc_v0.4_full.tar.gz`.
Обучающий набор, промежуточные результаты и архив весов не включены в Git.
