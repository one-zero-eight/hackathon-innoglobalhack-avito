"""Самодостаточный HTML для ручного сравнения сохранённых ответов моделей."""

import argparse
import json
import logging
from pathlib import Path


# Тот же корень, что в benchmark_support. Не импортируем его numpy и dotenv:
# для просмотра сохранённых ответов достаточно стандартной библиотеки Python.
ROOT = Path(__file__).resolve().parents[4]
LOG = logging.getLogger(__name__)


HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ручное сравнение черновиков</title>
<style>
:root { color-scheme: light dark; --bg: #f5f6f8; --panel: #fff; --ink: #202532;
  --muted: #535c6d; --border: #c9cfda; --accent: #2458a6;
  --warning-bg: #fff3ce; --warning-ink: #624600; --error: #a12128; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #15181e; --panel: #212630; --ink: #edf0f5; --muted: #bdc5d3;
    --border: #505969; --accent: #9ac0ff; --warning-bg: #3c321b;
    --warning-ink: #ffe3a0; --error: #ffa3a8; }
}
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: var(--bg);
  font: 16px/1.55 system-ui, sans-serif; }
main { max-width: 1800px; margin: auto; padding: 1.25rem; }
h1 { font-size: 1.8rem; margin: 0 0 .75rem; }
h2 { font-size: 1.3rem; margin: 0 0 .7rem; }
h3 { font-size: 1.05rem; margin: 0; }
p { margin: .5rem 0; }
section, .card { background: var(--panel); border: 1px solid var(--border);
  border-radius: .5rem; padding: 1rem; margin-bottom: 1rem; min-width: 0; }
.warning { background: var(--warning-bg); color: var(--warning-ink);
  border-left: 4px solid currentColor; padding: .7rem 1rem; }
.muted { color: var(--muted); }
.error { color: var(--error); font-weight: 600; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: .65rem 1rem; }
label { display: inline-flex; flex-wrap: wrap; align-items: center; gap: .5rem; }
select, button { font: inherit; color: var(--ink); background: var(--panel);
  border: 1px solid var(--border); border-radius: .3rem; padding: .35rem .6rem; }
select { max-width: 100%; }
button { cursor: pointer; }
button:hover { border-color: var(--accent); }
button:disabled { opacity: .45; cursor: default; }
:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
#sample-select { width: min(100%, 52rem); }
.quick { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .7rem; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: .92rem; }
caption { text-align: left; color: var(--muted); padding: .5rem 0; }
th, td { text-align: left; vertical-align: top; padding: .55rem .65rem;
  border-bottom: 1px solid var(--border); }
th { font-weight: 650; }
pre { margin: .65rem 0 0; white-space: pre-wrap; overflow-wrap: anywhere;
  word-break: normal; font: inherit; tab-size: 4; }
.source { max-height: 32rem; overflow: auto; padding: .5rem; border: 1px solid var(--border); }
.cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem;
  align-items: stretch; }
.card { display: flex; flex-direction: column; margin: 0; }
.card-header { min-height: 9rem; border-bottom: 1px solid var(--border); padding-bottom: .7rem; }
.draft { flex: 1; padding-bottom: 1rem; }
.card details { border-top: 1px solid var(--border); padding-top: .6rem; }
summary { cursor: pointer; color: var(--accent); }
.raw { max-height: 36rem; overflow: auto; font: .9rem/1.5 ui-monospace, monospace; }
.error-detail { font-size: .9rem; white-space: pre-wrap; overflow-wrap: anywhere; }
.scores { border-top: 1px solid var(--border); padding-top: .65rem; margin-bottom: .7rem; }
.scores p { margin: .15rem 0; font-size: .9rem; }
[hidden] { display: none !important; }
@media (max-width: 1200px) { .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 650px) {
  main { padding: .6rem; }
  .cards { grid-template-columns: minmax(0, 1fr); }
  .card-header { min-height: 0; }
  h1 { font-size: 1.45rem; }
}
</style>
</head>
<body>
<main>
<h1>Ручное сравнение черновиков</h1>
<p class="warning"><strong>Автоматические оценки — не ручной эталон.</strong>
Первые две метрики показаны в процентах, дополнительные — по шкале 0–4.
Это не доказательство качества и не итоговый рейтинг. Для ручного сравнения
можно скрыть оценки и названия моделей.</p>
<p class="muted">Только просмотр сохранённых входов и реальных ответов: без новой генерации,
редактирования или публикации объявлений. Отчёт работает без подключения к сети.</p>
<div class="controls">
<label><input id="blind" type="checkbox"> Скрыть названия моделей (A/B/C/D)</label>
<label><input id="show-scores" type="checkbox" checked> Показать автоматические оценки</label>
<label>Группа моделей
<select id="cohort">
<option value="local">Три локальные модели</option>
<option value="all">Все четыре модели</option>
</select></label>
</div>
<section aria-labelledby="factual-heading" style="margin-top: 1rem">
<h2 id="factual-heading">Первые две метрики: факты и полезные детали</h2>
<p id="factual-cohort-description" class="muted"></p>
<p id="factual-hidden" class="muted" hidden>Автооценки скрыты для ручного сравнения.</p>
<div id="factual-content">
<div class="table-wrap"><table id="factual-table"></table></div>
<p class="muted">Micro — отношение сумм, не среднее процентов. Выборка здесь может быть
меньше, чем у дополнительных критериев: нужны непустой эталон и корректные цитаты.
Наличие оценки содержания не отменяет ошибку формата генератора.</p>
</div>
</section>
<section aria-labelledby="extra-heading">
<h2 id="extra-heading">Дополнительные автоматические метрики</h2>
<p id="cohort-description" class="muted"></p>
<p id="score-hidden" class="muted">Значения скрыты, чтобы не влиять на ручное сравнение.
Включите «Показать автоматические оценки» для просмотра таблицы и оценок карточек.</p>
<div id="score-content" hidden>
<div class="table-wrap"><table id="score-table"></table></div>
<div id="criterion-notes" class="muted"></div>
</div>
</section>
<section aria-labelledby="technical-heading">
<h2 id="technical-heading">Технические результаты сохранённых прогонов</h2>
<p class="warning">Время — историческая задержка реальных сохранённых запросов, а не новый
замер скорости. Разные условия запуска и ограничения API не позволяют считать это
синхронным сравнением производительности. Ошибки без ответа не входят в статистику
времени и длины; число наблюдений указано отдельно.</p>
<div class="table-wrap"><table id="technical-table"></table></div>
<details>
<summary>Методика, квоты и ограничения замеров</summary>
<div id="technical-notes" class="muted"></div>
<div id="metadata-notes" class="muted"></div>
</details>
</section>
<section aria-labelledby="sample-heading">
<h2 id="sample-heading">Исходный текст и реальные черновики</h2>
<div class="controls">
<label>Категория <select id="category"></select></label>
<label>Вход <select id="sample-select"></select></label>
<button id="previous" type="button">← Предыдущий</button>
<button id="next" type="button">Следующий →</button>
</div>
<p class="muted">Быстрые примеры: первый вход каждой категории, выбранный до оценивания.</p>
<div id="quick-examples" class="quick"></div>
<p id="position" aria-live="polite"></p>
<h3 id="target"></h3>
<p id="sample-id" class="muted"></p>
<p id="reference-caveat" class="warning" hidden>В предварительной автоматической разметке
не найдено фактов для целевой категории. Это повод для ручной проверки исходника,
а не доказанная ошибка метки и не окончательный эталон.</p>
<h3>Исходное объявление — без изменений</h3>
<pre id="source" class="source"></pre>
</section>
<div id="cards" class="cards" aria-label="Черновики четырёх моделей"></div>
<noscript><p class="warning">Для выбора входов и просмотра ответов включите JavaScript в браузере.
Внешние ресурсы не используются.</p></noscript>
</main>
<script id="report-data" type="application/json">__REPORT_JSON__</script>
<script>
"use strict";
const report = JSON.parse(document.getElementById("report-data").textContent);
const byId = id => document.getElementById(id);
const samples = report.samples;
const models = report.models;
const sampleById = new Map(samples.map(sample => [sample.id, sample]));
const modelById = new Map(models.map(model => [model.id, model]));
const number = new Intl.NumberFormat("ru-RU", {maximumFractionDigits: 2});
let selectedId = samples[0].id;
let filtered = samples;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}
function exactText(value) {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}
function valueText(value) {
  return typeof value === "number" && Number.isFinite(value) ? number.format(value) : "нет данных";
}
function modelName(id) {
  const index = models.findIndex(model => model.id === id);
  return byId("blind").checked ? "Модель " + String.fromCharCode(65 + index) : modelById.get(id).name;
}
function metricWithN(value, n, suffix = "") {
  return valueText(value) + (typeof value === "number" ? suffix : "") + " (N=" + valueText(n) + ")";
}
function addRow(parent, values, header = false) {
  const row = element("tr");
  for (const value of values) row.append(element(header ? "th" : "td", value));
  if (header) for (const cell of row.children) cell.scope = "col";
  parent.append(row);
}
function tableParts(table, caption) {
  table.replaceChildren();
  table.append(element("caption", caption));
  const head = element("thead");
  const body = element("tbody");
  table.append(head, body);
  return [head, body];
}
function percentText(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? number.format(value * 100) + " %" : "нет данных";
}
function renderFactualScores() {
  const cohort = report.factual_cohorts[byId("cohort").value];
  byId("factual-cohort-description").textContent = cohort.name + ": N=" +
    valueText(cohort.size) + " одинаковых входов, где обе метрики определены у всех моделей.";
  const visible = byId("show-scores").checked;
  byId("factual-hidden").hidden = visible;
  byId("factual-content").hidden = !visible;
  const [head, body] = tableParts(byId("factual-table"),
    "Сохранённые проверки атомарных фактов; проценты, числители и знаменатели. Не ручной эталон.");
  addRow(head, ["Модель", ...report.factual_criteria.map(metric => metric.name + ", %")], true);
  for (const id of cohort.model_ids) {
    addRow(body, [modelName(id), ...report.factual_criteria.map(metric => {
      const value = cohort.scores[id][metric.id];
      return percentText(value.micro) + " (" + value.numerator + "/" + value.denominator +
        "; N=" + value.macro_defined_count + ")";
    })]);
  }
}
function renderScores() {
  renderFactualScores();
  const cohort = report.cohorts[byId("cohort").value];
  const targets = new Map(samples.map(sample => [String(sample.target_mc_id), sample.target_name]));
  const counts = Object.entries(cohort.target_counts).map(([id, n]) =>
    (targets.get(id) || "Категория " + id) + ": " + valueText(n)).join("; ");
  byId("cohort-description").textContent = cohort.name + ": N=" + valueText(cohort.size) +
    " одинаковых входов для всех моделей этой выборки. Категории: " + counts + ".";
  const visible = byId("show-scores").checked;
  byId("score-hidden").hidden = visible;
  byId("score-content").hidden = !visible;
  const [head, body] = tableParts(byId("score-table"),
    "Среднее автоматического судьи; в каждой ячейке — фактическое N успешных оценок. Итогового балла нет.");
  addRow(head, ["Модель", ...report.criteria.map(criterion => criterion.name + " (0–" + criterion.scale_max + ")")], true);
  for (const id of cohort.model_ids) {
    const scores = cohort.scores[id] || {};
    addRow(body, [modelName(id), ...report.criteria.map(criterion => {
      const score = scores[criterion.id];
      return score ? metricWithN(score.mean, score.n) : "нет данных (N=нет данных)";
    })]);
  }
  const notes = byId("criterion-notes");
  notes.replaceChildren();
  for (const criterion of report.criteria) notes.append(element("p", criterion.name + ": " + criterion.description));
}
function renderTechnical() {
  const [head, body] = tableParts(byId("technical-table"),
    "Последние сохранённые попытки каждого входа, независимо от выбранной выборки. «Нет данных» не означает ноль.");
  addRow(head, ["Модель", "Ответы / входы", "Строгий валидный JSON / полученные ответы",
    "Медиана задержки, с", "95-й процентиль задержки, с", "Медианная длина, символы"], true);
  const notes = byId("technical-notes");
  notes.replaceChildren();
  const uniqueNotes = new Set();
  for (const model of models) {
    const stats = report.technical[model.id];
    const latency = stats.latency_seconds;
    const length = stats.length_chars;
    const ratio = typeof stats.format_valid_given_response === "number"
      ? " (" + number.format(stats.format_valid_given_response * 100) + " %)" : " (доля: нет данных)";
    addRow(body, [modelName(model.id), valueText(stats.responses) + " / " + valueText(stats.inputs),
      valueText(stats.pipeline_valid) + " / " + valueText(stats.responses) + ratio,
      metricWithN(latency.p50, latency.n), metricWithN(latency.p95, latency.n),
      metricWithN(length.median, length.n)]);
    if (stats.note) uniqueNotes.add(stats.note);
  }
  for (const note of uniqueNotes) notes.append(element("p", note));
}
function renderCards(sample) {
  const cards = byId("cards");
  cards.replaceChildren();
  for (const model of models) {
    const answer = sample.models[model.id];
    const card = element("article", null, "card");
    const header = element("div", null, "card-header");
    header.append(element("h3", modelName(model.id)));
    const received = answer.raw_response !== null && answer.raw_response !== undefined;
    const success = answer.generation_status === "success";
    const status = !received ? "Нет ответа модели" : success
      ? "Генерация: успешно, строгий формат принят"
      : "Генерация: ошибка; строгий формат не принят";
    header.append(element("p", status, success && received ? "" : "error"));
    header.append(element("p", "Задержка запроса: " + valueText(answer.latency_seconds) +
      (typeof answer.latency_seconds === "number" ? " с" : ""), "muted"));
    if (answer.content_source === "fenced_json_content_only") {
      header.append(element("p", "Ниже только содержимое JSON из Markdown-обёртки. Наличие текста не отменяет ошибку формата.", "warning"));
    }
    for (const [label, error] of [["Ошибка транспорта: ", answer.transport_error],
      ["Ошибка генерации: ", answer.generation_error]]) {
      if (error !== null && error !== undefined) header.append(element("p", label + exactText(error), "error-detail"));
    }
    card.append(header);
    if (answer.draft_text !== null && answer.draft_text !== undefined) {
      card.append(element("pre", answer.draft_text, "draft"));
    } else {
      card.append(element("p", received
        ? "Черновик недоступен: ответ не удалось преобразовать в содержимое. Оригинальный ответ — ниже."
        : "Ответ модели не получен. Текст черновика отсутствует; ошибка запроса указана выше.", "draft muted"));
    }
    const scores = element("div", null, "scores");
    scores.hidden = !byId("show-scores").checked;
    scores.append(element("strong", "Автоматические оценки — не ручной эталон"));
    for (const criterion of report.criteria) scores.append(element("p", criterion.name + ": " +
      valueText(answer.scores?.[criterion.id]) + " / " + criterion.scale_max));
    const quality = {success: "успешно", failed: "ошибка", unavailable: "недоступно"};
    scores.append(element("p", "Статус автоматической оценки: " + (quality[answer.quality_status] || "неизвестен")));
    const original = answer.original_metrics || {};
    scores.append(element("p", "Подтверждённость фактов: " + percentText(original.factual_support)));
    scores.append(element("p", "Сохранение полезных деталей: " + percentText(original.useful_detail_retention)));
    card.append(scores);
    const details = element("details");
    details.append(element("summary", "Оригинальный ответ модели — целиком"));
    details.append(element("pre", received ? answer.raw_response : "Нет ответа модели. Оригинальный текст отсутствует.", "raw"));
    card.append(details);
    cards.append(card);
  }
}
function renderSample() {
  const sample = sampleById.get(selectedId);
  byId("sample-select").value = selectedId;
  const index = filtered.findIndex(item => item.id === selectedId);
  byId("position").textContent = "Вход " + sample.ordinal + " из " + samples.length +
    "; в выбранной категории: " + (index + 1) + " из " + filtered.length + ".";
  byId("target").textContent = "Цель: " + sample.target_name + " (" + sample.target_mc_id + ")";
  byId("sample-id").textContent = "Идентификатор входа: " + sample.id;
  byId("source").textContent = sample.source;
  byId("reference-caveat").hidden = sample.reference_fact_count !== 0;
  byId("previous").disabled = index <= 0;
  byId("next").disabled = index >= filtered.length - 1;
  renderCards(sample);
}
function filterSamples() {
  const category = byId("category").value;
  filtered = samples.filter(sample => category === "all" || String(sample.target_mc_id) === category);
  const select = byId("sample-select");
  select.replaceChildren();
  for (const sample of filtered) {
    const option = element("option", "№ " + sample.ordinal + " · " + sample.target_name + " · " + sample.id);
    option.value = sample.id;
    select.append(option);
  }
  if (!filtered.some(sample => sample.id === selectedId)) selectedId = filtered[0].id;
  renderSample();
}
const categories = new Map(samples.map(sample => [String(sample.target_mc_id), sample.target_name]));
const allOption = element("option", "Все категории");
allOption.value = "all";
byId("category").append(allOption);
for (const [id, name] of categories) {
  const option = element("option", name + " (" + id + ")");
  option.value = id;
  byId("category").append(option);
}
for (const id of report.manual_example_ids) {
  const sample = sampleById.get(id);
  const button = element("button", "№ " + sample.ordinal + " · " + sample.target_name);
  button.type = "button";
  button.addEventListener("click", () => {
    selectedId = id;
    byId("category").value = "all";
    filterSamples();
  });
  byId("quick-examples").append(button);
}
for (const note of report.metadata.notes) byId("metadata-notes").append(element("p", note));
byId("category").addEventListener("change", filterSamples);
byId("sample-select").addEventListener("change", event => {
  selectedId = event.target.value;
  renderSample();
});
for (const [id, offset] of [["previous", -1], ["next", 1]]) {
  byId(id).addEventListener("click", () => {
    const index = filtered.findIndex(sample => sample.id === selectedId);
    const next = filtered[index + offset];
    if (next) { selectedId = next.id; renderSample(); }
  });
}
byId("cohort").addEventListener("change", renderScores);
byId("show-scores").addEventListener("change", () => { renderScores(); renderSample(); });
byId("blind").addEventListener("change", () => { renderScores(); renderTechnical(); renderSample(); });
renderScores();
renderTechnical();
filterSamples();
</script>
</body>
</html>
"""


def render_report(report: dict) -> str:
    """Встроить согласованный отчёт в автономную страницу без изменения текстов."""
    if not isinstance(report, dict):
        raise ValueError("Отчёт должен быть объектом JSON")
    required = {"metadata", "models", "criteria", "cohorts", "factual_criteria",
                "factual_cohorts", "technical", "manual_example_ids", "samples"}
    missing = required.difference(report)
    if missing:
        raise ValueError("В отчёте отсутствуют поля: " + ", ".join(sorted(missing)))
    if not isinstance(report["samples"], list) or not report["samples"]:
        raise ValueError("В отчёте нет сохранённых входов")
    if not isinstance(report["models"], list) or len(report["models"]) != 4:
        raise ValueError("Для сравнения необходимы четыре модели")
    data = json.dumps(report, ensure_ascii=False, allow_nan=False)
    # Даже ответ с </script> остаётся данными, а не разметкой страницы.
    data = (data.replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    return HTML.replace("__REPORT_JSON__", data, 1)


def main() -> None:
    """Прочитать сохранённый JSON и записать автономный HTML для ручного просмотра."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).with_name("review-data.json"),
                        help="Путь к сохранённому расширенному отчёту JSON")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("review.html"),
                        help="Путь к автономной странице HTML")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Входной JSON и выходной HTML должны иметь разные пути")
        report = json.loads(args.input.read_text(encoding="utf-8"))
        page = render_report(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(page, encoding="utf-8")
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        LOG.error("Не удалось создать отчёт: %s", exc)
        raise SystemExit(1) from exc
    LOG.info("Отчёт для ручного сравнения сохранён: %s", args.output)


if __name__ == "__main__":
    main()
