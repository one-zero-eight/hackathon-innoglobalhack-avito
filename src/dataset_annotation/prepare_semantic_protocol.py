"""Зафиксировать группы разработки/подтверждения без выбора по private-меткам."""

import csv
import hashlib
import json
import logging
from pathlib import Path
from random import Random
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from database import read_records, table_sha256
from main import MICROCATEGORIES
from run import normalize_description
from subcategory_extraction.lexical.snippet_logistic_regression.train import duplicate_groups

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "src/subcategory_extraction/benchmarks/classifier_methods/jev-semantic-split.json"
SEED = 20261115


def workbook_rows(path: Path, sheet: int) -> list[dict[str, str]]:
    """Прочитать ячейки XLSX без изменения исходного файла и стороннего Excel-движка."""
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        strings = (["".join(item.itertext()) for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
                   if "xl/sharedStrings.xml" in archive.namelist() else [])
        rows = []
        for row in ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet}.xml")).findall("s:sheetData/s:row", namespace):
            values = {}
            for cell in row.findall("s:c", namespace):
                value = cell.find("s:v", namespace)
                inline = cell.find("s:is", namespace)
                text = value.text if value is not None else "".join(inline.itertext()) if inline is not None else ""
                values["".join(char for char in cell.attrib["r"] if char.isalpha())] = (
                    strings[int(text)] if cell.attrib.get("t") == "s" else text
                )
            rows.append(values)
    return rows


def prepare() -> dict:
    """Проверить эквивалентность labels, затем делить только по текстам и группам."""
    private = read_records("private_test")
    training = read_records("manual_train")
    public = read_records("case_test") + read_records("manual_test")
    workbook = ROOT / "data/mcId_razmetka.xlsx"
    workbook_labels = {row["A"]: sorted(int(value) for value in row.get("C", "").split(",") if value.strip().isdigit())
                       for row in workbook_rows(workbook, 1)[1:]}
    for row in private:
        ids = sorted(item["mcId"] for item in row["found_microcategories"])
        if workbook_labels.get(row["_extra"]["testcase"]) != ids or set(ids) - set(MICROCATEGORIES):
            raise ValueError("Private labels не совпали с workbook")
    combined = training + public + private
    groups, grouping = duplicate_groups([row["description"] for row in combined])
    train_groups = groups[:len(training)]
    public_groups = set(map(int, groups[len(training):len(training) + len(public)]))
    private_groups = groups[len(training) + len(public):]
    all_private = set(map(int, private_groups))
    # Ранее прочитанные вступительные примеры/пояснения исключены из confirmation.
    previewed = {f"case-{index:03}" for index in range(1, 41)} | {"case-064"}
    protected = {int(group) for group, row in zip(private_groups, private, strict=True)
                 if int(group) in set(map(int, train_groups)) | public_groups or row["_extra"]["testcase"] in previewed}
    eligible = sorted(all_private - protected)
    Random(SEED).shuffle(eligible)
    holdout = set(eligible[:max(1, round(len(eligible) * 0.30))])
    representatives = {}
    for row, group in zip(training, train_groups, strict=True):
        if int(group) not in all_private | public_groups:
            representatives.setdefault(int(group), row)
    eligible_train = sorted(representatives)
    Random(SEED).shuffle(eligible_train)
    if len(eligible_train) < 144:
        raise ValueError("Недостаточно независимых train-групп")
    report = {
        "seed": SEED, "grouping": grouping,
        "table_sha256": {name: table_sha256(name) for name in ("manual_train", "private_test", "case_test", "manual_test")},
        "workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
        "workbook_labels_equal_private_snapshot": True,
        "selection_limit": {"native_prompts": ["task", "strict"], "structured_arbiter": 1},
        "private_labels_fit_thresholds": False, "private_confirmation_opened": False,
        "prior_exposure": "baseline full-set aggregates were already measured; confirmation is not a pristine external test",
        "private": [{"row_id": row["_row_id"], "testcase": row["_extra"]["testcase"], "group": int(group),
                     "description_sha256": hashlib.sha256(row["description"].encode()).hexdigest(),
                     "split": "confirmation" if int(group) in holdout else "development"}
                    for row, group in zip(private, private_groups, strict=True)],
        "training": {name: [{"row_id": representatives[group]["_row_id"], "group": group,
                              "description_sha256": hashlib.sha256(representatives[group]["description"].encode()).hexdigest()}
                             for group in selected]
                     for name, selected in (("fit", eligible_train[:96]), ("control", eligible_train[96:144]))},
        "protected_private_groups": len(protected),
    }
    sets = [set(item["group"] for item in report["private"] if item["split"] == split)
            for split in ("development", "confirmation")]
    sets += [set(item["group"] for item in report["training"][split]) for split in ("fit", "control")]
    if sum(map(len, sets)) != len(set.union(*sets)):
        raise ValueError("Группы пересекают части протокола")
    return report


def main() -> None:
    """Создать манифест один раз; существующий протокол не перерисовывается."""
    if OUTPUT.exists():
        raise FileExistsError("Протокол уже зафиксирован; существующие split не меняются")
    report = prepare()
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    logging.info("Private split: %s", {part: sum(row["split"] == part for row in report["private"])
                                      for part in ("development", "confirmation")})
    logging.info("Manual train: 96 fit / 48 control; excluded private and public-test groups")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
