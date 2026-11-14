"""Сбор описаний исполнителей из публичных каталогов Яндекса."""

import argparse
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

HOST = "uslugi.yandex.ru"
REGIONS = (
    "43-kazan", "213-moscow", "2-saint-petersburg", "65-novosibirsk",
    "54-yekaterinburg", "47-nizhniy-novgorod", "51-samara", "172-ufa",
    "35-krasnodar", "39-rostov-na-donu", "56-chelyabinsk", "66-omsk",
    "62-krasnoyarsk", "50-perm", "193-voronezh", "38-volgograd",
)
CATEGORIES = (
    "remont-i-stroitelstvo--1344",
    "remont-i-stroitelstvo/remont-kvartir-i-domov--1816",
    "remont-i-stroitelstvo/elektromontazhnyie-rabotyi--2007",
    "remont-i-stroitelstvo/santehnicheskie-rabotyi-i-otoplenie--1844",
)
DEFAULT_CATALOGS = [
    f"https://{HOST}/{region}/category/{category}"
    for category in CATEGORIES
    for region in REGIONS
]
STATE = re.compile(r'<script\b[^>]*\bid="__PRELOADED_STATE__"[^>]*>(.*?)</script>', re.S)
PHONE = re.compile(r"(?<!\d)(?:\+7|8)[\s(.-]*\d{3}[\s).-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)


def page_rows(html: str, catalog_url: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Извлекает описание профиля и параметры пагинации."""
    match = STATE.search(html)
    if not match:
        raise ValueError("В каталоге отсутствуют данные страницы")
    state = json.loads(match.group(1))
    search = state["search"]
    workers = state["workers"]["items"]
    rows = []
    for worker_id in search["workerIds"]:
        description = workers[worker_id]["personalInfo"].get("description")
        if isinstance(description, str) and description.strip():
            rows.append({
                "platform": "yandex",
                "provider_id": worker_id,
                "catalog_url": catalog_url,
                "description": EMAIL.sub("[email]", PHONE.sub("[phone]", description)).strip(),
            })
    return rows, search["params"]["pagination"]


def read_rows(path: Path) -> list[dict[str, str]]:
    """Читает записи Яндекса из JSONL."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if any(row.get("platform") != "yandex" for row in rows):
        raise ValueError(f"Файл должен содержать только записи Яндекса: {path}")
    return rows


def collect(
    catalogs: list[str], output: Path, pages: int, minimum_length: int, target: int | None,
) -> None:
    """Загружает новые страницы и продолжает сбор с сохранённого места."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = output.with_suffix(".cache.jsonl")
    checkpoint_path = output.with_suffix(".pages.json")
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    existing = read_rows(output)
    cached = read_rows(cache_path)
    seen_ids = {row["provider_id"] for row in existing}
    seen_text = {" ".join(row["description"].casefold().split()) for row in existing}
    cached_ids = {row["provider_id"] for row in cached}

    with output.open("a", encoding="utf-8") as stream, cache_path.open("a", encoding="utf-8") as cache:
        def accept(row: dict[str, str]) -> None:
            key = " ".join(row["description"].casefold().split())
            provider_id = row["provider_id"]
            if target is not None and len(seen_ids) >= target:
                return
            if provider_id in seen_ids or key in seen_text or len(row["description"]) < minimum_length:
                return
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen_ids.add(provider_id)
            seen_text.add(key)

        for row in cached:
            accept(row)
        stream.flush()
        for catalog in catalogs:
            if target is not None and len(seen_ids) >= target:
                break
            parts = urlsplit(catalog)
            if parts.netloc != HOST or parts.scheme != "https" or "/category/" not in parts.path:
                raise ValueError("Нужна ссылка на публичный каталог Яндекс Исполнителей")
            query = parse_qs(parts.query)
            start = checkpoint.get(catalog, int(query.get("p", ["0"])[0]))
            if start is None:
                continue
            for page in range(start, start + pages):
                query["p"] = [str(page)]
                url = urlunsplit(parts._replace(query=urlencode(query, doseq=True), fragment=""))
                response = subprocess.run(
                    ["curl", "--fail", "--silent", "--show-error", "--max-time", "20", url],
                    check=True, capture_output=True, encoding="utf-8",
                )
                rows, pagination = page_rows(response.stdout, url)
                reset = page >= 10 and pagination["p"] == 0
                if reset:
                    logging.warning("%s: каталог вернул первую страницу, переход к следующему каталогу", url)
                    rows = []
                elif pagination["p"] != page:
                    raise ValueError(f"Запрошена страница {page}, получена {pagination['p']}")
                before = len(seen_ids)
                for row in rows:
                    if row["provider_id"] not in cached_ids:
                        cache.write(json.dumps(row, ensure_ascii=False) + "\n")
                        cached_ids.add(row["provider_id"])
                cache.flush()
                for row in rows:
                    accept(row)
                stream.flush()
                exhausted = reset or (page + 1) * pagination["perPage"] >= pagination["totalItems"]
                checkpoint[catalog] = None if exhausted else page + 1
                temp = checkpoint_path.with_suffix(".tmp")
                temp.write_text(json.dumps(checkpoint, ensure_ascii=False))
                temp.replace(checkpoint_path)
                logging.info("%s, описаний %s, сохранено %s", url, len(rows), len(seen_ids) - before)
                if exhausted or (target is not None and len(seen_ids) >= target):
                    break
                time.sleep(1)
            time.sleep(1)
    logging.info("Всего сохранено %s, в кэше %s", len(seen_ids), len(cached_ids))


def main() -> None:
    """Запускает сбор."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogs", nargs="*", default=DEFAULT_CATALOGS)
    parser.add_argument("--output", type=Path, default=Path("out/yandex/profiles.jsonl"))
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--min-length", type=int, default=300)
    parser.add_argument("--target", type=int)
    args = parser.parse_args()
    if args.pages < 1 or args.min_length < 1 or (args.target is not None and args.target < 1):
        parser.error("--pages, --min-length и --target должны быть положительными")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        collect(args.catalogs, args.output, args.pages, args.min_length, args.target)
    except subprocess.CalledProcessError as error:
        parser.exit(1, error.stderr or f"{error}\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
