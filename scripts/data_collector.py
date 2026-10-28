"""Scarica la wiki di Something Is Killing the Children (Fandom) in Markdown strutturato.

Due fasi separate:
  python data_collector.py fetch   -> scarica l'HTML grezzo in collection/raw/ (incrementale per revid)
  python data_collector.py build   -> trasforma il grezzo in .md con frontmatter + tabelle CSV

Così puoi cambiare la pulizia e rigenerare i .md senza riscaricare la wiki.

Dipendenze: pip install requests beautifulsoup4 markdownify pyyaml tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import time
import itertools
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify
from tqdm import tqdm

API = "https://something-is-killing-the-children.fandom.com/api.php"
WIKI = "https://something-is-killing-the-children.fandom.com/wiki/"
# MediaWiki chiede uno User-Agent che identifichi il client e un contatto:
# fingersi un browser rende più probabile il blocco, non meno.
USER_AGENT = "siktc-rag-collector/0.2 (study project; contact: [EMAIL_ADDRESS])"

ROOT = Path("collection2")
RAW_DIR = ROOT / "raw"
MD_DIR = ROOT / "siktc"
TABLE_DIR = ROOT / "tables"

# Elementi da eliminare prima della conversione (selettori CSS).
DROP_SELECTORS = [
    "script", "style", "noscript",
    ".navbox", ".toc", "#toc", ".mw-editsection",
    "sup.reference", ".reference", ".mw-references-wrap", "ol.references",
    ".wikia-gallery", ".gallery", "figure", ".thumb", "img",
    ".noprint", ".mw-empty-elt",
]
# Sezioni (h2) da scartare per intero.
DROP_SECTIONS = {
    "gallery", "videos", "references", "notes", "external links",
    "see also", "promotional images", "navigation",
}
# Righe di rumore residuo da togliere dopo la conversione.
NOISE_LINES = {"advertisement", "see more", "see more →", "tba"}

# Regole per assegnare doc_type dalle categorie: la prima che corrisponde vince.
# Adattale dopo aver guardato le categorie reali della wiki.
DOC_TYPE_RULES = [
    ("adaptation", ("television", "episodes", "season", "film", "cast", "netflix")),
    ("publication", ("publication", "timeline", "collected editions", "trade paperback")),
    ("issue", ("issues", "one-shots", "cross-overs")),
    ("character", ("characters", "slaughter family")),
    ("monster", ("monsters", "creatures")),
    ("organization", ("organizations", "orders", "house of slaughter")),
    ("location", ("locations", "places")),
]

log = logging.getLogger("collector")


# --------------------------------------------------------------------------- client

class ApiError(Exception):
    pass


class FandomClient:
    """Sessione HTTP con rate limit, backoff esponenziale e rispetto di Retry-After/maxlag."""

    def __init__(self, min_interval: float = 0.5, max_retries: int = 6):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = 0.0

    def get(self, **params) -> dict:
        params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

            try:
                res = self.session.get(API, params=params, timeout=20)
            except requests.RequestException as exc:
                log.warning("Errore di rete (%s), tentativo %d/%d", exc, attempt, self.max_retries)
            else:
                if res.status_code == 200:
                    data = res.json()
                    error = data.get("error")
                    if error is None:
                        return data
                    if error.get("code") != "maxlag":
                        raise ApiError(f"{error.get('code')}: {error.get('info')}")
                    log.info("Server sotto carico (maxlag), attendo")
                elif res.status_code in {429, 500, 502, 503, 504}:
                    retry_after = res.headers.get("Retry-After", "")
                    if retry_after.isdigit():
                        delay = max(delay, float(retry_after))
                    log.warning("HTTP %d, tentativo %d/%d", res.status_code, attempt, self.max_retries)
                else:
                    # 400/403/404 non migliorano riprovando
                    raise ApiError(f"HTTP {res.status_code} per {params.get('action')} {params}")

            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 60)

        raise ApiError(f"Troppi tentativi falliti per {params}")

    def list_articles(self):
        """Tutte le pagine del namespace principale, senza redirect, con l'ultimo revid."""
        cont: dict = {}
        while True:
            data = self.get(
                action="query", generator="allpages", gapnamespace=0,
                gapfilterredir="nonredirects", gaplimit="max", prop="info", **cont,
            )
            for page in data.get("query", {}).get("pages", []):
                yield page["pageid"], page["title"], page["lastrevid"]
            if "continue" not in data:
                break
            cont = data["continue"]

    def main_page_title(self) -> str:
        data = self.get(action="query", meta="siteinfo", siprop="general")
        return data["query"]["general"]["mainpage"]

    def parse(self, pageid: int) -> dict:
        data = self.get(
            action="parse", pageid=pageid,
            prop="text|categories|revid|displaytitle",
            disableeditsection=1, disabletoc=1, disablelimitreport=1,
        )
        return data["parse"]


# --------------------------------------------------------------------------- fetch

def fetch(limit: int | None = None) -> None:
    client = FandomClient()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    main_page = client.main_page_title()

    articles = [a for a in client.list_articles() if a[1] != main_page]
    if limit:
        articles = articles[:limit]

    skipped = failed = 0
    for pageid, title, revid in tqdm(articles, desc="Pagine"):
        raw_path = RAW_DIR / f"{pageid}.json"
        if raw_path.exists():
            cached = json.loads(raw_path.read_text(encoding="utf-8"))
            if cached.get("revid") == revid:
                skipped += 1
                continue
        try:
            parsed = client.parse(pageid)
        except ApiError as exc:
            failed += 1
            log.error("Pagina %r saltata: %s", title, exc)
            continue

        record = {
            "pageid": pageid,
            "title": title,
            "revid": parsed["revid"],
            "categories": [c["category"].replace("_", " ")
                           for c in parsed.get("categories", []) if not c.get("hidden")],
            "html": parsed["text"],
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        raw_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    log.info("Fetch completato: %d pagine, %d invariate, %d fallite",
             len(articles), skipped, failed)


# --------------------------------------------------------------------------- transform

def slugify(text: str, sep: str = "-") -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[\s_-]+", sep, text)


def extract_infobox(soup: BeautifulSoup) -> dict:
    """Converte l'infobox di Fandom in un dizionario e la rimuove dal corpo."""
    box = soup.select_one("aside.portable-infobox")
    if box is None:
        return {}
    data: dict = {}
    for item in box.select(".pi-data"):
        label = item.select_one(".pi-data-label")
        value = item.select_one(".pi-data-value")
        if label is None or value is None:
            continue
        for br in value.find_all("br"):
            br.replace_with("\n")
        if values := [
            v.strip()
            for v in value.get_text("").split("\n")
            if v.strip() and v.strip().upper() not in {"TBA", "N/A", "UNKNOWN"}
        ]:
            data[slugify(label.get_text(" ", strip=True), "_")] = values if len(values) > 1 else values[0]
    box.decompose()
    return data


def drop_sections(soup: BeautifulSoup) -> None:
    for h2 in list(soup.find_all("h2")):
        if h2.get_text(" ", strip=True).lower() not in DROP_SECTIONS:
            continue
        # MediaWiki recenti avvolgono l'intestazione in <div class="mw-heading">
        start = h2.parent if "mw-heading" in (h2.parent.get("class") or []) else h2
        doomed = [start]
        for sib in start.next_siblings:
            if isinstance(sib, Tag) and (sib.name == "h2" or sib.find("h2")):
                break
            doomed.append(sib)
        for node in doomed:
            node.extract()


def table_rows(table: Tag) -> list[list[str]]:
    """Righe di una tabella HTML con rowspan/colspan espansi (es. le date nella timeline)."""
    grid: dict[tuple[int, int], str] = {}
    for r, tr in enumerate(table.find_all("tr")):
        c = 0
        for cell in tr.find_all(["th", "td"], recursive=False):
            while (r, c) in grid:
                c += 1
            text = cell.get_text(" ", strip=True)
            try:
                rowspan = int(cell.get("rowspan", 1))
                colspan = int(cell.get("colspan", 1))
            except ValueError:
                rowspan = colspan = 1
            for dr, dc in itertools.product(range(rowspan), range(colspan)):
                grid[(r + dr, c + dc)] = text
            c += colspan
    if not grid:
        return []
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    rows = [[grid.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]
    return [row for row in rows if any(row)]


def markdown_table(rows: list[list[str]]) -> str:
    def esc(s: str) -> str:
        return s.replace("|", "\\|")
    head, *body = rows
    lines = ["| " + " | ".join(map(esc, head)) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join(map(esc, row)) + " |" for row in body]
    return "\n".join(lines)


HEADING = re.compile(r"^(#{2,6}) ")


def drop_empty_sections(md: str) -> str:
    """Toglie le intestazioni senza contenuto (ma tiene quelle che hanno solo sottosezioni)."""
    lines = md.split("\n")
    changed = True
    while changed:
        changed = False
        idx = [i for i, line in enumerate(lines) if HEADING.match(line)]
        for k, i in enumerate(idx):
            j = idx[k + 1] if k + 1 < len(idx) else len(lines)
            body = [l for l in lines[i + 1:j] if l.strip()]
            level = len(HEADING.match(lines[i]).group(1))
            next_level = len(HEADING.match(lines[j]).group(1)) if j < len(lines) else 0
            if not body and next_level <= level:
                del lines[i:j]
                changed = True
                break
    return "\n".join(lines)


def doc_type_for(categories: list[str]) -> str:
    lowered = [c.lower() for c in categories]
    return next(
        (doc_type for doc_type, keywords in DOC_TYPE_RULES
         if any(k in c for c in lowered for k in keywords)),
        "lore",
    )


def html_to_markdown(html: str, stem: str) -> tuple[str, dict, int]:
    soup = BeautifulSoup(html, "html.parser")
    infobox = extract_infobox(soup)
    for selector in DROP_SELECTORS:
        for node in soup.select(selector):
            node.decompose()
    drop_sections(soup)

    # Tabelle: CSV a parte (dati strutturati) + tabella markdown al loro posto.
    tables: dict[str, str] = {}
    for n, table in enumerate(soup.find_all("table")):
        rows = table_rows(table)
        if not rows:
            table.decompose()
            continue
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        with open(TABLE_DIR / f"{stem}--{n}.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        token = f"QQTABLE{n}QQ"
        tables[token] = markdown_table(rows)
        table.replace_with(soup.new_string(f"\n\n{token}\n\n"))

    md = markdownify(str(soup), heading_style="ATX", bullets="-", strip=["a", "span"])
    for token, table_md in tables.items():
        md = md.replace(token, table_md)

    lines = [line.rstrip() for line in md.split("\n")]
    lines = [l for l in lines if l.strip().lower() not in NOISE_LINES]
    md = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    md = drop_empty_sections(md)
    return md, infobox, len(tables)


def build() -> None:
    MD_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(RAW_DIR.glob("*.json"))
    for raw_path in tqdm(raw_files, desc="Markdown"):
        record = json.loads(raw_path.read_text(encoding="utf-8"))
        stem = f"{slugify(record['title'])}--{record['pageid']}"   # pageid evita collisioni
        body, infobox, n_tables = html_to_markdown(record["html"], stem)

        if len(body) < 200 and not infobox:
            log.info("Pagina quasi vuota, saltata: %s", record["title"])
            continue

        front = {
            "title": record["title"],
            "pageid": record["pageid"],
            "revid": record["revid"],
            "source_url": WIKI + record["title"].replace(" ", "_"),
            "license": "CC BY-SA 3.0",
            "fetched_at": record["fetched_at"],
            "doc_type": doc_type_for(record["categories"]),
            "categories": record["categories"],
            "infobox": infobox or None,
            "tables": n_tables,
            # Da riempire in seguito dalla tabella di pubblicazione (spoiler guard):
            "issue_min": None,
            "issue_max": None,
        }
        front = {k: v for k, v in front.items() if v not in (None, [], {})} | {
            "issue_min": None, "issue_max": None}
        frontmatter = yaml.safe_dump(front, allow_unicode=True, sort_keys=False)
        (MD_DIR / f"{stem}.md").write_text(
            f"---\n{frontmatter}---\n\n# {record['title']}\n\n{body}\n", encoding="utf-8")


# --------------------------------------------------------------------------- cli

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_fetch = sub.add_parser("fetch", help="scarica l'HTML grezzo (incrementale)")
    p_fetch.add_argument("--limit", type=int, help="solo le prime N pagine, per provare")
    sub.add_parser("build", help="genera i .md dal grezzo")
    args = parser.parse_args()

    if args.cmd == "fetch":
        fetch(args.limit)
    else:
        build()
