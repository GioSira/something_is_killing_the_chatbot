from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tiktoken
import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# Default chunking strategy - adaptable via yaml
CHUNK_CONFIG = {
    "chunk_size": 400,
    "chunk_overlap": 60,
    "min_chunk_tokens": 20,      # chunk più corti vengono scartati
    "min_section_tokens": 80,    # sezioni più corte si accorpano alla vicina
    "breadcrumb_reserve": 40,    # token lasciati liberi per il breadcrumb
}

# (separatore, nome del metadato): il separatore va per primo
MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

SKIP_DOC_TYPES = {"publication", "hub"}
# Campi del frontmatter che non servono al retrieval: restano nel .md, non sui chunk
DROP_META_KEYS = {"fetched_at", "license", "tables", "revid"}

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
TABLE_LINE = re.compile(r"^\s*\|")
SEPARATOR_LINE = re.compile(r"^\s*\|(\s*:?-{3,}:?\s*\|)+\s*$")
RANGE = re.compile(r"(.+?)\s*#(\d+)(?:\s*[–-]\s*#?(\d+))?", re.DOTALL)


_enc = tiktoken.get_encoding("cl100k_base")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    # il breadcrumb viene aggiunto dopo: senza riserva i chunk superano chunk_size
    chunk_size=CHUNK_CONFIG["chunk_size"] - CHUNK_CONFIG["breadcrumb_reserve"],
    chunk_overlap=CHUNK_CONFIG["chunk_overlap"],
    # "" (taglio per carattere) va per ultimo: ciò che viene dopo non è mai usato
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    keep_separator="end",
)

head_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=MARKDOWN_HEADERS,
    strip_headers=True,
)


def num_tokens(text: str) -> int:
    return len(_enc.encode(text))


def load_markdown(file_path: Path) -> Tuple[Dict[str, Any], str]:
    raw = file_path.read_text(encoding="utf-8")

    if match := FRONTMATTER.match(raw):
        meta = yaml.safe_load(match.group(1)) or {}
        content = raw[match.end():]
    else:
        meta = {}
        content = raw

    return meta, content


def flattening_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    flat_metadata = {}

    def _recursive_flattening(key: str, value: Any):

        if value is None:
            return

        if isinstance(value, (str, bool, int, float)):
            flat_metadata[key] = value

        elif isinstance(value, list):
            if not value:
                return
            if all(isinstance(v, (str, bool, int, float, type(None))) for v in value):
                flat_metadata[key] = json.dumps(value, ensure_ascii=False)
            elif all(isinstance(v, dict) for v in value):
                for i, v in enumerate(value):
                    _recursive_flattening(f"{key}.{i}", v)
            else:
                flat_metadata[key] = str(value)

        elif isinstance(value, dict):
            for k, v in value.items():
                _recursive_flattening(f"{key}.{k}", v)

        else:
            flat_metadata[key] = str(value)

    for key, value in metadata.items():
        if key not in DROP_META_KEYS:
            _recursive_flattening(key, value)

    return flat_metadata


def infobox_text(infobox: Dict[str, Any]) -> str:
    """L'infobox come testo leggibile: "First appearance: SIKTC #1"."""

    lines = []
    for key, value in infobox.items():
        label = key.replace("_", " ").capitalize()
        lines.append(f"{label}: {', '.join(map(str, value)) if isinstance(value, list) else value}")

    return "\n".join(lines)


def publication_range(infobox: Dict[str, Any]) -> Dict[str, Any]:
    src = infobox.get("originally_published_in") or infobox.get("originally_publishied_in")
    if isinstance(src, list):
        src = src[0]
    m = RANGE.match(src or "")
    if not m:
        return {}
    start = int(m.group(2))
    return {"series": m.group(1).strip(), "issue_min": start, "issue_max": int(m.group(3) or start)}


def document_metadata(front: Dict[str, Any]) -> Dict[str, Any]:
    """Metadati comuni a tutti i chunk di una pagina, compreso il range per lo spoiler guard."""
    meta = flattening_metadata(front)
    meta |= publication_range(front.get("infobox") or {})
    return meta


# =========================================== SECTIONS ===============================

def section_path(metadata: Dict[str, Any]) -> str:
    return " > ".join(metadata[k] for k in ("h1", "h2", "h3") if metadata.get(k))


def _render_group(parts: List[Tuple[str, str]]) -> Tuple[str, str]:
    """Unisce sezioni accorpate: breadcrumb = prefisso comune, il resto diventa un'etichetta."""

    if len(parts) == 1:
        return parts[0]

    split_paths = [p.split(" > ") for p, _ in parts]
    common: list[str] = []

    for level in zip(*split_paths):
        if len(set(level)) != 1:
            break
        common.append(level[0])

    pieces = []
    for segments, (_, text) in zip(split_paths, parts):
        rest = " > ".join(segments[len(common):])
        pieces.append(f"**{rest}**\n{text}" if rest else text)

    return " > ".join(common) or parts[0][0], "\n\n".join(pieces)


def load_sections(path: Path) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """
    (frontmatter, [(breadcrumb, testo)]) con scheda infobox e sezioni corte accorpate.

    Usata sia dal chunking sia da parent_sections(), così i parent_id coincidono sempre.
    """
    front, body = load_markdown(path)
    title = front.get("title", path.stem)

    raw: List[Tuple[str, str]] = []
    if front.get("infobox"):
        raw.append((f"{title} > Scheda", infobox_text(front["infobox"])))

    for section in head_splitter.split_text(body):
        if text := section.page_content.strip():
            raw.append((section_path(section.metadata) or title, text))

    min_tokens = CHUNK_CONFIG["min_section_tokens"]
    groups: List[List[Tuple[str, str]]] = []
    sizes: List[int] = []
    has_table: List[bool] = []
    for path_str, text in raw:
        tokens = num_tokens(text)
        table = any(TABLE_LINE.match(line) for line in text.split("\n"))
        # Accorpa se la nuova sezione è corta, oppure se lo è il gruppo precedente
        # (così anche un'introduzione di una riga si unisce alla sezione dopo).
        # Le sezioni con tabelle restano sole: accorpate perderebbero l'intestazione
        # nel breadcrumb, perché la tabella finisce in un chunk separato dal testo.
        if (groups and not table and not has_table[-1]
                and (tokens < min_tokens or sizes[-1] < min_tokens)
                and sizes[-1] + tokens <= CHUNK_CONFIG["chunk_size"]):
            groups[-1].append((path_str, text))
            sizes[-1] += tokens
        else:
            groups.append([(path_str, text)])
            sizes.append(tokens)
            has_table.append(table)

    return front, [_render_group(g) for g in groups]


# =========================================== BLOCKS =================================

def split_blocks(text: str) -> List[Tuple[str, str]]:
    """Divide una sezione in blocchi ("text", ...) e ("table", ...) mantenendo l'ordine."""

    blocks: List[Tuple[str, List[str]]] = []

    for line in (l.rstrip() for l in text.split("\n")):
        kind = "table" if TABLE_LINE.match(line) else "text"
        if blocks and blocks[-1][0] == kind:
            blocks[-1][1].append(line)
        else:
            blocks.append((kind, [line]))

    return [(kind, "\n".join(lines).strip()) for kind, lines in blocks
            if "\n".join(lines).strip()]


def split_table(table: str, budget: int) -> List[str]:
    """Gruppi di righe entro il budget di token, con l'intestazione ripetuta in ognuno."""

    lines = table.split("\n")
    if len(lines) >= 2 and SEPARATOR_LINE.match(lines[1]):
        header, rows = lines[:2], lines[2:]
    else:
        header, rows = [], lines

    header_tokens = num_tokens("\n".join(header))

    pieces, current, used = [], [], header_tokens
    for row in rows:
        cost = num_tokens(row) + 1
        if current and used + cost > budget:
            pieces.append("\n".join(header + current))
            current, used = [], header_tokens
        current.append(row)
        used += cost

    if current:
        pieces.append("\n".join(header + current))

    return pieces


# =========================================== CHUNKING =================================

def chunk_file(path: Path) -> List[Document]:
    """Chunk un singolo file markdown in documenti."""

    front, sections = load_sections(path)
    if front.get("doc_type") in SKIP_DOC_TYPES:
        return []

    clean_metadata = document_metadata(front)
    pageid = front.get("pageid", path.stem)

    chunks: List[Document] = []
    for s_idx, (path_str, text) in enumerate(sections):
        breadcrumb = f"[{path_str}]\n"
        budget = CHUNK_CONFIG["chunk_size"] - num_tokens(breadcrumb)
        parent_id = f"{pageid}:{s_idx}"
        is_infobox = path_str.endswith("> Scheda")

        part = 0   # contatore dei pezzi, non dei blocchi: un blocco può dare più pezzi
        for block_type, block in split_blocks(text):
            pieces = split_table(block, budget) if block_type == "table" else text_splitter.split_text(block)
            for piece in pieces:
                if num_tokens(piece) < CHUNK_CONFIG["min_chunk_tokens"]:
                    continue
                chunks.append(Document(
                    page_content=breadcrumb + piece,
                    metadata=clean_metadata | {
                        "source_path": path.as_posix(),
                        "section_path": path_str,
                        "chunk_kind": "infobox" if is_infobox and block_type == "text" else block_type,
                        "parent_id": parent_id,
                        "chunk_id": f"{parent_id}:{part}",
                        "chunk_tokens": num_tokens(breadcrumb + piece),
                    },
                ))
                part += 1

    return chunks


def chunk_folder(folder: str | Path = "collection/siktc") -> List[Document]:
    docs: List[Document] = []
    for path in sorted(Path(folder).glob("*.md")):
        docs.extend(chunk_file(path))

    return docs


def parent_sections(folder: str | Path = "collection/siktc") -> Dict[str, str]:
    """Testo completo di ogni sezione (dopo l'accorpamento), indicizzato per parent_id."""

    parents: Dict[str, str] = {}

    for path in sorted(Path(folder).glob("*.md")):
        front, sections = load_sections(path)
        if front.get("doc_type") in SKIP_DOC_TYPES:
            continue

        pageid = front.get("pageid", path.stem)
        for s_idx, (path_str, text) in enumerate(sections):
            parents[f"{pageid}:{s_idx}"] = f"[{path_str}]\n{text}"

    return parents


if __name__ == "__main__":

    import sys
    from collections import Counter

    folder = sys.argv[1] if len(sys.argv) > 1 else "collection/siktc"
    docs = chunk_folder(folder)
    sizes = sorted(d.metadata["chunk_tokens"] for d in docs)
    print(f"{len(docs)} chunk da {len({d.metadata['pageid'] for d in docs})} pagine, "
          f"{sum(sizes)} token in totale")
    if not sizes:
        sys.exit()

    ids = Counter(d.metadata["chunk_id"] for d in docs)
    duplicates = [i for i, n in ids.items() if n > 1]
    print(f"chunk_id duplicati: {len(duplicates)}")
    print(f"token: min {sizes[0]}, mediana {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    print("tipi:", dict(Counter(d.metadata["chunk_kind"] for d in docs)))
    print("doc_type:", dict(Counter(d.metadata.get("doc_type") for d in docs)))
    print(f"pagine con range di uscite: "
          f"{len({d.metadata['pageid'] for d in docs if 'issue_min' in d.metadata})}")

    print("\nSezioni con più chunk di tabella:")
    tables = Counter(d.metadata["section_path"].split(" > ")[-1]
                     for d in docs if d.metadata["chunk_kind"] == "table")

    for name, count in tables.most_common(10):
        print(f"  {count:4d}  {name}")

    print("\nI 5 chunk più corti:")
    for d in sorted(docs, key=lambda d: d.metadata["chunk_tokens"])[:5]:
        print(f"  {d.metadata['chunk_tokens']:4d}  {d.page_content[:100]!r}")

    print("\nchunk example")
    print(docs[0].page_content)
    print(f"chunk tokens: {docs[0].metadata['chunk_tokens']}")
