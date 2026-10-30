from typing import Any, Sequence, Iterator, Dict, Hashable, Mapping, List, Tuple
from typing import TypeVar, Optional

import json
import re

UNKNOWN_ISSUE = -1

T = TypeVar("T")

_FIELD_NAME = re.compile(r"[^0-9A-Za-z_]")



# ============================== HELPER ==============================

def _hit_key(item: Any) -> str:
    return str(_read(item, "id", "_id"))


def _read(obj: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def _batches(items: Sequence[Any], size: int = 16) -> Iterator[Sequence[Any]]:

    for start in range(0, len(items), size):
        yield items[start:start + size]


def _reciprocal_rank_fusion(rankings: Sequence[Sequence[Any]], k: int = 60) -> List[Tuple[Any, float]]:
    """Fonde per id: lo stesso chunk trovato in più liste somma i contributi."""

    scores: Dict[str, float] = {}
    best: Dict[str, Any] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            best.setdefault(hit.id, hit)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(best[i], score) for i, score in ordered]


# ============================== RECORD ==============================

def _field_value(value: Any) -> Any:
    """Pinecone accetta solo stringhe, numeri, booleani e liste di stringhe."""

    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value

        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value

    return None


def to_pinecone_record(record: Dict[str, Any], vector_key: str) -> Dict[str, Any]:
    """Record del JSONL -> documento Pinecone: metadati al primo livello, nomi di campo puliti."""
    
    fields: Dict[str, Any] = {}
    
    for key, value in record.get("metadata", {}).items():
        value = _field_value(value)

        if value not in (None, "", []):
            # "infobox.written_by" -> "infobox_written_by": il punto nei nomi dei campi
            # è ambiguo nei filtri
            fields[_FIELD_NAME.sub("_", key)] = value

    # Sentinella per lo spoiler guard: un filtro su issue_max escluderebbe i documenti
    # senza il campo, quindi tutte le pagine senza range.
    fields.setdefault("issue_min", UNKNOWN_ISSUE)
    fields.setdefault("issue_max", UNKNOWN_ISSUE)

    return {
        **fields,
        "_id": record["id"],
        "page_content": record["page_content"],
        "summary": record.get("summary", ""),
        "embedding": [float(x) for x in record[vector_key]],
    }


def spoiler_filter(progress: Dict[str, int], 
                   strict: bool = True,
                   allow_unknown: bool = True) -> Optional[Dict[str, Any]]:

    """
    Filtro Pinecone per lo spoiler guard.

    progress: {"Something Is Killing the Children": 20, "House of Slaughter": 5}
    strict=True: il chunk passa solo se l'utente ha finito il suo arco (issue_max).
    """

    bound = "issue_max" if strict else "issue_min"

    clauses: List[Dict[str, Any]] = [
        {"$and": [{"series": {"$eq": series}},
                  {bound: {"$lte": last}},
                  {"issue_min": {"$gte": 0}}]}
        for series, last in progress.items()
    ]

    if allow_unknown:
        clauses.append({"issue_min": {"$eq": UNKNOWN_ISSUE}})

    if not clauses:
        return None

    return clauses[0] if len(clauses) == 1 else {"$or": clauses}