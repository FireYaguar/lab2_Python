"""Boolean search over the inverted index.

    python -m findex.search index.bin "query" [--engine merge|set] [--limit 20]

Query syntax (evaluated left to right; Lab 3 adds a real parser):
    a b         a AND b (implicit)
    a OR b      union
    a NOT b     a AND NOT b;  "NOT b" alone = all documents except b
Operators are UPPERCASE words; everything else goes through the same tokenizer as the corpus.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from findex.index import InvertedIndex
from findex.tokenize import tokenize

type Engine = Literal["merge", "set"]


# --- two-pointer merges on sorted doc_id lists: O(len(a) + len(b)), no hashing ---


def merge_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def merge_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_not(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """a AND NOT b."""
    out: list[int] = []
    i = j = 0
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


# --- set-based alternative: simpler, O(len(a) + len(b)) hashing + a sort of the result ---


def set_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) & set(b))


def set_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) | set(b))


def set_not(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) - set(b))


type Op = Callable[[Sequence[int], Sequence[int]], list[int]]
ENGINES: dict[str, tuple[Op, Op, Op]] = {
    "merge": (merge_and, merge_or, merge_not),
    "set": (set_and, set_or, set_not),
}


def doc_ids(index: InvertedIndex, term: str) -> list[int]:
    return [p.doc_id for p in index.postings.get(term, ())]


def search(index: InvertedIndex, query: str, engine: Engine = "merge") -> list[int]:
    op_and, op_or, op_not = ENGINES[engine]
    all_docs = sorted(index.doc_lengths)
    result: list[int] | None = None
    op = "AND"
    negate = False
    for word in query.split():
        if word in ("AND", "OR"):
            op = word
            continue
        if word == "NOT":
            negate = True
            continue
        ids: list[int] | None = None
        for term in tokenize(word):  # "well-known" -> AND of its tokens
            t_ids = doc_ids(index, term)
            ids = t_ids if ids is None else op_and(ids, t_ids)
        if ids is None:  # word had no tokens (pure punctuation)
            continue
        if negate:
            if result is None or op == "OR":
                ids = op_not(all_docs, ids)
            else:
                result, ids = op_not(result, ids), None
        if ids is not None:
            if result is None:
                result = ids
            elif op == "OR":
                result = op_or(result, ids)
            else:
                result = op_and(result, ids)
        op, negate = "AND", False
    return result or []


def main(argv: list[str] | None = None) -> None:
    from findex.stats import measure
    from findex.store import load

    ap = argparse.ArgumentParser(prog="python -m findex.search", description=__doc__)
    ap.add_argument("index", type=Path)
    ap.add_argument("query")
    ap.add_argument("--engine", choices=["merge", "set"], default="merge")
    ap.add_argument("--limit", type=int, default=20, help="how many titles to print")
    args = ap.parse_args(argv)

    index, t_load, peak_load = measure(lambda: load(args.index))
    hits, t_search, _ = measure(lambda: search(index, args.query, args.engine))
    for doc_id in hits[: args.limit]:
        meta = index.doc_meta[doc_id]
        print(f"{doc_id:>5}  {meta.title[:60]:<60}  {meta.path}")
    print(f"\n{len(hits):,} documents match ({args.engine})")
    print(f"load:   {t_load:.2f} s, peak memory {peak_load / 2**20:.1f} MiB (tracemalloc)")
    print(f"search: {t_search * 1000:.2f} ms")


if __name__ == "__main__":
    main()
