"""Korean Wikipedia dump (pages-articles.xml.bz2) -> plain sentences, one per line, for `train-lm`.

Rough wiki-markup stripping (templates, tables, refs, files/categories, link syntax, emphasis), then
sentence splitting on . ! ? followed by whitespace. This is deliberately conservative: list items,
headings and table rows are dropped. Sentences are written as found; `train-lm` applies the project
normalization and excludes lines with digits / Latin / Hanja etc. (they are not converted).

    python scripts/kowiki_to_sentences.py data/raw/kowiki/kowiki-latest-pages-articles.xml.bz2 \
        data/lm/kowiki_sentences.txt --max-sentences 3000000

Licence: Wikipedia text is CC BY-SA 4.0 (record in docs/data_card.md).
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_REF = re.compile(r"<ref[^>/]*?/>|<ref[^>]*?>.*?</ref>", re.S | re.I)
RE_TAG_BLOCK = re.compile(r"<(math|code|pre|syntaxhighlight|source|gallery|timeline|score|chem|hiero)[^>]*>.*?</\1>", re.S | re.I)
RE_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
RE_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
RE_FILE_LINK = re.compile(r"\[\[(?:파일|file|image|그림|분류|category):[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]", re.I)
RE_LINK_PIPE = re.compile(r"\[\[[^\[\]|]*\|([^\[\]]*)\]\]")
RE_LINK = re.compile(r"\[\[([^\[\]]*)\]\]")
RE_EXT_LINK = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\s*([^\]]*)\]")
RE_URL = re.compile(r"https?://\S+")
RE_EMPH = re.compile(r"'{2,}")
RE_SENT = re.compile(r"(?<=[.!?])\s+")
RE_HANGUL = re.compile(r"[가-힣]")


def strip_markup(text: str) -> str:
    text = RE_COMMENT.sub("", text)
    text = RE_REF.sub("", text)
    text = RE_TAG_BLOCK.sub("", text)
    text = RE_TABLE.sub("", text)
    for _ in range(6):  # nested templates, innermost first
        new = RE_TEMPLATE.sub("", text)
        if new == text:
            break
        text = new
    text = RE_FILE_LINK.sub("", text)
    text = RE_LINK_PIPE.sub(r"\1", text)
    text = RE_LINK.sub(r"\1", text)
    text = RE_EXT_LINK.sub(r"\1", text)
    text = RE_URL.sub("", text)
    text = RE_TAG.sub("", text)
    text = RE_EMPH.sub("", text)
    return text


def paragraphs(text: str):
    for line in text.split("\n"):
        line = line.strip()
        if not line or line[0] in "*#:;|!={}" or line.startswith("----"):
            continue
        yield line


def sentences(text: str, min_len: int, max_len: int):
    for para in paragraphs(strip_markup(text)):
        for s in RE_SENT.split(para):
            s = s.strip()
            if not (min_len <= len(s) <= max_len) or s[-1] not in ".!?":
                continue
            if len(RE_HANGUL.findall(s)) < 0.6 * len(s.replace(" ", "")):
                continue
            yield s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("output")
    ap.add_argument("--max-sentences", type=int, default=3_000_000)
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=150)
    args = ap.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_pages = n_sent = n_redirect = 0
    with bz2.open(args.dump, "rb") as f, open(out, "w", encoding="utf-8") as fo:
        ns = None
        for event, elem in ET.iterparse(f, events=("end",)):
            tag = elem.tag
            if ns is None and "}" in tag:
                ns = tag[: tag.index("}") + 1]
            if tag != f"{ns}page":
                continue
            if elem.findtext(f"{ns}ns") == "0":
                if elem.find(f"{ns}redirect") is not None:
                    n_redirect += 1
                else:
                    text = elem.findtext(f"{ns}revision/{ns}text") or ""
                    for s in sentences(text, args.min_len, args.max_len):
                        fo.write(s + "\n")
                        n_sent += 1
                        if n_sent >= args.max_sentences:
                            break
                    n_pages += 1
            elem.clear()
            if n_sent >= args.max_sentences:
                break
            if n_pages and n_pages % 20000 == 0:
                print(f"  {n_pages:,} pages, {n_sent:,} sentences, {time.time() - t0:.0f}s", file=sys.stderr)
    stats = {"dump": str(Path(args.dump).resolve()), "pages": n_pages, "redirects_skipped": n_redirect,
             "sentences_written": n_sent, "max_sentences": args.max_sentences, "min_len": args.min_len,
             "max_len": args.max_len, "seconds": time.time() - t0}
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
