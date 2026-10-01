"""Sentence-aware chunking shared by every knowledge ingestion script.

Fixed-size windows cut words and sentences in half ("nt more output per
employee..."), which hurts both retrieval and what Maya reads back. Here we:
  1. normalise PDF text (join hard-wrapped lines, keep paragraph/bullet breaks),
  2. split into sentences,
  3. pack whole sentences into chunks of ~TARGET chars (never above MAX),
  4. carry the last sentence over as overlap so context isn't lost at edges.
"""

from __future__ import annotations

import re
from pathlib import Path

TARGET_CHARS = 700
MAX_CHARS = 1000
OVERLAP_SENTENCES = 1

_BULLET = re.compile(r"^\s*(?:[•\-–*·]|\d+[.)]|0\d\b)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(•])")


def normalise(text: str) -> list[str]:
    """Return paragraphs: hard-wrapped PDF lines joined, blank lines / bullets kept as breaks."""
    paragraphs: list[str] = []
    current: list[str] = []
    # PDF bullet glyphs often extract as DEL (\x7f); other control chars are noise.
    text = text.replace("\x7f", "•")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        # A bullet or a short heading-like line starts a new paragraph.
        if current and (_BULLET.match(line) or (len(line) < 40 and not line.endswith((".", ",")))):
            paragraphs.append(" ".join(current))
            current = []
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return [re.sub(r"\s+", " ", p).strip() for p in paragraphs if p.strip()]


def _sentences(paragraph: str) -> list[str]:
    parts = _SENTENCE_END.split(paragraph)
    out: list[str] = []
    for s in parts:
        s = s.strip()
        # Hard-split pathological run-ons at word boundaries.
        while len(s) > MAX_CHARS:
            cut = s.rfind(" ", 0, MAX_CHARS)
            cut = cut if cut > 0 else MAX_CHARS
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            out.append(s)
    return out


def chunk_text(text: str, title: str | None = None) -> list[str]:
    """Split document text into sentence-aligned chunks.

    `title` (e.g. the document name) is prefixed to each chunk so every chunk
    stays self-describing when retrieved on its own.
    """
    sentences: list[str] = []
    for para in normalise(text):
        sentences.extend(_sentences(para))

    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for s in sentences:
        if current and size + len(s) + 1 > TARGET_CHARS:
            chunks.append(current)
            current = current[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
            # Don't let the overlap alone push the next chunk over MAX.
            if sum(len(x) + 1 for x in current) + len(s) > MAX_CHARS:
                current = []
            size = sum(len(x) + 1 for x in current)
        current.append(s)
        size += len(s) + 1
    if current:
        chunks.append(current)

    prefix = f"[{title}] " if title else ""
    return [prefix + " ".join(c) for c in chunks]


def title_from_filename(filename: str) -> str:
    """'1790781422631_Zryth_Company_Profile.pdf' -> 'Zryth Company Profile'."""
    stem = Path(filename).stem
    head, _, rest = stem.partition("_")
    if head.isdigit() and rest:
        stem = rest
    return stem.replace("_", " ").strip()
