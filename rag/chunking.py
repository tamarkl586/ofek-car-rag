"""Splits documents into chunks. Each chunk keeps its own text and a copy of the document's metadata."""
import re
from dataclasses import dataclass

from rag.loader import Doc


@dataclass
class Chunk:
    id: str            # "<doc id>#<number>", e.g. "of-210-city#2"
    doc_id: str
    text: str          # the original text, exactly as it will be shown to the model
    meta: dict


def chunk_fixed(doc: Doc, size: int = 350, overlap: int = 50) -> list[Chunk]:
    """Baseline: cut every `size` characters, with `overlap` characters repeated between neighbours."""
    chunks = []
    step = size - overlap
    for start in range(0, max(len(doc.text), 1), step):
        piece = doc.text[start:start + size]
        if piece.strip():
            chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, piece, dict(doc.meta)))
        if start + size >= len(doc.text):
            break
    return chunks


def chunk_by_headings(doc: Doc) -> list[Chunk]:
    """One chunk per "## " section.

    - Every chunk starts with a heading path line: "<document title> › <section heading>".
      The text before the first "## " goes in a chunk whose path line is just the title.
    - Skip the "# " title line itself (the path line already carries the title).
    - Keep each section whole, so a table stays together with its header row.
    - Copy the document metadata and add meta["section"] = the section heading.
    - Ids: f"{doc.id}#{n}", numbered from 0.
    """
    head, *sections = re.split(r"^## ", doc.text, flags=re.MULTILINE)
    intro = "\n".join(l for l in head.splitlines() if not l.startswith("# ")).strip()
    parts = [(doc.title, intro)] if intro else []
    for section in sections:
        heading, _, body = section.partition("\n")
        parts.append((heading.strip(), body.strip()))
    chunks = []
    for heading, body in parts:
        path = doc.title if heading == doc.title else f"{doc.title} › {heading}"
        chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, f"{path}\n\n{body}", {**doc.meta, "section": heading}))
    return chunks
