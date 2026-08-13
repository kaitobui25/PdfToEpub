"""Minimal EPUB3 writer/validator with no heavyweight EPUB dependency."""

from __future__ import annotations

from html import escape
from pathlib import Path
import re
import uuid
import zipfile
import xml.etree.ElementTree as ET

from .models import BookSide


CSS = """body { font-family: serif; line-height: 1.55; margin: 5%; }
p { margin: 0 0 1em 0; }
h1 { font-size: 1.15em; margin: 0 0 1.2em 0; }
"""


def _xhtml(title: str, paragraphs: list[str]) -> str:
    body = "\n".join(f"<p>{escape(p)}</p>" for p in paragraphs if p.strip())
    return f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="vi" xml:lang="vi">
<head><meta charset="utf-8"/><title>{escape(title)}</title><link rel="stylesheet" type="text/css" href="style.css"/></head>
<body><h1>{escape(title)}</h1>{body}</body>
</html>'''


def write_epub(path: Path, title: str, sides: list[BookSide]) -> None:
    """Write one XHTML spine item per logical book side."""

    path.parent.mkdir(parents=True, exist_ok=True)
    book_id = f"urn:uuid:{uuid.uuid4()}"
    manifest_items: list[str] = []
    spine_items: list[str] = []
    nav_items: list[str] = []
    pages: list[tuple[str, str, str]] = []

    for i, side in enumerate(sides, 1):
        filename = f"p{i:03d}.xhtml"
        item_id = f"p{i}"
        label = side.tag
        manifest_items.append(f'<item id="{item_id}" href="{filename}" media-type="application/xhtml+xml"/>')
        spine_items.append(f'<itemref idref="{item_id}"/>')
        nav_items.append(f'<li><a href="{filename}">{escape(label)}</a></li>')
        pages.append((filename, label, _xhtml(label, side.paragraphs)))

    opf = f'''<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">{book_id}</dc:identifier>
    <dc:title>{escape(title)}</dc:title>
    <dc:language>vi</dc:language>
    <meta property="dcterms:modified">2026-08-13T00:00:00Z</meta>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="css" href="style.css" media-type="text/css"/>
    {''.join(manifest_items)}
  </manifest>
  <spine>{''.join(spine_items)}</spine>
</package>'''

    nav = f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="vi">
<head><meta charset="utf-8"/><title>Navigation</title></head>
<body><nav epub:type="toc" id="toc"><h1>Contents</h1><ol>{''.join(nav_items)}</ol></nav></body>
</html>'''

    container_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="EPUB/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>'''

    with zipfile.ZipFile(path, "w") as zf:
        # EPUB requires an uncompressed mimetype entry first.
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("EPUB/style.css", CSS)
        zf.writestr("EPUB/nav.xhtml", nav)
        zf.writestr("EPUB/content.opf", opf)
        for filename, _, content in pages:
            zf.writestr(f"EPUB/{filename}", content)

    validate_epub(path)


def validate_epub(path: Path) -> None:
    """Catch corrupt ZIP/XML output immediately after writing."""

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if not names or names[0] != "mimetype":
            raise ValueError("EPUB mimetype must be the first ZIP entry")
        if zf.read("mimetype") != b"application/epub+zip":
            raise ValueError("Invalid EPUB mimetype")
        for name in names:
            if name.endswith((".xml", ".opf", ".xhtml")):
                ET.fromstring(zf.read(name))


def parse_local_text(text: str) -> list[tuple[int, str, list[str]]]:
    """Parse the LOCAL_TURBO text representation back into logical sides."""

    marker = re.compile(r"^===== PDF(\d{3})-([LR]) =====$")
    result: list[tuple[int, str, list[str]]] = []
    current_key: tuple[int, str] | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_key
        if current_key is None:
            return
        paragraphs = [chunk.strip() for chunk in "\n".join(buffer).split("\n\n") if chunk.strip()]
        result.append((current_key[0], current_key[1], paragraphs))
        buffer = []

    for raw in text.splitlines():
        match = marker.match(raw.strip())
        if match:
            flush()
            current_key = (int(match.group(1)), match.group(2))
            continue
        if current_key is not None:
            buffer.append(raw)
    flush()
    return result


def sides_from_text(text: str) -> list[BookSide]:
    sides: list[BookSide] = []
    for page_number, side, paragraphs in parse_local_text(text):
        sides.append(BookSide(page_number=page_number, side=side, image_path=Path(), paragraphs=paragraphs))
    return sides
