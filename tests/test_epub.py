from pathlib import Path
import zipfile

from pdf_to_epub.epub import sides_from_text, validate_epub, write_epub


def test_epub_round_trip(tmp_path: Path) -> None:
    text = "===== PDF001-L =====\n\nĐoạn một.\n\nĐoạn hai.\n\n\n===== PDF001-R =====\n\nTrang phải.\n"
    sides = sides_from_text(text)
    output = tmp_path / "book.epub"
    write_epub(output, "Test", sides)
    validate_epub(output)

    with zipfile.ZipFile(output) as zf:
        assert zf.namelist()[0] == "mimetype"
        assert zf.read("mimetype") == b"application/epub+zip"
        assert "EPUB/p001.xhtml" in zf.namelist()
        assert "EPUB/p002.xhtml" in zf.namelist()
