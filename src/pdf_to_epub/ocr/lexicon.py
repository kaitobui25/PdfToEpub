"""Best-effort access to Tesseract's Vietnamese word DAWG.

The pipeline never downloads dictionaries.  It discovers the local Tesseract
installation, extracts `vie.traineddata` when helper tools are present, and
caches the word list under the run work directory.  OCR still runs if this
optional lexical evidence is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable


def _program(name: str) -> str | None:
    direct = shutil.which(name)
    if direct:
        return direct
    tess = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if tess:
        candidate = Path(tess).with_name(name + (".exe" if os.name == "nt" else ""))
        if candidate.exists():
            return str(candidate)
    return None


def _tessdata_candidates() -> Iterable[Path]:
    configured = os.environ.get("TESSDATA_PREFIX")
    if configured:
        root = Path(configured)
        yield root
        yield root / "tessdata"

    tess = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if tess:
        exe = Path(tess).resolve()
        yield exe.parent / "tessdata"
        yield exe.parent / "share" / "tessdata"
        yield exe.parent.parent / "share" / "tessdata"
        yield exe.parent.parent / "share" / "tesseract-ocr" / "5" / "tessdata"
        yield exe.parent.parent / "share" / "tesseract-ocr" / "4.00" / "tessdata"

    # Common Windows locations when tesseract is not on PATH but is installed.
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        value = os.environ.get(env_name)
        if value:
            yield Path(value) / "Tesseract-OCR" / "tessdata"


def _find_traineddata(language: str) -> Path | None:
    seen: set[Path] = set()
    for directory in _tessdata_candidates():
        try:
            directory = directory.resolve()
        except OSError:
            continue
        if directory in seen:
            continue
        seen.add(directory)
        trained = directory / f"{language}.traineddata"
        if trained.exists():
            return trained
    return None


def load_vietnamese_words(work_dir: Path) -> set[str]:
    cache = work_dir / "vie_lexicon" / "vie.words.txt"
    if cache.exists():
        return {
            line.strip().casefold()
            for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        }

    combine = _program("combine_tessdata")
    dawg2words = _program("dawg2wordlist")
    trained = _find_traineddata("vie")
    if not combine or not dawg2words or trained is None:
        return set()

    out_dir = cache.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "vie."
    try:
        subprocess.run(
            [combine, "-u", str(trained), str(prefix)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        dawg = out_dir / "vie.lstm-word-dawg"
        charset = out_dir / "vie.lstm-unicharset"
        if not dawg.exists() or not charset.exists():
            return set()
        subprocess.run(
            [dawg2words, str(charset), str(dawg), str(cache)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return set()

    if not cache.exists():
        return set()
    return {
        line.strip().casefold()
        for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    }
