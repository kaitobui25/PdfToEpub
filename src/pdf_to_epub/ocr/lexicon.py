"""Best-effort access to Tesseract's Vietnamese word DAWG.

The pipeline never downloads dictionaries. If Tesseract ships the helper tools
needed to decode `vie.traineddata`, we cache its word list under `_work`; if not,
OCR still runs with evidence-only refinement.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


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


def load_vietnamese_words(work_dir: Path) -> set[str]:
    cache = work_dir / "vie_lexicon" / "vie.words.txt"
    if cache.exists():
        return {line.strip().casefold() for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()}

    combine = _program("combine_tessdata")
    dawg2words = _program("dawg2wordlist")
    tessdata_prefix = os.environ.get("TESSDATA_PREFIX")
    if not combine or not dawg2words or not tessdata_prefix:
        return set()

    trained = Path(tessdata_prefix) / "vie.traineddata"
    if not trained.exists():
        return set()

    out_dir = cache.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "vie."
    try:
        subprocess.run([combine, "-u", str(trained), str(prefix)], check=True, capture_output=True, text=True, timeout=30)
        dawg = out_dir / "vie.lstm-word-dawg"
        charset = out_dir / "vie.lstm-unicharset"
        if not dawg.exists() or not charset.exists():
            return set()
        proc = subprocess.run([dawg2words, str(charset), str(dawg), str(cache)], check=True, capture_output=True, text=True, timeout=30)
        _ = proc
    except (OSError, subprocess.SubprocessError):
        return set()

    if not cache.exists():
        return set()
    return {line.strip().casefold() for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()}
