"""Best-effort access to Tesseract's Vietnamese word DAWG.

The pipeline never downloads dictionaries. It discovers the local Tesseract
installation, extracts the installed `vie.traineddata`, converts its word DAWG
to a UTF-8 word list, and caches that list under the run work directory.

Windows Tesseract distributions are slightly inconsistent around PATH and
component extraction paths. Discovery therefore uses the traineddata install
root as the strongest executable hint, then PATH/TESSERACT_CMD/common Windows
install locations. Failures are written to a small status file instead of being
silently swallowed.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable


def _exe_name(name: str) -> str:
    return name + (".exe" if os.name == "nt" and not name.lower().endswith(".exe") else "")


def _program(name: str, install_root: Path | None = None) -> str | None:
    """Find a Tesseract companion executable without relying only on PATH."""

    direct = shutil.which(name)
    if direct:
        return direct

    exe_name = _exe_name(name)
    candidates: list[Path] = []

    # Strongest hint: vie.traineddata lives in <install>/tessdata, so its parent
    # is exactly where Windows installers normally place the helper executables.
    if install_root is not None:
        candidates.extend((install_root / exe_name, install_root / "bin" / exe_name))

    configured_tess = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if configured_tess:
        tess_dir = Path(configured_tess).resolve().parent
        candidates.extend((tess_dir / exe_name, tess_dir / "bin" / exe_name))

    # PowerShell/where.exe and Python's shutil.which can disagree on some Windows
    # setups, so probe the standard installation roots explicitly as a fallback.
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        value = os.environ.get(env_name)
        if value:
            root = Path(value) / "Tesseract-OCR"
            candidates.extend((root / exe_name, root / "bin" / exe_name))

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_file():
            return str(resolved)
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


def lexicon_status_path(work_dir: Path) -> Path:
    return work_dir / "vie_lexicon" / "status.txt"


def _write_status(work_dir: Path, lines: list[str]) -> None:
    path = lexicon_status_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _read_words(cache: Path) -> set[str]:
    if not cache.exists() or cache.stat().st_size == 0:
        return set()
    return {
        line.strip().casefold()
        for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    }


def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _component_pairs(out_dir: Path) -> list[tuple[Path, Path, str]]:
    """Return usable (unicharset, word-dawg, label) pairs, LSTM first."""

    files = [path for path in out_dir.iterdir() if path.is_file()]

    def find_suffix(suffix: str) -> list[Path]:
        return sorted((path for path in files if path.name.casefold().endswith(suffix.casefold())), key=lambda p: p.name)

    pairs: list[tuple[Path, Path, str]] = []
    lstm_sets = find_suffix(".lstm-unicharset")
    lstm_dawgs = find_suffix(".lstm-word-dawg")
    if lstm_sets and lstm_dawgs:
        pairs.append((lstm_sets[0], lstm_dawgs[0], "lstm"))

    legacy_sets = [path for path in find_suffix(".unicharset") if not path.name.casefold().endswith(".lstm-unicharset")]
    legacy_dawgs = [path for path in find_suffix(".word-dawg") if not path.name.casefold().endswith(".lstm-word-dawg")]
    if legacy_sets and legacy_dawgs:
        pairs.append((legacy_sets[0], legacy_dawgs[0], "legacy"))
    return pairs


def _extract_components(combine: str, trained: Path, out_dir: Path, status: list[str]) -> list[tuple[Path, Path, str]]:
    # Prefer targeted extraction. It avoids relying on how Windows handles a
    # PATHPREFIX ending in a period for `combine_tessdata -u`.
    lstm_charset = out_dir / "vie.lstm-unicharset"
    lstm_dawg = out_dir / "vie.lstm-word-dawg"
    targeted = _run([combine, "-e", str(trained), str(lstm_charset), str(lstm_dawg)])
    status.append(f"targeted_extract_rc={targeted.returncode}")
    if targeted.stdout.strip():
        status.append("targeted_stdout=" + targeted.stdout.strip().replace("\n", " | "))
    if targeted.stderr.strip():
        status.append("targeted_stderr=" + targeted.stderr.strip().replace("\n", " | "))

    pairs = _component_pairs(out_dir)
    if pairs:
        return pairs

    # Some traineddata variants/distributions behave better with a full unpack.
    # Discover the actual filenames afterwards instead of assuming one prefix.
    prefix = str(out_dir / "vie_unpack") + "."
    unpacked = _run([combine, "-u", str(trained), prefix])
    status.append(f"full_unpack_rc={unpacked.returncode}")
    if unpacked.stdout.strip():
        status.append("unpack_stdout=" + unpacked.stdout.strip().replace("\n", " | "))
    if unpacked.stderr.strip():
        status.append("unpack_stderr=" + unpacked.stderr.strip().replace("\n", " | "))
    return _component_pairs(out_dir)


def load_vietnamese_words(work_dir: Path) -> set[str]:
    out_dir = work_dir / "vie_lexicon"
    cache = out_dir / "vie.words.txt"
    cached = _read_words(cache)
    if cached:
        _write_status(work_dir, ["source=cache", f"words={len(cached)}", f"cache={cache}"])
        return cached
    if cache.exists():
        cache.unlink(missing_ok=True)

    trained = _find_traineddata("vie")
    install_root = trained.parent.parent if trained is not None else None
    combine = _program("combine_tessdata", install_root)
    dawg2words = _program("dawg2wordlist", install_root)
    status = [
        f"install_root={install_root or 'NOT_FOUND'}",
        f"combine_tessdata={combine or 'NOT_FOUND'}",
        f"dawg2wordlist={dawg2words or 'NOT_FOUND'}",
        f"vie_traineddata={trained or 'NOT_FOUND'}",
    ]
    if not combine or not dawg2words or trained is None:
        _write_status(work_dir, status + ["result=unavailable_dependency"])
        return set()

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        pairs = _extract_components(combine, trained, out_dir, status)
        status.append("discovered_pairs=" + ",".join(label for _, _, label in pairs))
        for charset, dawg, label in pairs:
            cache.unlink(missing_ok=True)
            converted = _run([dawg2words, str(charset), str(dawg), str(cache)])
            status.append(f"dawg2wordlist[{label}]_rc={converted.returncode}")
            if converted.stdout.strip():
                status.append(f"dawg2wordlist[{label}]_stdout=" + converted.stdout.strip().replace("\n", " | "))
            if converted.stderr.strip():
                status.append(f"dawg2wordlist[{label}]_stderr=" + converted.stderr.strip().replace("\n", " | "))
            words = _read_words(cache)
            if words:
                status.extend([f"selected_pair={label}", f"words={len(words)}", "result=ok"])
                _write_status(work_dir, status)
                return words
    except (OSError, subprocess.SubprocessError) as exc:
        status.append(f"exception={type(exc).__name__}:{exc}")

    cache.unlink(missing_ok=True)
    status.append("result=no_wordlist")
    _write_status(work_dir, status)
    return set()
