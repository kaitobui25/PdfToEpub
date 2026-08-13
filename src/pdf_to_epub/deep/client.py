"""OpenCode CLI adapter used to call the free DeepSeek model."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
FENCE_RE = re.compile(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", re.DOTALL | re.IGNORECASE)
MAX_RESPONSE_ATTEMPTS = 3


def find_opencode() -> Path | None:
    direct = shutil.which("opencode")
    if direct:
        return Path(direct)

    appdata = os.environ.get("APPDATA")
    candidates: list[Path] = []
    if appdata:
        npm = Path(appdata) / "npm"
        candidates.extend([npm / "opencode.cmd", npm / "opencode.exe", npm / "opencode.ps1"])
    home = Path.home()
    candidates.extend([
        home / ".opencode" / "bin" / "opencode.exe",
        home / ".opencode" / "bin" / "opencode.cmd",
        home / ".local" / "bin" / "opencode.exe",
        home / ".local" / "bin" / "opencode.cmd",
    ])
    return next((path for path in candidates if path.exists()), None)


def run_exe(
    exe: Path,
    args: list[str],
    timeout: int = 180,
    cwd: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run native/.cmd/.ps1 executables safely on Windows with UTF-8 output."""

    suffix = exe.suffix.casefold()
    if os.name == "nt" and suffix in {".cmd", ".bat"}:
        body = subprocess.list2cmdline([str(exe), *args])
        command = ["cmd.exe", "/d", "/s", "/c", body]
    elif os.name == "nt" and suffix == ".ps1":
        command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(exe), *args]
    else:
        command = [str(exe), *args]

    env = os.environ.copy()
    env.update({"NO_COLOR": "1", "FORCE_COLOR": "0", "PYTHONUTF8": "1"})
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def extract_json(output: str) -> Any:
    """Extract the first valid JSON object/array from noisy CLI output."""

    clean = ANSI_RE.sub("", output)
    fenced = FENCE_RE.search(clean)
    if fenced:
        return json.loads(fenced.group(1))

    decoder = json.JSONDecoder()
    for index, char in enumerate(clean):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(clean[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("OpenCode output did not contain valid JSON")


def _normalize_response(value: Any) -> dict[str, Any]:
    """Require the exact Deep response container so partial nested JSON is never accepted."""

    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
        return {"items": value}
    if not isinstance(value, dict):
        raise ValueError(f"DeepSeek response must be a JSON object or item array, got {type(value).__name__}")
    items = value.get("items")
    if not isinstance(items, list) or not all(isinstance(row, dict) for row in items):
        raise ValueError("DeepSeek response must contain an 'items' array of objects")
    return value


def deepseek_call(
    opencode: Path,
    model: str,
    prompt: str,
    workdir: Path,
    call_name: str,
    timeout: int,
    database_path: Path | None = None,
) -> dict[str, Any]:
    """Call DeepSeek using a UTF-8 prompt file, matching the prior runner.

    ``database_path`` isolates concurrent OpenCode processes from the shared
    SQLite session database. Provider/auth configuration remains shared through
    OpenCode's normal config/auth files.

    Malformed model JSON is transient, so only that individual OpenCode call is
    retried. Successful parallel batches are never repeated by this retry loop.
    """

    workdir = workdir.resolve()
    prompt_dir = workdir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", call_name)
    prompt_file = prompt_dir / f"{safe_name}.txt"
    prompt_file.write_text(prompt, encoding="utf-8", newline="\n")

    env_overrides: dict[str, str] = {}
    if database_path is not None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        env_overrides["OPENCODE_DB"] = str(database_path.resolve())

    command = [
        "run",
        "--model",
        model,
        "--dir",
        str(workdir),
        "Read the attached UTF-8 instruction file and follow it exactly. Return ONLY the requested JSON. Do not use tools.",
        "--file",
        str(prompt_file),
    ]

    last_response_error: ValueError | None = None
    for _attempt in range(1, MAX_RESPONSE_ATTEMPTS + 1):
        cp = run_exe(
            opencode,
            command,
            timeout=timeout,
            cwd=workdir,
            env_overrides=env_overrides,
        )
        combined = (cp.stdout or "") + "\n" + (cp.stderr or "")
        if cp.returncode != 0:
            raise RuntimeError(f"OpenCode call failed ({cp.returncode}): {combined[:4000]}")
        try:
            return _normalize_response(extract_json(combined))
        except ValueError as exc:
            last_response_error = exc

    raise ValueError(
        f"DeepSeek returned invalid JSON after {MAX_RESPONSE_ATTEMPTS} attempts: {last_response_error}"
    ) from last_response_error
