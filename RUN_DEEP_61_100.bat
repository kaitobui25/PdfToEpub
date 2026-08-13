@echo off
setlocal EnableExtensions

rem ============================================================
rem  One-click DEEP_ONLY runner for the existing LOCAL 61..100
rem  - Never runs OCR/Tesseract.
rem  - Reads the two existing LOCAL files from runs\61_100.
rem  - Writes all Deep outputs back into the same folder.
rem ============================================================

cd /d "%~dp0"

set "RUN_DIR=%CD%\runs\61_100"
set "LOCAL_TXT=%RUN_DIR%\pdf_PDF_61_100_V4_LOCAL_TURBO.txt"
set "LOCAL_AUDIT=%RUN_DIR%\local_refine_audit.json"
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "MODEL=opencode/deepseek-v4-flash-free"
set "AI_BATCH_SIZE=6"
set "AI_WORKERS=4"

echo ============================================================
echo   PDF to EPUB - DEEP ONLY - PDF 61..100
echo ============================================================
echo Project : %CD%
echo Run dir : %RUN_DIR%
echo.

if not exist "%RUN_DIR%" (
    echo [ERROR] Folder does not exist:
    echo         %RUN_DIR%
    echo.
    echo Create it and copy these two files into it:
    echo   pdf_PDF_61_100_V4_LOCAL_TURBO.txt
    echo   local_refine_audit.json
    echo.
    pause
    exit /b 2
)

if not exist "%LOCAL_TXT%" goto :missing_files
if not exist "%LOCAL_AUDIT%" goto :missing_files

if exist "%VENV_PY%" (
    set "PYTHON=%VENV_PY%"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python not found.
        echo Expected venv Python at:
        echo   %VENV_PY%
        echo.
        echo Create the venv first, then run this BAT again.
        pause
        exit /b 3
    )
    set "PYTHON=python"
    echo [WARN] .venv Python was not found. Using Python from PATH.
)

where opencode >nul 2>nul
if errorlevel 1 (
    echo [ERROR] OpenCode CLI is not available on PATH.
    echo Run: opencode --version
    echo and fix OpenCode installation/PATH first.
    pause
    exit /b 4
)

echo [OK] Input TXT   : %LOCAL_TXT%
echo [OK] Input audit : %LOCAL_AUDIT%
echo [OK] Python      : %PYTHON%
echo [OK] Model       : %MODEL%
echo [OK] Batch       : %AI_BATCH_SIZE%
echo [OK] Parallel    : %AI_WORKERS%
echo.
echo Starting Deep-only. OCR/Tesseract will NOT run.
echo.

"%PYTHON%" -m pdf_to_epub pdf.pdf ^
  --start 61 ^
  --end 100 ^
  --output "%RUN_DIR%" ^
  --deep-only ^
  --ai-batch-size %AI_BATCH_SIZE% ^
  --ai-workers %AI_WORKERS% ^
  --model "%MODEL%"

set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
    echo [ERROR] DEEP_ONLY failed. Exit code: %RC%
    echo Check:
    echo   %RUN_DIR%\run_deep_only.log
    echo.
    pause
    exit /b %RC%
)

echo ============================================================
echo   DONE
echo ============================================================
echo All outputs are in:
echo   %RUN_DIR%
echo.
echo Expected Deep outputs:
echo   pdf_PDF_61_100_V4_LOCAL_TURBO_DEEP.txt
echo   pdf_PDF_61_100_V4_LOCAL_TURBO_DEEP.epub
echo   deep_ai_queue.json
echo   deep_ai_audit.json
echo   SUMMARY_DEEP_ONLY.json
echo   run_deep_only.log
echo.
pause
exit /b 0

:missing_files
echo [ERROR] Required input file is missing.
echo.
echo Expected EXACTLY:
echo   %LOCAL_TXT%
echo   %LOCAL_AUDIT%
echo.
echo Files currently found in runs\61_100:
echo ------------------------------------------------------------
dir /b "%RUN_DIR%"
echo ------------------------------------------------------------
echo.
echo Common Windows issue: Explorer may hide extensions, so a file can
echo accidentally be named *.txt.txt or *.json.json.
echo.
pause
exit /b 2
