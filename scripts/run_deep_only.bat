@echo off
setlocal
cd /d "%~dp0\.."

set "PDF=pdf.pdf"
set "START_PAGE=61"
set "END_PAGE=100"
set "MODEL=opencode/deepseek-v4-flash-free"
set "AI_BATCH_SIZE=6"
set "AI_WORKERS=4"

echo ============================================================
echo   PDF to EPUB V4 - DEEP ONLY

echo   OCR/Tesseract will NOT run.
echo   PDF range: %START_PAGE%..%END_PAGE%
echo   Batch: %AI_BATCH_SIZE%   Parallel calls: %AI_WORKERS%
echo ============================================================
echo.

python -m pdf_to_epub "%PDF%" --start %START_PAGE% --end %END_PAGE% --deep-only ^
  --ai-batch-size %AI_BATCH_SIZE% ^
  --ai-workers %AI_WORKERS% ^
  --model "%MODEL%"
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" echo [ERROR] Deep-only failed with exit code %RC%.
if "%RC%"=="0" echo [OK] DEEP_ONLY completed. LOCAL output was not overwritten.
exit /b %RC%
