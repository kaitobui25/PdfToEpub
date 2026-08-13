@echo off
setlocal
cd /d "%~dp0\.."

set "PDF=pdf.pdf"
set "START_PAGE=61"
set "END_PAGE=100"

echo ============================================================
echo   PDF to EPUB V4 LOCAL_TURBO - LOCAL ONLY / NO AI
echo   PDF range: %START_PAGE%..%END_PAGE%
echo ============================================================
echo.

if not exist "%PDF%" (
  echo [ERROR] Khong tim thay %CD%\%PDF%
  exit /b 1
)

python -m pdf_to_epub "%PDF%" --start %START_PAGE% --end %END_PAGE%
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" echo [ERROR] Pipeline failed with exit code %RC%.
if "%RC%"=="0" echo [OK] LOCAL_TURBO completed.
exit /b %RC%
