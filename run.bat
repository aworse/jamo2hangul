@echo off
rem jamo->Hangul training launcher (Windows). Resumable: double-click again to continue after any stop.
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=
for %%v in (3.13 3.12 3.11 3.10) do (
  if not defined PY (py -%%v -c "pass" >nul 2>nul && set "PY=py -%%v")
)
if not defined PY (python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul && set "PY=python")
if not defined PY (
  echo [!] Python 3.10-3.13 required: https://www.python.org/downloads/
  pause & exit /b 2
)
%PY% run_j2h.py %*
set RC=%ERRORLEVEL%
echo.
if "%RC%"=="0" (echo [ok] done. model: artifacts\run\best.pt) else (echo [!] stopped code %RC%. Double-click again to resume.)
pause
exit /b %RC%
