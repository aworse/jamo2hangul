@echo off
rem Self-retrying overnight launcher: runs run_j2h.py, retries transient failures, resumes via stage markers.
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=
for %%v in (3.13 3.12 3.11 3.10) do (
  if not defined PY (py -%%v -c "pass" >nul 2>nul && set "PY=py -%%v")
)
if not defined PY (python -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul && set "PY=python")
if not defined PY (echo [!] Python 3.10-3.13 required & pause & exit /b 2)
set /a N=0
:loop
echo ===== run attempt !N! at %date% %time% =====
%PY% run_j2h.py %*
if "%ERRORLEVEL%"=="0" goto done
set /a N+=1
if !N! GEQ 40 goto give
echo [retry !N!/40] a stage failed; resuming in 60s...
timeout /t 60 /nobreak >nul
goto loop
:done
echo [ok] ALL DONE. model: artifacts\run\best.pt
goto end
:give
echo [!] gave up after !N! retries. See artifacts\logs.
:end
echo (window stays open; safe to close)
timeout /t 86400 /nobreak >nul
