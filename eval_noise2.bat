@echo off
set "ROOT=C:\Users\Park Bugeon\Desktop\private project\jamo2hangul"
echo launched> "%ROOT%\artifacts\run_noise\LAUNCHED.txt"
cd /d "%ROOT%"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
(
  echo CWD=%CD%
  echo START %DATE% %TIME%
  "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\eval_examples.py" --dir "%ROOT%\artifacts\run_noise" --ckpt last.pt --device auto
  echo DONE %TIME%
) > "%ROOT%\artifacts\run_noise\run_examples.log" 2>&1
