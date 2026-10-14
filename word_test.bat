@echo off
set "ROOT=C:\Users\Park Bugeon\Desktop\private project\jamo2hangul"
echo launched> "%ROOT%\artifacts\run\WORD_LAUNCHED.txt"
cd /d "%ROOT%"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
(
  echo START %DATE% %TIME%
  "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\word_eval.py" --dir "%ROOT%\artifacts\run" --ckpt best.pt --device auto --n 8000
  echo DONE %TIME%
) > "%ROOT%\artifacts\run\word_log.txt" 2>&1
