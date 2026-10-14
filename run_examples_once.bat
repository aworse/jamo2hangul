@echo off
rem 자동 실행용(일시정지 없음): 커스텀 예시 평가 -> 리포트 + 로그. 끝나면 창 자동 종료.
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" eval_examples.py --dir artifacts\run > "artifacts\run\run_examples.log" 2>&1
