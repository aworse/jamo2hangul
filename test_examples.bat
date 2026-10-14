@echo off
rem 자모->한글 모델 성능 평가: 커스텀 예시로 추론 후 리포트 생성
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (echo [!] .venv 없음 - 먼저 run.bat 로 학습/환경 준비 & pause & exit /b 1)
if not exist "artifacts\run\best.pt" (echo [!] best.pt 없음 - 학습이 아직 평가 체크포인트를 안 만듦 & pause & exit /b 1)
".venv\Scripts\python.exe" eval_examples.py --dir artifacts\run %*
echo.
echo [리포트] artifacts\run\examples_report.txt   (JSON: artifacts\run\examples_report.json)
pause
