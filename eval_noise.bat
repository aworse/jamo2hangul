@echo off
rem 노이즈 파인튜닝 모델(run_noise) 성능 평가 - 깨끗/오류 입력 리포트 생성
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist "artifacts\run_noise\last.pt" (echo [!] run_noise\last.pt 없음 - 먼저 train_noise.bat 실행 & pause & exit /b 1)
".venv\Scripts\python.exe" eval_examples.py --dir artifacts\run_noise --ckpt last.pt --device auto > "artifacts\run_noise\run_examples.log" 2>&1
echo [리포트] artifacts\run_noise\examples_report.txt
pause
