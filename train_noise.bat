@echo off
rem 노이즈 주입 파인튜닝: 깨끗 학습된 모델을 이어받아 '오류 섞인 자모' 교정을 학습.
rem 기존 clean 모델(artifacts\run)은 그대로 두고 artifacts\run_noise 에서 진행. 중간에 멈춰도 다시 실행하면 이어서 함.
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (echo [!] .venv 없음 & pause & exit /b 1)
if not exist "artifacts\run\best.pt" (echo [!] clean best.pt 없음 - 먼저 학습 완료 필요 & pause & exit /b 1)
if not exist "artifacts\run_noise" mkdir "artifacts\run_noise"
if not exist "artifacts\run_noise\vocab.json"  copy "artifacts\run\vocab.json"  "artifacts\run_noise\" >nul
if not exist "artifacts\run_noise\train.txt"    copy "artifacts\run\train.txt"    "artifacts\run_noise\" >nul
if not exist "artifacts\run_noise\dev.txt"      copy "artifacts\run\dev.txt"      "artifacts\run_noise\" >nul
if not exist "artifacts\run_noise\prepare.json" copy "artifacts\run\prepare.json" "artifacts\run_noise\" >nul
if not exist "artifacts\run_noise\last.pt"      copy "artifacts\run\best.pt"      "artifacts\run_noise\last.pt" >nul
".venv\Scripts\python.exe" train.py train --dir artifacts\run_noise --device cuda --resume --noise 0.05,0.03,0.03 --steps 70000 --warmup 500 --lr 2e-4 --batch-tokens 8000 --eval-every 2500 --layers 6 --d-model 512 --workers 2 > "artifacts\run_noise\train_noise.log" 2>&1
