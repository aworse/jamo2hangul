@echo off
set "ROOT=C:\Users\Park Bugeon\Desktop\private project\jamo2hangul"
set "REPO=aworse/jamo2hangul"
set "TAG=v2.0"
cd /d "%ROOT%"
echo launched> "%ROOT%\artifacts\PUBLISH_LAUNCHED.txt"
> "%ROOT%\publish_github.log" 2>&1 (
  echo ===== PUBLISH START %DATE% %TIME% =====
  echo CWD=%CD%  REPO=%REPO%  TAG=%TAG%

  echo.
  echo --- [1] strip models to fp16 ---
  if not exist models mkdir models
  ".venv\Scripts\python.exe" strip_model.py "artifacts\run\best.pt" "models\jamo2hangul_clean_v2.pt"
  ".venv\Scripts\python.exe" strip_model.py "artifacts\run_noise\last.pt" "models\jamo2hangul_noise_v2.pt"

  echo.
  echo --- [2] collect result reports ---
  if not exist results mkdir results
  copy /y "artifacts\run\examples_report.txt" "results\clean_examples_report.txt"
  copy /y "artifacts\run\word_report.txt" "results\clean_word_report.txt"
  copy /y "artifacts\run\eval.json" "results\clean_eval.json"
  copy /y "artifacts\run_noise\examples_report.txt" "results\noise_examples_report.txt"

  echo.
  echo --- [3] tool versions ---
  git --version
  gh --version
  gh auth status

  echo.
  echo --- [4] git init + commit ---
  git init -b main
  git add .gitignore README.md *.py *.bat results
  git -c user.name="aworse" -c user.email="aworse@users.noreply.github.com" commit -m "v2.0: jamo->Hangul restoration (clean + noise models)"

  echo.
  echo --- [5] create private repo on GitHub and push (gh) ---
  gh repo create %REPO% --private --source . --remote origin --push

  echo.
  echo --- [5b] fallback: ensure remote and push ---
  git remote add origin https://github.com/%REPO%.git
  git branch -M main
  git push -u origin main

  echo.
  echo --- [6] tag %TAG% ---
  git tag %TAG%
  git push origin %TAG%

  echo.
  echo --- [7] Release %TAG% with model assets ---
  gh release create %TAG% "models\jamo2hangul_clean_v2.pt" "models\jamo2hangul_noise_v2.pt" --repo %REPO% --title "jamo2hangul %TAG%" --notes "clean + noise fp16 models. clean word-acc 96.7%% / CER 1.36%%; noise cuts noisy-input CER 32.8%%->14.9%%."

  echo.
  echo ===== PUBLISH DONE %DATE% %TIME% =====
)
