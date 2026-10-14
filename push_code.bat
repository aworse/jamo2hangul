@echo off
rem 코드(+성능 리포트)만 GitHub(aworse/jamo2hangul)에 커밋/푸시 + v2.0 태그. 모델 파일은 제외(.gitignore).
set "ROOT=C:\Users\Park Bugeon\Desktop\private project\jamo2hangul"
set "REPO=aworse/jamo2hangul"
set "TAG=v2.0"
cd /d "%ROOT%"
echo launched> "%ROOT%\artifacts\PUSHCODE_LAUNCHED.txt"
> "%ROOT%\push_code.log" 2>&1 (
  echo ===== PUSH CODE START %DATE% %TIME% =====
  echo CWD=%CD%  REPO=%REPO%  TAG=%TAG%

  echo.
  echo --- tool versions / auth ---
  git --version
  gh --version
  gh auth status

  echo.
  echo --- collect result reports (small, text) ---
  if not exist results mkdir results
  copy /y "artifacts\run\examples_report.txt" "results\clean_examples_report.txt"
  copy /y "artifacts\run\word_report.txt" "results\clean_word_report.txt"
  copy /y "artifacts\run\eval.json" "results\clean_eval.json"
  copy /y "artifacts\run_noise\examples_report.txt" "results\noise_examples_report.txt"

  echo.
  echo --- git init + commit (code + reports only; models excluded by .gitignore) ---
  git init -b main
  git add .gitignore README.md *.py *.bat results
  git -c user.name="aworse" -c user.email="aworse@users.noreply.github.com" commit -m "v2.0: jamo->Hangul restoration (code + eval results)"

  echo.
  echo --- create private repo + push (gh); fallback to plain git push ---
  gh repo create %REPO% --private --source . --remote origin --push
  git remote add origin https://github.com/%REPO%.git
  git branch -M main
  git push -u origin main

  echo.
  echo --- tag %TAG% ---
  git tag %TAG%
  git push origin %TAG%

  echo.
  echo ===== PUSH CODE DONE %DATE% %TIME% =====
)
