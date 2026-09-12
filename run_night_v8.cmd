@echo off
REM ============================================================================
REM  Ночной прогон v8 (Fracode / STS-Prog). ЕДИНСТВЕННЫЙ правильный лаунчер.
REM
REM  ПОЧЕМУ НЕ run_v8_night.bat (phase01\exp_vq\): тот зовёт C:\Python313\python.exe,
REM  где НЕТ numpy / tokenizers / fastapi / uvicorn (torch есть) -> прогон умирает
REM  сразу после строки "corpus ... chars", без внятной ошибки. Здесь используется
REM  hermes-venv, где torch(cuda)+tokenizers+numpy+fastapi+uvicorn на месте.
REM
REM  Использование:
REM     run_night_v8.cmd                 - 40000 шагов (полная рецептура)
REM     run_night_v8.cmd 10000           - 10000 шагов
REM     run_night_v8.cmd 40000 --resume  - продолжить с полного ckpt
REM
REM  Пайплайн сам сначала прогоняет предполётный гейт и не стартует, если что-то
REM  не так. Каждый прогон пишет всё в frakod\runs\<timestamp>\.
REM ============================================================================
setlocal
set "PY=%LOCALAPPDATA%\hermes\hermes-agent\venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [FAIL] не найден интерпретатор: "%PY%"
  echo        нужен python с torch(cuda)+tokenizers+numpy+fastapi+uvicorn
  echo        (C:\Python313 НЕ подходит: там нет numpy/tokenizers/fastapi/uvicorn)
  exit /b 3
)

set "STEPS=%~1"
if "%STEPS%"=="" set "STEPS=40000"
shift

set "FRK=%~dp0"
if not exist "%FRK%night_v8_pipeline.py" set "FRK=%CD%\frakod\"

echo === Ночной v8: %STEPS% шагов ===
echo python: %PY%
echo пайплайн: %FRK%night_v8_pipeline.py
echo логи прогона: %FRK%runs\^<timestamp^>\
echo.

"%PY%" -u "%FRK%night_v8_pipeline.py" --steps %STEPS% --batch 48 %1 %2 %3 %4
set RC=%ERRORLEVEL%
echo.
echo === rc=%RC% ===
echo последний отчёт: %FRK%night_v8_report.json
exit /b %RC%
