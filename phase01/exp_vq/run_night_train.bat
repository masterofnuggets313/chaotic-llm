@echo off
REM run_night_train.bat — ночной прогон русской модели (НЕ запускать без команды "спокойной ночи")
REM Данные: yandex 1.3M пар (ru_chat_yandex_full.json)
REM Модель: d=512 L=12 V=2048(BPE) W=512, 100K шагов, batch 64, lr 5e-4, warmup 2000
REM GPU: RTX 3060 12GB. ~19M параметров.

setlocal
set PHASE=C:\Users\Geroin\chaotic-llm\phase01
set VQ=%PHASE%\exp_vq
set PY=C:\Python313\python.exe
set DATA=%VQ%\ru_chat_yandex_full.json
set CKPT=%VQ%\ru_chat_night.pt
set LOG=%VQ%\ru_chat_night.log

echo [1/3] Проверка данных: %DATA%
if not exist "%DATA%" (
  echo  ОШИБКА: данные не собраны! Запустите сначала prep_yandex_chat.py
  pause
  exit /b 1
) else (
  echo  (данные найдены)
)

echo [2/3] Обучение модели: d=512 L=12 W=512 steps=100000 batch=64 lr=5e-4 warmup=2000
if not exist "%CKPT%" (
  %PY% "%VQ%\train_chat.py" --data "%DATA%" --d 512 --layers 12 --window 512 --steps 100000 --batch 64 --lr 5e-4 --warmup 2000 --ckpt "%CKPT%" --log "%LOG%"
) else (
  echo  (чекпоинт уже есть: %CKPT% — обучение пропущено)
)

echo [3/3] Smoke-тест генерации
%PY% "%VQ%\live_chat_yandex.py" --model "%CKPT%" --probe

echo ГОТОВО. Чекпоинт: %CKPT%
endlocal
