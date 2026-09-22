@echo off
cd /d "%~dp0"
title Spike Scanner

python --version >nul 2>nul
if errorlevel 1 (
  echo.
  echo  Python is not installed.
  echo  Opening python.org - install it and CHECK "Add Python to PATH".
  echo  Then double-click this file again.
  start https://www.python.org/downloads/
  pause
  exit /b
)

rem skip Streamlit's first-run email question
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  mkdir "%USERPROFILE%\.streamlit" 2>nul
  (echo [general]& echo email = "") > "%USERPROFILE%\.streamlit\credentials.toml"
)

if not exist ".installed" (
  echo  First run: installing components, this takes a minute or two...
  python -m pip install -r requirements.txt
  if errorlevel 1 ( echo Install failed. & pause & exit /b )
  echo ok> .installed
)

echo.
echo  Scanner is starting. Your browser will open shortly.
echo  Keep this window open while you use it. Close it to stop.
echo.
python -m streamlit run app.py
pause
