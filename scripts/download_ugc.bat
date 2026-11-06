@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_ugc.ps1" %*
pause
