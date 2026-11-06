@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_netflix.ps1" %*
pause
