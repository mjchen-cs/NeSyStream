@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
REM Download UVG 1080p 8-bit 4:2:0 raw sequences (7z) into ..\data\uvg_raw, then extract them.
REM Resumable: run again if interrupted. Requires 7-Zip (https://www.7-zip.org) for extraction.
cd /d "%~dp0.."
if not exist data\uvg_raw mkdir data\uvg_raw
cd data\uvg_raw
set SEVENZ=
if exist "C:\Program Files\7-Zip\7z.exe" set SEVENZ="C:\Program Files\7-Zip\7z.exe"

echo === Beauty
set OK=0
for %%U in ("https://ultravideo.fi/video/Beauty_1920x1080_120fps_420_8bit_YUV_RAW.7z" "https://ultravideo.fi/video/Beauty_1920x1080_120fps_420_8bit_YUV.7z") do (
  if !OK!==0 (
    curl.exe -f -L -C - --retry 5 -o Beauty.7z %%U && set OK=1
  )
)
if !OK!==0 (echo [failed] Beauty: open https://ultravideo.fi/dataset.html, right-click its 1080p 8-bit RAW link, copy the address and download it manually) else (
  if defined SEVENZ (!SEVENZ! x -y Beauty.7z >nul && echo extracted Beauty) else (tar.exe -xf Beauty.7z && echo extracted Beauty)
)

echo === Bosphorus
set OK=0
for %%U in ("https://ultravideo.fi/video/Bosphorus_1920x1080_120fps_420_8bit_YUV_RAW.7z" "https://ultravideo.fi/video/Bosphorus_1920x1080_120fps_420_8bit_YUV.7z") do (
  if !OK!==0 (
    curl.exe -f -L -C - --retry 5 -o Bosphorus.7z %%U && set OK=1
  )
)
if !OK!==0 (echo [failed] Bosphorus: open https://ultravideo.fi/dataset.html, right-click its 1080p 8-bit RAW link, copy the address and download it manually) else (
  if defined SEVENZ (!SEVENZ! x -y Bosphorus.7z >nul && echo extracted Bosphorus) else (tar.exe -xf Bosphorus.7z && echo extracted Bosphorus)
)

echo === HoneyBee
set OK=0
for %%U in ("https://ultravideo.fi/video/HoneyBee_1920x1080_120fps_420_8bit_YUV_RAW.7z" "https://ultravideo.fi/video/HoneyBee_1920x1080_120fps_420_8bit_YUV.7z") do (
  if !OK!==0 (
    curl.exe -f -L -C - --retry 5 -o HoneyBee.7z %%U && set OK=1
  )
)
if !OK!==0 (echo [failed] HoneyBee: open https://ultravideo.fi/dataset.html, right-click its 1080p 8-bit RAW link, copy the address and download it manually) else (
  if defined SEVENZ (!SEVENZ! x -y HoneyBee.7z >nul && echo extracted HoneyBee) else (tar.exe -xf HoneyBee.7z && echo extracted HoneyBee)
)

echo === Jockey
set OK=0
for %%U in ("https://ultravideo.fi/video/Jockey_1920x1080_120fps_420_8bit_YUV_RAW.7z" "https://ultravideo.fi/video/Jockey_1920x1080_120fps_420_8bit_YUV.7z") do (
  if !OK!==0 (
    curl.exe -f -L -C - --retry 5 -o Jockey.7z %%U && set OK=1
  )
)
if !OK!==0 (echo [failed] Jockey: open https://ultravideo.fi/dataset.html, right-click its 1080p 8-bit RAW link, copy the address and download it manually) else (
  if defined SEVENZ (!SEVENZ! x -y Jockey.7z >nul && echo extracted Jockey) else (tar.exe -xf Jockey.7z && echo extracted Jockey)
)

echo === ReadySetGo
set OK=0
for %%U in ("https://ultravideo.fi/video/ReadySetGo_1920x1080_120fps_420_8bit_YUV_RAW.7z" "https://ultravideo.fi/video/ReadySetGo_1920x1080_120fps_420_8bit_YUV.7z") do (
  if !OK!==0 (
    curl.exe -f -L -C - --retry 5 -o ReadySetGo.7z %%U && set OK=1
  )
)
if !OK!==0 (echo [failed] ReadySetGo: open https://ultravideo.fi/dataset.html, right-click its 1080p 8-bit RAW link, copy the address and download it manually) else (
  if defined SEVENZ (!SEVENZ! x -y ReadySetGo.7z >nul && echo extracted ReadySetGo) else (tar.exe -xf ReadySetGo.7z && echo extracted ReadySetGo)
)

echo === ShakeNDry
set OK=0
for %%U in ("https://ultravideo.fi/video/ShakeNDry_1920x1080_120fps_420_8bit_YUV_RAW.7z" "https://ultravideo.fi/video/ShakeNDry_1920x1080_120fps_420_8bit_YUV.7z") do (
  if !OK!==0 (
    curl.exe -f -L -C - --retry 5 -o ShakeNDry.7z %%U && set OK=1
  )
)
if !OK!==0 (echo [failed] ShakeNDry: open https://ultravideo.fi/dataset.html, right-click its 1080p 8-bit RAW link, copy the address and download it manually) else (
  if defined SEVENZ (!SEVENZ! x -y ShakeNDry.7z >nul && echo extracted ShakeNDry) else (tar.exe -xf ShakeNDry.7z && echo extracted ShakeNDry)
)

echo === YachtRide
set OK=0
for %%U in ("https://ultravideo.fi/video/YachtRide_1920x1080_120fps_420_8bit_YUV_RAW.7z" "https://ultravideo.fi/video/YachtRide_1920x1080_120fps_420_8bit_YUV.7z") do (
  if !OK!==0 (
    curl.exe -f -L -C - --retry 5 -o YachtRide.7z %%U && set OK=1
  )
)
if !OK!==0 (echo [failed] YachtRide: open https://ultravideo.fi/dataset.html, right-click its 1080p 8-bit RAW link, copy the address and download it manually) else (
  if defined SEVENZ (!SEVENZ! x -y YachtRide.7z >nul && echo extracted YachtRide) else (tar.exe -xf YachtRide.7z && echo extracted YachtRide)
)

echo Done. Raw .yuv files are in data\uvg_raw (convert them with scripts/prepare_sources.py).
pause
