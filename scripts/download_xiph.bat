@echo off
chcp 65001 >nul
REM Download Xiph derf 1080p sequences into ..\data\sources (resumable: just run again if interrupted)
cd /d "%~dp0.."
if not exist data\sources mkdir data\sources
cd data\sources

echo === crowd_run_1080p50.y4m
curl.exe -L -C - --retry 5 -o crowd_run_1080p50.y4m https://media.xiph.org/video/derf/y4m/crowd_run_1080p50.y4m
if errorlevel 1 echo [failed] crowd_run_1080p50 - check the file name on the website
echo === ducks_take_off_1080p50.y4m
curl.exe -L -C - --retry 5 -o ducks_take_off_1080p50.y4m https://media.xiph.org/video/derf/y4m/ducks_take_off_1080p50.y4m
if errorlevel 1 echo [failed] ducks_take_off_1080p50 - check the file name on the website
echo === in_to_tree_1080p50.y4m
curl.exe -L -C - --retry 5 -o in_to_tree_1080p50.y4m https://media.xiph.org/video/derf/y4m/in_to_tree_1080p50.y4m
if errorlevel 1 echo [failed] in_to_tree_1080p50 - check the file name on the website
echo === old_town_cross_1080p50.y4m
curl.exe -L -C - --retry 5 -o old_town_cross_1080p50.y4m https://media.xiph.org/video/derf/y4m/old_town_cross_1080p50.y4m
if errorlevel 1 echo [failed] old_town_cross_1080p50 - check the file name on the website
echo === park_joy_1080p50.y4m
curl.exe -L -C - --retry 5 -o park_joy_1080p50.y4m https://media.xiph.org/video/derf/y4m/park_joy_1080p50.y4m
if errorlevel 1 echo [failed] park_joy_1080p50 - check the file name on the website
echo === pedestrian_area_1080p25.y4m
curl.exe -L -C - --retry 5 -o pedestrian_area_1080p25.y4m https://media.xiph.org/video/derf/y4m/pedestrian_area_1080p25.y4m
if errorlevel 1 echo [failed] pedestrian_area_1080p25 - check the file name on the website
echo === riverbed_1080p25.y4m
curl.exe -L -C - --retry 5 -o riverbed_1080p25.y4m https://media.xiph.org/video/derf/y4m/riverbed_1080p25.y4m
if errorlevel 1 echo [failed] riverbed_1080p25 - check the file name on the website
echo === rush_hour_1080p25.y4m
curl.exe -L -C - --retry 5 -o rush_hour_1080p25.y4m https://media.xiph.org/video/derf/y4m/rush_hour_1080p25.y4m
if errorlevel 1 echo [failed] rush_hour_1080p25 - check the file name on the website
echo === station2_1080p25.y4m
curl.exe -L -C - --retry 5 -o station2_1080p25.y4m https://media.xiph.org/video/derf/y4m/station2_1080p25.y4m
if errorlevel 1 echo [failed] station2_1080p25 - check the file name on the website
echo === sunflower_1080p25.y4m
curl.exe -L -C - --retry 5 -o sunflower_1080p25.y4m https://media.xiph.org/video/derf/y4m/sunflower_1080p25.y4m
if errorlevel 1 echo [failed] sunflower_1080p25 - check the file name on the website
echo === tractor_1080p25.y4m
curl.exe -L -C - --retry 5 -o tractor_1080p25.y4m https://media.xiph.org/video/derf/y4m/tractor_1080p25.y4m
if errorlevel 1 echo [failed] tractor_1080p25 - check the file name on the website
echo === blue_sky_1080p25.y4m
curl.exe -L -C - --retry 5 -o blue_sky_1080p25.y4m https://media.xiph.org/video/derf/y4m/blue_sky_1080p25.y4m
if errorlevel 1 echo [failed] blue_sky_1080p25 - check the file name on the website

echo Done.
pause
