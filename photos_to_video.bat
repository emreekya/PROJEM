@echo off
setlocal
cd /d "%~dp0"

echo =============================================
echo   MoneyPrinterTurbo - Fotoğraftan Video
echo =============================================
echo.

if not exist "photos" mkdir "photos"
if not exist "outputs" mkdir "outputs"

set "DURATION=3"
set /p DURATION=Her foto kac saniye gorunsun? Varsayilan 3: 
if "%DURATION%"=="" set "DURATION=3"

set "SIZE=1080x1920"
set /p SIZE=Video boyutu? Shorts icin 1080x1920, yatay icin 1920x1080. Varsayilan 1080x1920: 
if "%SIZE%"=="" set "SIZE=1080x1920"

set "FIT=cover"
set /p FIT=Fit modu? cover kirpar/doldurur, contain kirpmaz siyah dolgu. Varsayilan cover: 
if "%FIT%"=="" set "FIT=cover"

echo.
echo Fotograflarini photos klasorune koydugundan emin ol.
echo Cikis dosyasi: outputs\photo-video.mp4
echo.
pause

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" tools\photos_to_video.py --input photos --output outputs\photo-video.mp4 --duration %DURATION% --size %SIZE% --fit %FIT%
) else (
  python tools\photos_to_video.py --input photos --output outputs\photo-video.mp4 --duration %DURATION% --size %SIZE% --fit %FIT%
)

echo.
pause
