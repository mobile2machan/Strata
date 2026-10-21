@echo off
setlocal
cd /d "%~dp0apps\host"

if "%STRATA_PORT%"=="" set STRATA_PORT=8791
if "%STRATA_REAL_ENGINE%"=="" set STRATA_REAL_ENGINE=1
if "%STRATA_SERVER_URL%"=="" set STRATA_SERVER_URL=http://192.168.1.252:8080/v1
if "%STRATA_SERVER_API_KEY%"=="" set STRATA_SERVER_API_KEY=strata
if exist "%~dp0apps\harness\BUNDLE.json" set "STRATA_HARNESS_DIR=%~dp0apps\harness"

echo Starting Strata GUI on http://127.0.0.1:%STRATA_PORT% (Remote Engine: %STRATA_SERVER_URL%) ...
start /b powershell -NoProfile -WindowStyle Hidden -Command "for ($i=0; $i -lt 30; $i++) { Start-Sleep -Milliseconds 500; try { $null = Invoke-WebRequest -Uri 'http://127.0.0.1:%STRATA_PORT%/api/health' -UseBasicParsing -TimeoutSec 1; Start-Process 'http://127.0.0.1:%STRATA_PORT%'; break } catch {} }"
node server\index.mjs --dist dist
