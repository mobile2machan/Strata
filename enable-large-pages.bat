@echo off
title Enable Large Pages (Lock Pages in Memory)
cd /d "%~dp0"

echo ===================================================
echo  Enabling Lock Pages in Memory (SeLockMemoryPrivilege)
echo  Target user: %USERNAME%
echo ===================================================
echo.

net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] This script must be run as Administrator!
    echo Please right-click this file and choose "Run as administrator".
    echo.
    pause
    exit /b 1
)

set "TEMP_CFG=%TEMP%\secpol_%RANDOM%.cfg"
set "TEMP_DB=%TEMP%\secpol_%RANDOM%.sdb"

echo [1/3] Exporting current security policy ...
secedit /export /cfg "%TEMP_CFG%" >nul

echo [2/3] Adding SeLockMemoryPrivilege for %USERNAME% ...
powershell -NoProfile -Command ^
    "$u = $env:USERNAME; " ^
    "$content = [System.IO.File]::ReadAllText('%TEMP_CFG%'); " ^
    "if ($content -match 'SeLockMemoryPrivilege\s*=') { " ^
    "    $content = $content -replace '(SeLockMemoryPrivilege\s*=\s*)(.*)', ('$1$2,*' + $u); " ^
    "} else { " ^
    "    $content = $content -replace '(\[Privilege Rights\])', ('$1`r`nSeLockMemoryPrivilege = *' + $u); " ^
    "} " ^
    "[System.IO.File]::WriteAllText('%TEMP_CFG%', $content)"

echo [3/3] Applying security policy ...
secedit /configure /db "%TEMP_DB%" /cfg "%TEMP_CFG%" /areas USER_RIGHTS >nul

del /f /q "%TEMP_CFG%" "%TEMP_DB%" 2>nul

echo.
echo [SUCCESS] "Lock pages in memory" privilege has been assigned to %USERNAME%!
echo.
echo IMPORTANT: Windows requires a SIGN OUT (log off) or RESTART
echo for the privilege change to take effect for your user account.
echo.
pause
