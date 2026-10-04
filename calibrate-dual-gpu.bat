@echo off
title Strata - Auto Calibration (Dual GPU)
cd /d "%~dp0"

echo ===================================================================
echo  Starting Strata Auto-Calibration (Dual GPU Benchmark)
echo  This measures and tunes:
echo    1. PCIe Transfer Fraction (--pcie-frac)
echo    2. Speculative Draft Floor (--spec-min-p)
echo    3. CPU Worker Threads      (--pool-workers)
echo ===================================================================
echo.
echo NOTE: Please ensure no other Strata server is running!
echo.

"D:\GenerativeAI\Strata\.venv\Scripts\python.exe" "D:\GenerativeAI\Strata\tools\run_calibration.py" "D:\GenerativeAI\Strata\strata-iq3_s-dual-gpu.json"
pause
