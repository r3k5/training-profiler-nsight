@echo off
setlocal enabledelayedexpansion

REM One-click runner for Windows
REM - Cleans results/
REM - Activates venv (if present)
REM - Runs Nsight profile + CSV/plot export
REM Usage: scripts\run_all.bat [noclean]

REM optional arg to skip cleaning
set "SKIP_CLEAN="
if /I "%~1"=="noclean" set "SKIP_CLEAN=1"

REM activate local venv if it exists (CMD-style activator)
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

REM clean old results unless user asked not to
if not defined SKIP_CLEAN (
  echo [1/3] Cleaning .\results ...
  if exist results rmdir /S /Q results
)

REM ensure results directory exists
if not exist results mkdir results

echo [2/3] Profiling with Nsight Systems ...
call "scripts\profile_nsys.bat"
if errorlevel 1 goto :err

echo [3/3] Exporting CSVs and plot ...
call "scripts\export_nsys_csv.bat"
if errorlevel 1 goto :err

echo.
echo Done.
echo   Trace:    results\nsys_report.nsys-rep  (open in Nsight Systems GUI)
echo   SQLite:   results\nsys_report.sqlite
echo   CSVs:     results\nsys_csv\gpukernsum.csv, cudaapisum.csv
echo   Plot:     results\nsys_csv\top10_kernels.png
echo.
goto :eof

:err
echo.
echo ERROR: See messages above.
exit /b 1
