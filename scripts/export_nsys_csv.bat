:: SPDX-License-Identifier: MIT
:: Copyright (c) 2025 Mike Davis

@echo off
setlocal
set "NSYS=C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.5.1\target-windows-x64\nsys.exe"
set "REP=results\nsys_report.nsys-rep"
set "SQLITE=results\nsys_report.sqlite"
set "OUT=results\nsys_csv"

if not exist "%REP%" (
  echo ERROR: "%REP%" not found. Run scripts\profile_nsys.bat first.
  exit /b 1
)

if not exist "%OUT%" mkdir "%OUT%"

echo Creating/refreshing SQLite DB...
"%NSYS%" stats "%REP%" 1> "%OUT%\_stats_stdout.txt" 2> "%OUT%\_stats_stderr.txt"

if not exist "%SQLITE%" (
  echo ERROR: "%SQLITE%" not found after stats. See "%OUT%\_stats_stderr.txt".
  exit /b 1
)

echo Extracting CSVs from SQLite...
python scripts\extract_from_sqlite.py "%SQLITE%" "%OUT%"
if errorlevel 1 exit /b 1

echo Plotting top-10 kernels...
python scripts\plot_top10.py
if errorlevel 1 exit /b 1

echo Done. Open "%REP%" in Nsight Systems GUI. CSVs/plot are in "%OUT%".

echo Generating README snippet...
python scripts\gen_readme_snippet.py
endlocal
