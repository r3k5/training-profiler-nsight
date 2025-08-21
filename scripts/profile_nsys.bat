:: SPDX-License-Identifier: MIT
:: Copyright (c) 2025 Mike Davis

@echo off
if not exist results mkdir results
set "NSYS=C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.5.1\target-windows-x64\nsys.exe"

"%NSYS%" profile ^
  --force-overwrite=true ^
  --trace=cuda,nvtx,cublas,cuDNN,wddm ^
  -o results\nsys_report ^
  python -u src\train.py --epochs 1 --batch-size 64 --seq-len 128 --d-model 256 --layers 4 --amp %*
