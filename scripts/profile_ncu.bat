:: SPDX-License-Identifier: MIT
:: Copyright (c) 2025 Mike Davis

@echo off
if not exist results mkdir results

ncu --force-overwrite --set full --target-processes all -o results\ncu_report ^
  python -u src\train.py --epochs 1 --batch-size 64 --seq-len 128 --d-model 256 --layers 4 --amp