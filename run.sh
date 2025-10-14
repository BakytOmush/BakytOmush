#!/usr/bin/env bash
set -e
python code/gru_lstm_h1_eurusd_cpu.py \  --data data/EURUSD60_sample.csv \  --start-date 2024-08-01 \  --time-step 240 \  --split 0.9 \  --epochs 10 \  --batch-size 64 \  --no-tune
