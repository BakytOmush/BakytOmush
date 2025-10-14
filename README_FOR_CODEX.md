# FX H1 — Minimal Repro for CODEX

**Goal:** improve one-step EURUSD H1 forecasting so that the model beats naive baselines in **MAE (pips)** on a CPU-only box.

## Contents
- `code/gru_lstm_h1_eurusd_cpu.py` — original training script.
- `data/EURUSD60_sample.csv` — small H1 sample (Date, Time, Open, High, Low, Close, Volume).
- `tasks/codex_prompt.txt` — precise tasks & success criteria.
- `run.sh` — one-line repro.

## Quick start
```bash
python code/gru_lstm_h1_eurusd_cpu.py --data data/EURUSD60_sample.csv --start-date 2024-08-01 --time-step 240 --split 0.9 --epochs 10 --batch-size 64 --no-tune
```

## Notes
- TensorFlow 2.13+, scikit-learn 1.3+.
- Script is CPU-oriented and resamples to H1 with limited forward-fill (weekend-safe).
- Baseline in pips is included for honest comparison.
