#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EURUSD H1 forecasting (Close level) — CPU-friendly GRU+LSTM with optional HPO.
- Safe scaling (fit on train only)
- Resample to 1H with limited ffill to avoid weekend smearing
- Clean logging (console + file)
- Baseline in pips (last-close persistence)
- One-step ahead forecast
- Optional Hyperband via keras-tuner (moderate search space)
Tested with TensorFlow 2.13+ and scikit-learn 1.3+.
"""

# -----------------------------------------------------------------------------
# 0) Environment — set BEFORE importing TensorFlow
# -----------------------------------------------------------------------------
import os as _os
# CPU-only (machine doesn't support GPU)
_os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Enable oneDNN CPU acceleration
_os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
# Optional: control TF threading (tune for your CPU)
# _os.environ['TF_NUM_INTEROP_THREADS'] = '1'
# _os.environ['TF_NUM_INTRAOP_THREADS'] = '4'

# -----------------------------------------------------------------------------
# 1) Imports
# -----------------------------------------------------------------------------
import sys
import math
import json
import argparse
import logging
from logging import handlers
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GRU, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback, ModelCheckpoint

# seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------------------------------------------------------
# 2) Args & Paths
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="EURUSD H1 GRU+LSTM (Close-level) — CPU optimized")
    p.add_argument('--data', default='EURUSD60.csv', help='CSV file with columns: Date, Time, Open, High, Low, Close, Volume')
    p.add_argument('--start-date', default='2024-06-01', help='Use data from this date (inclusive, YYYY-MM-DD)')
    p.add_argument('--time-step', type=int, default=240, help='Window length')
    p.add_argument('--split', type=float, default=0.9, help='Train split fraction (0..1)')
    p.add_argument('--epochs', type=int, default=60, help='Max epochs for HPO / training')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--no-tune', action='store_true', help='Disable Hyperband tuning (use a default architecture)')
    p.add_argument('--logdir', default='outputs/logs', help='Directory for log files')
    p.add_argument('--outdir', default='outputs', help='Directory for models/metrics')
    return p.parse_args()

# -----------------------------------------------------------------------------
# 3) Logging
# -----------------------------------------------------------------------------
def setup_logging(log_dir: str):
    _os.makedirs(log_dir, exist_ok=True)
    log_file = _os.path.join(log_dir, f'train_20250813_110520.log')
    logger = logging.getLogger('eurusd_h1')
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    ch.setFormatter(ch_fmt)

    # File handler (rotating)
    fh = handlers.RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(fh_fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger, log_file

# -----------------------------------------------------------------------------
# 4) Data utilities
# -----------------------------------------------------------------------------
def load_data(csv_path: str, start_date: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, header=None, names=['Date','Time','Open','High','Low','Close','Volume'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = (df.set_index('Datetime')
            .drop(columns=['Date','Time'])
            .sort_index())
    # remove duplicated timestamps
    df = df[~df.index.duplicated(keep='first')]
    # filter by date
    df = df.loc[df.index >= pd.to_datetime(start_date)]
    if df.empty:
        raise ValueError(f"No data after {start_date}.")
    # resample hourly with limited ffill to avoid weekend smearing
    df = df.resample('H').ffill(limit=3)
    logger.info(f"Records after filtering/resample: {len(df)} | from {df.index.min()} to {df.index.max()}")
    return df

def add_indicators(df: pd.DataFrame, logger: logging.Logger):
    use_ta = False
    try:
        import pandas_ta as ta
        logger.info("pandas_ta found — computing indicators.")
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        bb = ta.bbands(df['Close'], length=20, std=2)
        df['BB_Upper'] = bb['BBU_20_2.0']
        df['BB_Lower'] = bb['BBL_20_2.0']
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ADX_14'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
        st = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
        df['Stoch_k'] = st['STOCHk_14_3_3']
        df['Stoch_d'] = st['STOCHd_14_3_3']
        use_ta = True
    except ImportError:
        logger.warning("pandas_ta not installed — proceeding without indicators.")
    df.dropna(inplace=True)
    base_feats = ['Open','High','Low','Close']
    ta_feats = ['SMA_20','EMA_20','RSI_14','MACD','MACD_Signal','BB_Upper','BB_Lower','ATR_14','ADX_14','Stoch_k','Stoch_d']
    features = base_feats + (ta_feats if use_ta else [])
    logger.info(f"Using features: {features} (total: {len(features)})")
    return df, features

def create_dataset(data: np.ndarray, time_step: int):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step, 3])  # target = scaled Close
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    return X, y

def inverse_close_only(scaler: MinMaxScaler, preds: np.ndarray, n_features: int) -> np.ndarray:
    """Inverse transform for a vector of Close predictions (shape: [N, 1])."""
    arr = np.zeros((len(preds), n_features), dtype=np.float32)
    arr[:, 3] = preds.reshape(-1)
    inv = scaler.inverse_transform(arr)[:, 3]
    return inv

def to_pips(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred) / 0.0001))

# -----------------------------------------------------------------------------
# 5) Model / Tuning
# -----------------------------------------------------------------------------
def build_model_fn(time_step: int, n_features: int):
    def build(hp):
        gru1 = hp.Int('gru1', 64, 256, step=64)
        gru2 = hp.Int('gru2', 64, 256, step=64)
        lstm_u = hp.Int('lstm', 64, 256, step=64)
        drp = hp.Float('drop', 0.0, 0.3, step=0.05)
        rdp = hp.Float('rec_drop', 0.0, 0.2, step=0.05)
        l2w = hp.Choice('l2', [1e-6, 1e-5, 1e-4])
        lr  = hp.Choice('lr', [1e-3, 5e-4, 1e-4])

        model = Sequential([
            GRU(gru1, return_sequences=True, input_shape=(time_step, n_features),
                kernel_regularizer=l2(l2w), dropout=drp, recurrent_dropout=rdp),
            GRU(gru2, return_sequences=True,
                kernel_regularizer=l2(l2w), dropout=drp, recurrent_dropout=rdp),
            LSTM(lstm_u, return_sequences=False,
                kernel_regularizer=l2(l2w), dropout=drp, recurrent_dropout=rdp),
            Dropout(hp.Float('drop_last', 0.0, 0.3, step=0.05)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model
    return build

def default_model(time_step: int, n_features: int) -> tf.keras.Model:
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(time_step, n_features), dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=l2(1e-5)),
        GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=l2(1e-5)),
        LSTM(128, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=l2(1e-5)),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=5e-4), loss='mse')
    return model

# -----------------------------------------------------------------------------
# 6) Main pipeline
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    logger, log_file = setup_logging(args.logdir)
    logger.info("Starting EURUSD H1 training (Close-level target) — CPU mode")
    logger.info(f"Args: {vars(args)}")
    _os.makedirs(args.outdir, exist_ok=True)
    out_models = _os.path.join(args.outdir, 'models')
    out_metrics = _os.path.join(args.outdir, 'metrics')
    out_hpo = _os.path.join(args.outdir, 'hpo')
    for d in [out_models, out_metrics, out_hpo]:
        _os.makedirs(d, exist_ok=True)

    # Load & features
    df = load_data(args.data, args.start_date, logger)
    df, features = add_indicators(df, logger)
    n_features = len(features)

    # Prepare arrays
    data_raw = df[features].astype('float32').values
    split_raw = int(len(data_raw) * args.split)
    if split_raw <= args.time_step + 1:
        raise ValueError("Not enough data for the given time_step and split.")

    scaler = MinMaxScaler((0,1))
    scaler.fit(data_raw[:split_raw])  # fit ONLY on train
    data = scaler.transform(data_raw)

    # Build datasets
    X, y = create_dataset(data, args.time_step)
    split_idx = split_raw - args.time_step - 1
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    logger.info(f"Shapes — X_train: {X_train.shape}, X_test: {X_test.shape}; y_train: {y_train.shape}, y_test: {y_test.shape}")


    # Callbacks
    early = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, min_delta=1e-6)
    rlr   = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6)
    ckpt_path = _os.path.join(out_models, f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
    ckpt = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=False)
    log_cb = LambdaCallback(on_epoch_end=lambda e, logs: logger.info(f"Epoch {e+1}: loss={logs['loss']:.6f}, val_loss={logs['val_loss']:.6f}"))

    # Build / Tune
    model = None
    if not args.no_tune:
        try:
            import keras_tuner as kt
            logger.info("keras_tuner found — running Hyperband (moderate)")
            tuner = kt.Hyperband(
                build_model_fn(args.time_step, n_features),
                objective='val_loss',
                max_epochs=args.epochs,
                factor=3,
                hyperband_iterations=1,
                directory=out_hpo,
                project_name=f'gru_lstm_20250813_110520',
                overwrite=False
            )
            tuner.search(X_train, y_train,
                         validation_data=(X_test, y_test),
                         epochs=args.epochs,
                         batch_size=args.batch_size,
                         callbacks=[early, rlr, log_cb])
            best_hp = tuner.get_best_hyperparameters(1)[0]
            logger.info(f"Best HPs: {best_hp.values}")
            model = tuner.get_best_models(1)[0]
        except Exception as e:
            logger.warning(f"Tuning failed or keras_tuner not available: {e} — falling back to default model.")
            model = default_model(args.time_step, n_features)
    else:
        logger.info("Tuning disabled — using default model.")
        model = default_model(args.time_step, n_features)

    # Fine-tuning (short)
    logger.info("Starting training / fine-tuning...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=max(30, args.epochs // 2),
        batch_size=args.batch_size,
        callbacks=[early, rlr, log_cb, ckpt],
        verbose=0
    )
    logger.info(f"Training finished. Best model saved to: {ckpt_path}")


    # Save final artifacts
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_model_path = _os.path.join(out_models, f'final_model_{stamp}.keras')
    weights_path = _os.path.join(out_models, f'final_weights_{stamp}.weights.h5')
    scaler_path = _os.path.join(out_models, f'scaler_{stamp}.npy')
    conf_path = _os.path.join(out_models, f'run_config_{stamp}.json')

    model.save(final_model_path)
    model.save_weights(weights_path)
    # Save scaler parameters
    np.save(scaler_path, {
        'scale_': getattr(scaler, 'scale_', None),
        'min_': getattr(scaler, 'min_', None),
        'data_min_': getattr(scaler, 'data_min_', None),
        'data_max_': getattr(scaler, 'data_max_', None),
        'data_range_': getattr(scaler, 'data_range_', None),
        'feature_range': scaler.feature_range
    }, allow_pickle=True)
    # Save config
    with open(conf_path, 'w', encoding='utf-8') as f:
        json.dump({
            'features': features,
            'time_step': args.time_step,
            'split_idx': int(split_idx),
            'close_col_index': 3,
            'start_date': args.start_date,
            'data_file': args.data,
            'log_file': log_file
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Artifacts saved: \n  model={final_model_path}\n  weights={weights_path}\n  scaler={scaler_path}\n  config={conf_path}")


    # Evaluation — inverse to real prices
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    y_tr_true = inverse_close_only(scaler, y_train.reshape(-1,1), n_features)
    y_te_true = inverse_close_only(scaler, y_test.reshape(-1,1), n_features)
    y_tr_pred = inverse_close_only(scaler, train_pred, n_features)
    y_te_pred = inverse_close_only(scaler, test_pred, n_features)

    # Naive baseline (last Close of each window)
    naive_scaled_test = X_test[:, -1, 3].reshape(-1,1)
    y_te_naive = inverse_close_only(scaler, naive_scaled_test, n_features)

    # Metrics
    def rmse(a,b): return float(math.sqrt(mean_squared_error(a,b)))
    tr_mae = float(mean_absolute_error(y_tr_true, y_tr_pred))
    te_mae = float(mean_absolute_error(y_te_true, y_te_pred))
    tr_rmse = rmse(y_tr_true, y_tr_pred)
    te_rmse = rmse(y_te_true, y_te_pred)

    te_mae_pips_model = to_pips(y_te_true, y_te_pred)
    te_mae_pips_naive = to_pips(y_te_true, y_te_naive)

    logger.info(f"Train MAE: {tr_mae:.6f} | RMSE: {tr_rmse:.6f} (abs)")
    logger.info(f" Test MAE: {te_mae:.6f} | RMSE: {te_rmse:.6f} (abs)")
    logger.info(f" Test MAE (pips) — Model: {te_mae_pips_model:.2f} | Naive: {te_mae_pips_naive:.2f}")

    # One-step ahead forecast
    last_window = data[-args.time_step:]
    next_scaled = model.predict(last_window.reshape(1, args.time_step, n_features), verbose=0)
    next_close = inverse_close_only(scaler, next_scaled.reshape(-1,1), n_features)[0]
    next_t = df.index[-1] + pd.Timedelta(hours=1)
    logger.info(f"One-step forecast for {next_t}: {next_close:.6f}" )

    # Save metrics json
    metrics_path = _os.path.join(out_metrics, f'metrics_{stamp}.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_mae': tr_mae,
            'test_mae': te_mae,
            'train_rmse': tr_rmse,
            'test_rmse': te_rmse,
            'test_mae_pips_model': te_mae_pips_model,
            'test_mae_pips_naive': te_mae_pips_naive,
            'one_step_forecast_time': str(next_t),
            'one_step_forecast_close': float(next_close),
            'n_train': int(len(y_tr_true)),
            'n_test': int(len(y_te_true))
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Metrics saved: {metrics_path}" )
    logger.info("Done.")

if __name__ == '__main__':
    main()
