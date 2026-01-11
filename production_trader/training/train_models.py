"""
Model Training Module
=====================
Trains XGBoost models for breakout strategy on historical data.

Key feature: Parameterized training cutoff date to prevent lookahead bias.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features (matches strategy_15m.py)"""
    df = df.copy()

    # Hour and minute features
    df['hour'] = df.index.hour
    df['minute_slot'] = df.index.minute // 15

    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df.index.dayofweek

    # Trading sessions (must match training script)
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['session_overlap'] = (df['european_session'] + df['us_session']) > 1

    # Weekend proximity
    df['friday_close'] = ((df['day_of_week'] == 4) & (df['hour'] >= 20)).astype(int)
    df['sunday_open'] = ((df['day_of_week'] == 6) & (df['hour'] <= 3)).astype(int)

    return df


def calculate_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Calculate technical features (matches strategy_15m.py)"""
    df = df.copy()

    # Add time features first
    df = add_time_features(df)

    # Breakout levels (using lookback period)
    df[f'high_{lookback}p'] = df['high'].rolling(lookback).max()
    df[f'low_{lookback}p'] = df['low'].rolling(lookback).min()
    df[f'range_{lookback}p'] = df[f'high_{lookback}p'] - df[f'low_{lookback}p']

    # Distance to breakout levels
    df['dist_to_high'] = (df[f'high_{lookback}p'] - df['close']) / df['close']
    df['dist_to_low'] = (df['close'] - df[f'low_{lookback}p']) / df['close']

    # EMAs (adjusted for 15m data)
    df['ema_12'] = df['close'].ewm(span=12).mean()  # 3 hours
    df['ema_26'] = df['close'].ewm(span=26).mean()  # 6.5 hours
    df['ema_50'] = df['close'].ewm(span=50).mean()  # 12.5 hours
    df['ema_100'] = df['close'].ewm(span=100).mean()  # 25 hours

    # Price relative to EMAs
    df['price_vs_ema12'] = (df['close'] - df['ema_12']) / df['close']
    df['price_vs_ema26'] = (df['close'] - df['ema_26']) / df['close']
    df['price_vs_ema50'] = (df['close'] - df['ema_50']) / df['close']
    df['price_vs_ema100'] = (df['close'] - df['ema_100']) / df['close']

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI (14 periods = 3.5 hours)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Volatility (ATR for 15m data)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']

    # Volume features
    df['volume_ma'] = df['volume'].rolling(96).mean()  # 24h average
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Recent momentum (shorter periods for 15m)
    df['return_1p'] = df['close'].pct_change(1)   # Last 15m
    df['return_4p'] = df['close'].pct_change(4)   # Last hour
    df['return_16p'] = df['close'].pct_change(16) # Last 4 hours
    df['return_96p'] = df['close'].pct_change(96) # Last 24 hours

    # Spread analysis
    df['spread_ma'] = df['spread_pct'].rolling(96).mean()  # 24h avg
    df['spread_ratio'] = df['spread_pct'] / df['spread_ma']

    return df


def calculate_targets(df: pd.DataFrame, lookback: int, forward_periods: int) -> pd.DataFrame:
    """
    Calculate forward-looking targets for training.

    CRITICAL: Target at bar i uses data from i+1 to i+forward_periods.
    This means last training bar's target extends forward_periods into the future.
    """
    df = df.copy()

    # Future high/low over next forward_periods bars
    df['future_high'] = df['high'].shift(-1).rolling(forward_periods).max()
    df['future_low'] = df['low'].shift(-1).rolling(forward_periods).min()

    # Current breakout levels
    high_level = df[f'high_{lookback}p']
    low_level = df[f'low_{lookback}p']

    # Did price break out in the next forward_periods bars?
    df['breakout_high'] = (df['future_high'] > high_level).astype(int)
    df['breakout_low'] = (df['future_low'] < low_level).astype(int)

    return df


def train_models(
    pairs: List[str],
    data_dir: str,
    end_date: str,
    training_months: int = 10,
    lookback_periods: int = 80,
    forward_periods: int = 24,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train XGBoost models for breakout strategy.

    CRITICAL: Trains on data ending at end_date (exclusive).
    The last training bar will have targets extending forward_periods into the future,
    so end_date should be set to (simulation_start - forward_periods * 15min) to avoid lookahead.

    Args:
        pairs: List of currency pairs (e.g., ['EURUSD', 'USDJPY'])
        data_dir: Directory containing {pair}_15m.csv files
        end_date: Training cutoff date - data before this date (YYYY-MM-DD HH:MM)
        training_months: How many months to train on (default: 10)
        lookback_periods: Feature lookback window in bars (default: 80 = 20 hours)
        forward_periods: Target forward window in bars (default: 24 = 6 hours)
        output_dir: Where to save models (None = don't save, keep in memory only)

    Returns:
        Dictionary with trained models and statistics:
        {
            'EURUSD': {
                'model_high': <XGBoost model>,
                'model_low': <XGBoost model>,
                'train_samples': 28776,
                'train_start': '2023-03-01 00:00',
                'train_end': '2023-12-31 18:00'
            },
            ...
        }
    """
    logger.info("="*80)
    logger.info("TRAINING MODELS FOR SIMULATION")
    logger.info("="*80)
    logger.info(f"Training cutoff: {end_date}")
    logger.info(f"Training window: {training_months} months")
    logger.info(f"Lookback periods: {lookback_periods} bars ({lookback_periods * 15 / 60:.1f} hours)")
    logger.info(f"Forward periods: {forward_periods} bars ({forward_periods * 15 / 60:.1f} hours)")
    logger.info("="*80)

    # Feature columns (MUST match strategy_15m.py exactly)
    feature_cols = [
        # Breakout features
        'dist_to_high', 'dist_to_low', f'range_{lookback_periods}p',
        # EMAs
        'price_vs_ema12', 'price_vs_ema26', 'price_vs_ema50', 'price_vs_ema100',
        # MACD
        'macd', 'macd_signal', 'macd_hist',
        # RSI
        'rsi_14',
        # Volatility
        'atr_pct',
        # Volume
        'volume_ratio',
        # Momentum
        'return_1p', 'return_4p', 'return_16p', 'return_96p',
        # Spread
        'spread_pct', 'spread_ratio',
        # Time features
        'hour', 'minute_slot', 'day_of_week',
        'asian_session', 'european_session', 'us_session', 'session_overlap',
        'friday_close', 'sunday_open'
    ]

    # Parse end date and make timezone-aware (UTC)
    end_timestamp = pd.Timestamp(end_date)
    if end_timestamp.tzinfo is None:
        end_timestamp = end_timestamp.tz_localize('UTC')

    # Calculate training start (training_months before end)
    train_start = end_timestamp - relativedelta(months=training_months)

    logger.info(f"Training period: {train_start} to {end_timestamp}")
    logger.info(f"Loading data from {data_dir}...")

    # Load and prepare data for all pairs
    all_data = {}
    for pair in pairs:
        file_path = Path(data_dir) / f'{pair}_15m.csv'

        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Filter to training window
        df = df[(df.index >= train_start) & (df.index < end_timestamp)]

        if len(df) < 1000:
            logger.warning(f"{pair}: Insufficient data ({len(df)} bars) - skipping")
            continue

        # Calculate features and targets
        df = calculate_features(df, lookback_periods)
        df = calculate_targets(df, lookback_periods, forward_periods)

        # Drop rows with NaN (from rolling windows)
        df_clean = df.dropna(subset=feature_cols + ['breakout_high', 'breakout_low'])

        all_data[pair] = df_clean

        logger.info(f"  {pair}: {len(df_clean):,} samples ({df_clean.index.min()} to {df_clean.index.max()})")

    if not all_data:
        raise ValueError("No data loaded for any pairs!")

    # Train models
    logger.info("")
    logger.info("Training XGBoost models...")
    trained_models = {}

    for pair, df in all_data.items():
        logger.info(f"\n{pair}:")
        logger.info(f"  Training samples: {len(df):,}")

        # Prepare training data
        X_train = df[feature_cols]
        y_train_high = df['breakout_high']
        y_train_low = df['breakout_low']

        # Train breakout high model
        model_high = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        model_high.fit(X_train, y_train_high, verbose=False)

        # Train breakout low model
        model_low = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        model_low.fit(X_train, y_train_low, verbose=False)

        # Store models
        trained_models[pair] = {
            'model_high': model_high,
            'model_low': model_low,
            'train_samples': len(df),
            'train_start': df.index.min().strftime('%Y-%m-%d %H:%M'),
            'train_end': df.index.max().strftime('%Y-%m-%d %H:%M'),
            'feature_cols': feature_cols,
            'lookback_periods': lookback_periods,
            'forward_periods': forward_periods
        }

        logger.info(f"  Models trained successfully")

    # Save models if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nSaving models to {output_path}...")

        for pair, model_dict in trained_models.items():
            # Save high breakout model
            high_path = output_path / f'xgboost_15m_{pair}_high.pkl'
            with open(high_path, 'wb') as f:
                pickle.dump(model_dict['model_high'], f)

            # Save low breakout model
            low_path = output_path / f'xgboost_15m_{pair}_low.pkl'
            with open(low_path, 'wb') as f:
                pickle.dump(model_dict['model_low'], f)

            logger.info(f"  Saved: {pair}")

        logger.info("Models saved successfully")
    else:
        logger.info("\nModels kept in memory only (not saved to disk)")

    logger.info("")
    logger.info("="*80)
    logger.info(f"TRAINING COMPLETE: {len(trained_models)} pairs")
    logger.info("="*80)

    return trained_models
