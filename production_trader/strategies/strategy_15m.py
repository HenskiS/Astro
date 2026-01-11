"""
15-Minute Breakout Strategy - Signal Generation
================================================
Generates trading signals using trained XGBoost models.

Process:
1. Fetch last 200 bars of 15m data from OANDA
2. Calculate technical features
3. Generate predictions using trained models
4. Filter by confidence (70%+) and spread conditions
5. Calculate position sizes
6. Return signals for position manager
"""
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

import pandas as pd
import numpy as np
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from production_trader.execution.oanda_broker import OandaBroker

logger = logging.getLogger(__name__)


class Strategy15m:
    """
    15-minute breakout strategy signal generator.

    Uses trained XGBoost models to predict breakout probabilities
    and generate trading signals.
    """

    def __init__(self, config, broker: OandaBroker, models: Dict = None, predictions: Dict = None):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration
            broker: OANDA broker instance
            models: Pre-trained models dict (optional, if None loads from disk)
                   Format: {'EURUSD': {'model_high': <model>, 'model_low': <model>}, ...}
            predictions: Pre-generated predictions dict (optional)
                        Format: {'EURUSD': DataFrame with 'breakout_high_prob', 'breakout_low_prob', 'high_80p', 'low_80p'}
        """
        self.config = config
        self.broker = broker
        self.models = {}
        self.predictions = predictions
        self.lookback_periods = config.lookback_periods

        # Use provided predictions, models, or load models from disk
        if predictions is not None:
            logger.info(f"Using pre-generated predictions for {len(predictions)} pairs")
        elif models is not None:
            self._use_provided_models(models)
        else:
            self._load_models()

        logger.info("Strategy15m initialized")

    def _use_provided_models(self, trained_models: Dict):
        """Use pre-trained models provided to the strategy"""
        for pair, model_dict in trained_models.items():
            self.models[f'{pair}_high'] = model_dict['model_high']
            self.models[f'{pair}_low'] = model_dict['model_low']
            logger.debug(f"Using provided models for {pair}")

        logger.info(f"Using {len(self.models)} pre-trained models")

    def _load_models(self):
        """Load trained XGBoost models for all pairs from disk"""
        models_dir = Path(__file__).parent.parent.parent / 'models'

        for pair in self.config.pairs:
            model_path = models_dir / f'xgboost_15m_{pair}_high.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[f'{pair}_high'] = pickle.load(f)
                logger.info(f"Loaded model: {pair}_high")
            else:
                logger.warning(f"Model not found: {model_path}")

            model_path = models_dir / f'xgboost_15m_{pair}_low.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[f'{pair}_low'] = pickle.load(f)
                logger.info(f"Loaded model: {pair}_low")
            else:
                logger.warning(f"Model not found: {model_path}")

        logger.info(f"Loaded {len(self.models)} models")

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()

        # Hour and minute features
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['minute_slot'] = (df['minute'] // 15).astype(int)

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

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for 15m data"""
        df = df.copy()

        # Add time features first
        df = self.add_time_features(df)

        # Breakout levels (using lookback period)
        lookback = self.lookback_periods
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

        # Spread analysis (spread_pct already provided by OANDA broker)
        df['spread_ma'] = df['spread_pct'].rolling(96).mean()  # 24h avg
        df['spread_ratio'] = df['spread_pct'] / df['spread_ma']

        return df

    def generate_signals(self, current_capital: float, existing_positions: Dict) -> List[Dict]:
        """
        Generate trading signals for all pairs.

        Args:
            current_capital: Current account capital
            existing_positions: Dictionary of existing positions by pair

        Returns:
            List of signal dictionaries
        """
        signals = []
        current_time = datetime.now()
        current_hour = current_time.hour

        # Skip if in avoid hours
        if current_hour in self.config.avoid_hours:
            logger.info(f"Skipping signal generation - hour {current_hour} in avoid list")
            return signals

        logger.info(f"Generating signals for {len(self.config.pairs)} pairs")

        for pair in self.config.pairs:
            try:
                # Fetch historical data to get current timestamp and prices
                df = self.broker.get_historical_candles(
                    pair=pair,
                    timeframe='M15',
                    count=5  # Just need latest bar for timestamp and prices
                )

                if df is None or len(df) < 1:
                    logger.warning(f"No data for {pair}")
                    continue

                # Get latest complete bar
                latest = df.iloc[-1]
                # Use broker's current time for timestamp (simulation compatibility)
                current_timestamp = getattr(self.broker, 'current_time', None)
                if current_timestamp is None:
                    current_timestamp = df.index[-1]  # Fallback to data timestamp

                # Get predictions - either from pre-loaded dict or generate with models
                if self.predictions is not None:
                    # Use pre-generated predictions
                    if pair not in self.predictions:
                        logger.warning(f"No predictions for {pair}")
                        continue

                    pair_preds = self.predictions[pair]
                    if current_timestamp not in pair_preds.index:
                        logger.warning(f"[TIMESTAMP MISMATCH] No prediction for {pair} at {current_timestamp}")
                        if len(pair_preds) > 0:
                            logger.warning(f"  First available: {pair_preds.index[0]}, Last: {pair_preds.index[-1]}")
                        continue

                    pred = pair_preds.loc[current_timestamp]
                    prob_high = pred['breakout_high_prob']
                    prob_low = pred['breakout_low_prob']
                    breakout_high_level = pred[f'high_{self.lookback_periods}p']
                    breakout_low_level = pred[f'low_{self.lookback_periods}p']

                    # Debug logging
                    if pair == 'EURUSD':
                        logger.info(f"  [DEBUG] {pair} @ {current_timestamp}: high={prob_high:.4f}, low={prob_low:.4f}")

                else:
                    # Generate predictions using models (original approach)
                    # Fetch more data for feature calculation
                    df = self.broker.get_historical_candles(
                        pair=pair,
                        timeframe='M15',
                        count=220  # Extra bars for rolling calculations
                    )

                    if df is None or len(df) < 200:
                        logger.warning(f"Insufficient data for {pair}: {len(df) if df is not None else 0} bars")
                        continue

                    # Calculate features
                    df = self.calculate_features(df)
                    latest = df.iloc[-1]

                    # Check if we have valid features
                    if pd.isna(latest[f'high_{self.lookback_periods}p']):
                        logger.warning(f"Invalid features for {pair} - skipping")
                        continue

                    # Get feature columns (MUST match training script exactly)
                    feature_cols = [
                        # Breakout features
                        'dist_to_high', 'dist_to_low', f'range_{self.lookback_periods}p',
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

                    # Prepare feature vector
                    X = latest[feature_cols].values.reshape(1, -1)

                    # Replace any NaN/inf with 0
                    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                    # Get predictions
                    model_high = self.models.get(f'{pair}_high')
                    model_low = self.models.get(f'{pair}_low')

                    if model_high is None or model_low is None:
                        logger.warning(f"Models not found for {pair}")
                        continue

                    # Predict probabilities
                    prob_high = model_high.predict_proba(X)[0, 1]
                    prob_low = model_low.predict_proba(X)[0, 1]
                    breakout_high_level = latest[f'high_{self.lookback_periods}p']
                    breakout_low_level = latest[f'low_{self.lookback_periods}p']

                # Determine best direction
                if prob_high > prob_low:
                    direction = 'long'
                    confidence = prob_high
                    breakout_level = breakout_high_level
                    target = breakout_level  # Use breakout level directly (matches backtest)
                else:
                    direction = 'short'
                    confidence = prob_low
                    breakout_level = breakout_low_level
                    target = breakout_level  # Use breakout level directly (matches backtest)

                # Log all predictions for monitoring
                logger.info(f"{pair} {direction.upper()}: confidence={confidence:.3f} (threshold={self.config.min_confidence:.2f})")

                # Filter by confidence
                if confidence < self.config.min_confidence:
                    logger.info(f"  → FILTERED: confidence {confidence:.3f} < {self.config.min_confidence:.2f}")
                    continue

                # Check position limits
                pair_positions = existing_positions.get(pair, [])
                if len(pair_positions) >= self.config.max_positions_per_pair:
                    logger.info(f"  → FILTERED: Max positions reached ({len(pair_positions)})")
                    continue

                # Check for competing positions (FIFO handling)
                if len(pair_positions) > 0:
                    existing_directions = set(p.direction for p in pair_positions)
                    if direction not in existing_directions:
                        logger.info(f"  → FILTERED: Competing position exists")
                        continue

                logger.info(f"  → SIGNAL CREATED ✓")

                # Calculate position size (40% of capital per trade)
                # IMPORTANT: OANDA units are ALWAYS in base currency, NOT USD!
                # For forex pairs, we need to convert USD capital to base currency units

                mid_price = latest['close']
                capital_for_trade = current_capital * self.config.position_size_pct

                # Determine pair type and calculate correct position size
                if pair.startswith('USD'):
                    # USD-base pairs (USDJPY, USDCAD, USDCHF)
                    # 1 unit = $1 USD, so units = desired dollar exposure
                    position_size = int(capital_for_trade)
                    logger.debug(f"{pair}: USD-base, size={position_size} units (${capital_for_trade:.2f})")

                elif pair.endswith('USD'):
                    # USD-quote pairs (EURUSD, GBPUSD, AUDUSD, NZDUSD)
                    # 1 unit = 1 base currency unit
                    # To risk $X, buy X/price units of base currency
                    position_size = int(capital_for_trade / mid_price)
                    logger.debug(f"{pair}: USD-quote, size={position_size} units (${capital_for_trade:.2f} / {mid_price:.5f})")

                else:
                    # Cross pairs (EURJPY, GBPJPY, etc.) - neither base nor quote is USD
                    # Need to convert USD → base currency using base/USD rate
                    base_currency = pair[:3]
                    base_usd_pair = f'{base_currency}USD'

                    logger.debug(f"{pair}: Cross pair, fetching {base_usd_pair} rate...")

                    # Fetch base/USD conversion rate
                    try:
                        base_usd_prices = self.broker.get_current_prices([base_usd_pair])

                        if base_usd_pair in base_usd_prices:
                            base_usd_rate = base_usd_prices[base_usd_pair].mid_close
                            # Convert USD capital to base currency units
                            # Example: $100 / (EURUSD 1.05) = 95.24 EUR units
                            position_size = int(capital_for_trade / base_usd_rate)
                            logger.info(f"{pair}: Cross pair conversion - ${capital_for_trade:.2f} / {base_usd_pair} {base_usd_rate:.5f} = {position_size} {base_currency} units")
                        else:
                            logger.error(f"Cannot find {base_usd_pair} rate for {pair} - skipping signal")
                            continue

                    except Exception as e:
                        logger.error(f"Error fetching {base_usd_pair} rate for {pair}: {e}")
                        continue

                # Create signal
                # Use capital_for_trade as position_value to match backtest behavior
                # (Percentage-based P/L: profit = price_pct_change * capital_risked)
                signal = {
                    'pair': pair,
                    'direction': direction,
                    'size': int(position_size),
                    'position_value_usd': capital_for_trade,  # Match backtest: use intended capital, not actual units value
                    'target': target,
                    'breakout_level': breakout_level,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                }

                signals.append(signal)
                logger.info(f"Signal generated: {pair} {direction.upper()} | "
                           f"Size: {int(position_size)} | Confidence: {confidence:.3f}")

            except Exception as e:
                logger.error(f"Error generating signal for {pair}: {e}", exc_info=True)
                continue

        logger.info(f"Generated {len(signals)} signals")
        return signals


# Test code
if __name__ == '__main__':
    from dotenv import load_dotenv
    import sys

    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import load_config

    load_dotenv()

    # Load config
    config = load_config('config.yaml')

    # Initialize broker
    broker = OandaBroker(
        api_key=config.oanda.api_key,
        account_id=config.oanda.account_id,
        account_type=config.oanda.account_type
    )

    # Initialize strategy
    strategy = Strategy15m(config.strategy_15m, broker)

    # Generate signals
    print("\nGenerating signals...")
    signals = strategy.generate_signals(current_capital=500.0, existing_positions={})

    print(f"\nGenerated {len(signals)} signals:")
    for sig in signals:
        print(f"  {sig['pair']} {sig['direction'].upper()} | "
              f"Size: {sig['size']} | Confidence: {sig['confidence']:.3f}")
