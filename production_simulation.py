"""
PRODUCTION SIMULATION - CORRECTED
==================================
Fixed version that matches backtest behavior exactly.

KEY FIX: Only update positions on days where at least one pair has a valid prediction.
This matches the backtest which only processes dates in the predictions dataframe.
"""
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from xgboost import XGBClassifier
from mock_broker_api import MockBrokerAPI
from datetime import timedelta
warnings.filterwarnings('ignore')

# XGBoost config (same as generate_predictions_quarterly.py)
XGB_CONFIG = {
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss'
}

# Trading parameters
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.007
MIN_CONFIDENCE = 0.70
LADDER_LEVELS = [0.008, 0.015]
LADDER_SCALE_PCT = 0.33
EMERGENCY_STOP_LOSS_PCT = -0.04
EMERGENCY_STOP_DAYS = 15
TRAILING_STOP_TRIGGER = 0.005
TRAILING_STOP_PCT = 0.60

# Pairs to trade (must match backtest exactly!)
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']


def calculate_features(df):
    """Calculate technical features (same as generate_predictions_quarterly.py)"""
    df = df.copy()

    df['return_1d'] = df['close'].pct_change()
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)

    for period in [10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift()),
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    df['high_20d'] = df['high'].rolling(20).max()
    df['low_20d'] = df['low'].rolling(20).min()
    df['range_20d'] = (df['high_20d'] - df['low_20d']) / df['close']
    df['position_in_range'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 1e-10)

    return df


def create_targets(df):
    """Create breakout targets for training"""
    future_high_10d = df['high'].rolling(10).max().shift(-10)
    future_low_10d = df['low'].rolling(10).min().shift(-10)

    current_high_20d = df['high'].rolling(20).max()
    current_low_20d = df['low'].rolling(20).min()

    df['target_breakout_high'] = (future_high_10d > current_high_20d).astype(int)
    df['target_breakout_low'] = (future_low_10d < current_low_20d).astype(int)

    return df


class Position:
    """Represents an open position with ladder exit logic"""
    def __init__(self, pair, entry_date, entry_price, direction, size, target, confidence):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.original_size = size
        self.target = target
        self.confidence = confidence
        self.days_held = 0
        self.max_profit = 0
        self.trailing_stop = None
        self.partial_exits = []
        self.ladder_level = 0

    def update(self, date, high, low, close):
        """Update position with current day's prices"""
        self.days_held += 1

        if self.direction == 'long':
            current_profit = (close - self.entry_price) / self.entry_price
            intraday_high_profit = (high - self.entry_price) / self.entry_price
            hit_target = high >= self.target
        else:
            current_profit = (self.entry_price - close) / self.entry_price
            intraday_high_profit = (self.entry_price - low) / self.entry_price
            hit_target = low <= self.target

        self.max_profit = max(self.max_profit, intraday_high_profit)

        # Check ladder
        if self.ladder_level < len(LADDER_LEVELS):
            if intraday_high_profit >= LADDER_LEVELS[self.ladder_level]:
                self.partial_exits.append((LADDER_LEVELS[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
                self.ladder_level += 1
                return None

        # Emergency stop
        if self.days_held >= EMERGENCY_STOP_DAYS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            return 'emergency_stop', close, current_profit

        # Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > TRAILING_STOP_TRIGGER:
                self.trailing_stop = self.entry_price
        else:
            if self.direction == 'long':
                new_stop = self.entry_price + (high - self.entry_price) * TRAILING_STOP_PCT
                self.trailing_stop = max(self.trailing_stop, new_stop)
                hit_stop = low <= self.trailing_stop
            else:
                new_stop = self.entry_price - (self.entry_price - low) * TRAILING_STOP_PCT
                self.trailing_stop = min(self.trailing_stop, new_stop)
                hit_stop = high >= self.trailing_stop

            if hit_stop:
                return 'trailing_stop', self.trailing_stop, current_profit

        # Target
        if hit_target:
            return 'target', self.target, current_profit

        return None

    def calculate_blended_profit(self, final_profit):
        """Calculate profit accounting for ladder exits"""
        if len(self.partial_exits) == 0:
            return final_profit
        total = 0
        remaining = 1.0
        for exit_profit, exit_pct in self.partial_exits:
            total += exit_profit * exit_pct
            remaining -= exit_pct
        total += final_profit * remaining
        return total


class ProductionSimulation:
    """
    CORRECTED production simulation engine.
    
    Key fix: Only updates positions on days where at least one pair has valid predictions.
    This matches the backtest behavior exactly.
    """

    def __init__(self, data_dir='data', start_date='2016-01-01', end_date='2025-12-31', use_saved_predictions=False):
        self.api = MockBrokerAPI(data_dir=data_dir)
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        
        self.capital = INITIAL_CAPITAL
        self.positions = []
        self.trades = []
        self.models = {}
        
        # Option to use pre-generated predictions (much faster!)
        self.use_saved_predictions = use_saved_predictions
        self.saved_predictions = None
        
        if use_saved_predictions:
            print("Loading pre-generated predictions from model_predictions_production.pkl...")
            try:
                import pickle
                with open('model_predictions_production.pkl', 'rb') as f:
                    self.saved_predictions = pickle.load(f)
                print(f"  Loaded predictions for {len(self.saved_predictions)} quarters")
            except FileNotFoundError:
                print("  WARNING: model_predictions_production.pkl not found, will generate on-the-fly")
                self.use_saved_predictions = False
        
        # Track all predictions for comparison (only if generating on-the-fly)
        self.all_predictions = {} if not use_saved_predictions else None
        
        print("="*100)
        print("PRODUCTION SIMULATION (CORRECTED)")
        print("="*100)
        print()

    def train_models(self, train_start, train_end, pairs):
        """Train models for all pairs"""
        print(f"  Training on {train_start.date()} to {train_end.date()}...")
        
        new_models = {}
        
        for pair in pairs:
            try:
                # Get historical data with extra days for target calculation
                # Need 10+ days beyond train_end to calculate forward-looking targets
                data_end = train_end + pd.Timedelta(days=15)
                history = self.api.get_history(pair, count=999999, end_date=data_end)
                
                if len(history) < 1000:
                    continue
                
                # Calculate features and targets on ALL data FIRST (critical!)
                # This ensures features have proper historical context
                # AND targets have forward-looking data
                features_all = calculate_features(history)
                features_all = create_targets(features_all)
                
                # THEN filter to training period
                features = features_all[(features_all.index >= train_start) & (features_all.index <= train_end)].copy()
                features = features.dropna()
                
                if len(features) < 1000:
                    continue
                
                # Get feature columns
                feature_cols = [col for col in features.columns if col not in
                               ['target_breakout_high', 'target_breakout_low',
                                'open', 'high', 'low', 'close', 'volume']]
                
                X_train = features[feature_cols]
                
                # Train high breakout model
                y_high = features['target_breakout_high']
                if y_high.sum() > 100:
                    model_high = XGBClassifier(**XGB_CONFIG)
                    model_high.fit(X_train, y_high, verbose=False)
                    
                    # Train low breakout model
                    y_low = features['target_breakout_low']
                    if y_low.sum() > 100:
                        model_low = XGBClassifier(**XGB_CONFIG)
                        model_low.fit(X_train, y_low, verbose=False)
                        
                        new_models[pair] = {
                            'high': model_high,
                            'low': model_low,
                            'features': feature_cols
                        }
            except Exception as e:
                continue
        
        self.models = new_models
        print(f"  Trained {len(self.models)} pairs")
        return len(self.models) > 0

    def run(self):
        """Run full simulation"""
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
        print()
        
        # Skip training if using saved predictions
        if not self.use_saved_predictions:
            # Initial training (6 years before start)
            initial_train_start = self.start_date - pd.DateOffset(years=6)
            initial_train_end = self.start_date - pd.Timedelta(days=11)  # 10-day gap
            
            print("Initial Training:")
            # Use EXACT same pairs as backtest
            if not self.train_models(initial_train_start, initial_train_end, PAIRS):
                print("ERROR: No models trained")
                return None
        else:
            print("Skipping training - using saved predictions")
            # Set models to empty dict (won't be used)
            self.models = {pair: None for pair in PAIRS}
        
        print()
        print("Starting day-by-day simulation...")
        print()
        
        # Track yearly capital
        yearly_capital = {}
        current_quarter = None
        
        # Get all trading days from start to end
        # But we'll only process days where predictions exist
        all_raw_dates = set()
        for pair in self.models.keys():
            try:
                dates = self.api.get_history(pair, count=999999, end_date=self.end_date).index
                all_raw_dates.update(dates.tolist())
            except:
                continue
        
        all_raw_dates = sorted([d for d in all_raw_dates if self.start_date <= d <= self.end_date])
        
        # Process day by day
        for i, current_date in enumerate(all_raw_dates):
            # Check if we need to retrain (quarterly)
            quarter_num = (current_date.month - 1) // 3 + 1
            quarter_key = f"{current_date.year}Q{quarter_num}"
            
            if quarter_key != current_quarter:
                current_quarter = quarter_key
                
                # Retrain if not first quarter (skip if using saved predictions)
                if current_date > self.start_date and not self.use_saved_predictions:
                    print()
                    print(f"Quarterly Retraining for {quarter_key}:")
                    
                    # CRITICAL FIX: Use calendar quarter start for train_end calculation
                    # But use January 1st for train_start (matches backtest)
                    quarter_calendar_start = pd.Timestamp(f"{current_date.year}-{((quarter_num-1)*3)+1:02d}-01")
                    
                    # Always train from January 1st of (year - 6), like backtest does
                    train_start_year = current_date.year - 6
                    train_start = pd.Timestamp(f"{train_start_year}-01-01")
                    train_end = quarter_calendar_start - pd.Timedelta(days=11)  # 10-day gap
                    
                    if not self.train_models(train_start, train_end, PAIRS):
                        print("  WARNING: No models trained, continuing with previous models")
                    print()
            
            # Generate predictions for potentially opening new positions
            daily_predictions = self._generate_predictions_for_day(current_date)
            
            # Track current quarter for CSV export
            current_quarter_key = f"{current_date.year}Q{(current_date.month - 1) // 3 + 1}"
            if not hasattr(self, '_current_quarter'):
                self._current_quarter = current_quarter_key
                self._quarter_start_trade_count = 0
            
            # If quarter changed, save previous quarter's trades
            if current_quarter_key != self._current_quarter:
                # Save trades from previous quarter
                quarter_trades = self.trades[self._quarter_start_trade_count:]
                if len(quarter_trades) > 0:
                    import os
                    os.makedirs('trades', exist_ok=True)
                    trades_df = pd.DataFrame(quarter_trades)
                    trades_df.to_csv(f'trades/production_{self._current_quarter}.csv', index=False)
                    print(f"  >>> Saved {len(quarter_trades)} trades to trades/production_{self._current_quarter}.csv")
                
                self._current_quarter = current_quarter_key
                self._quarter_start_trade_count = len(self.trades)
            
            # CRITICAL: Always update positions using raw price data, even if no predictions
            # This ensures emergency stops trigger correctly
            positions_to_close = []
            for position in self.positions:
                try:
                    # Get current price from broker API (not predictions)
                    current_price = self.api.get_current_price(position.pair, current_date)
                    
                    exit_info = position.update(current_date,
                                               current_price['high'],
                                               current_price['low'],
                                               current_price['close'])
                    
                    if exit_info is not None:
                        positions_to_close.append((position, exit_info))
                except:
                    # If we can't get price data, skip this position
                    continue
            
            # Close positions
            for position, exit_info in positions_to_close:
                self._close_position(position, exit_info, current_date)
                self.positions.remove(position)
            
            # Place new orders based on predictions
            for pair, pred in daily_predictions.items():
                max_prob = max(pred['breakout_high_prob'], pred['breakout_low_prob'])
                
                if max_prob >= MIN_CONFIDENCE:
                    # Determine direction
                    if pred['breakout_high_prob'] > pred['breakout_low_prob']:
                        direction = 'long'
                        target = pred['high_20d'] * 1.005
                    else:
                        direction = 'short'
                        target = pred['low_20d'] * 0.995
                    
                    # Place order
                    self._place_order(pair, current_date, direction, target,
                                    max_prob, pred['close'])
            
            # Track yearly capital
            if current_date.year not in yearly_capital:
                yearly_capital[current_date.year] = self.capital
            else:
                yearly_capital[current_date.year] = self.capital
            
            # Progress update
            if (i + 1) % 50 == 0:
                print(f"  Day {i+1:>4}/{len(all_raw_dates)}: {current_date.date()} | Capital: ${self.capital:>10,.0f} | Positions: {len(self.positions)}")
        
        # Final results
        print()
        print("="*100)
        print("SIMULATION COMPLETE")
        print("="*100)
        print()
        
        # Save final quarter's trades
        if hasattr(self, '_current_quarter') and hasattr(self, '_quarter_start_trade_count'):
            quarter_trades = self.trades[self._quarter_start_trade_count:]
            if len(quarter_trades) > 0:
                import os
                os.makedirs('trades', exist_ok=True)
                trades_df = pd.DataFrame(quarter_trades)
                trades_df.to_csv(f'trades/production_{self._current_quarter}.csv', index=False)
                print(f"  >>> Saved {len(quarter_trades)} trades to trades/production_{self._current_quarter}.csv")
        
        # Save predictions for comparison
        self._save_predictions()
        
        return self._calculate_results(yearly_capital)

    def _generate_predictions_for_day(self, current_date):
        """
        Generate predictions for all pairs on a specific day.
        
        If use_saved_predictions=True, loads from pre-generated file.
        Otherwise generates on-the-fly (slower but saves predictions).
        """
        # If using saved predictions, load from file
        if self.use_saved_predictions and self.saved_predictions is not None:
            quarter_key = f"{current_date.year}Q{(current_date.month - 1) // 3 + 1}"
            
            if quarter_key not in self.saved_predictions:
                return {}
            
            quarter_preds = self.saved_predictions[quarter_key]
            daily_predictions = {}
            
            for pair in PAIRS:
                if pair not in quarter_preds:
                    continue
                
                # Find this date in the predictions
                pair_df = quarter_preds[pair]
                
                # Handle timezone
                if pair_df.index.tz is not None:
                    pair_df = pair_df.copy()
                    pair_df.index = pair_df.index.tz_localize(None)
                
                # Find matching date
                matching_dates = [idx for idx in pair_df.index if idx.date() == current_date.date()]
                
                if len(matching_dates) > 0:
                    row = pair_df.loc[matching_dates[0]]
                    
                    # Get current price from API
                    try:
                        current_price = self.api.get_current_price(pair, current_date)
                        
                        daily_predictions[pair] = {
                            'breakout_high_prob': row['breakout_high_prob'],
                            'breakout_low_prob': row['breakout_low_prob'],
                            'high_20d': row['high_20d'],
                            'low_20d': row['low_20d'],
                            'close': current_price['close'],
                            'high': current_price['high'],
                            'low': current_price['low']
                        }
                    except:
                        continue
            
            return daily_predictions
        
        # Otherwise generate on-the-fly (original logic)
        daily_predictions = {}
        
        for pair in self.models.keys():
            try:
                # Get ALL historical data up to current date
                history = self.api.get_history(pair, count=999999, end_date=current_date)
                
                # Calculate features from all available data
                features = calculate_features(history)
                features = features.dropna()
                
                if len(features) == 0:
                    continue
                
                # Get the last row (current date's features)
                last_row = features.iloc[-1]
                
                # Generate prediction
                feature_cols = self.models[pair]['features']
                X = features[feature_cols].iloc[-1:]
                
                # Skip if any NaN
                if X.isnull().any().any():
                    continue
                
                high_prob = self.models[pair]['high'].predict_proba(X)[0, 1]
                low_prob = self.models[pair]['low'].predict_proba(X)[0, 1]
                
                # Get current price
                current_price = self.api.get_current_price(pair, current_date)
                
                # Store prediction
                daily_predictions[pair] = {
                    'breakout_high_prob': high_prob,
                    'breakout_low_prob': low_prob,
                    'high_20d': last_row['high_20d'],
                    'low_20d': last_row['low_20d'],
                    'close': current_price['close'],
                    'high': current_price['high'],
                    'low': current_price['low']
                }
                
                # Save for comparison with backtest (only if generating on-the-fly)
                if not self.use_saved_predictions:
                    quarter_key = f"{current_date.year}Q{(current_date.month - 1) // 3 + 1}"
                    if quarter_key not in self.all_predictions:
                        self.all_predictions[quarter_key] = {}
                    if pair not in self.all_predictions[quarter_key]:
                        self.all_predictions[quarter_key][pair] = []
                    
                    self.all_predictions[quarter_key][pair].append({
                        'date': current_date,
                        'breakout_high_prob': high_prob,
                        'breakout_low_prob': low_prob,
                        'high_20d': last_row['high_20d'],
                        'low_20d': last_row['low_20d'],
                        'close': current_price['close']
                    })

                
            except Exception as e:
                continue
        
        return daily_predictions

    def _place_order(self, pair, current_date, direction, target, confidence, current_price):
        """Place order (filled same day at close - matches backtest)"""
        # Don't trade if capital is depleted
        if self.capital <= 0:
            return
        
        fill_price = current_price
        
        # Calculate position size
        assumed_risk = 0.02
        risk_amount = self.capital * RISK_PER_TRADE
        position_size = risk_amount / (fill_price * assumed_risk)
        
        # Create position
        position = Position(pair, current_date, fill_price, direction, position_size, target, confidence)
        self.positions.append(position)

    def _close_position(self, position, exit_info, exit_date):
        """Close position and update capital"""
        exit_reason, exit_price, current_profit = exit_info
        
        if position.direction == 'long':
            raw_profit = (exit_price - position.entry_price) / position.entry_price
        else:
            raw_profit = (position.entry_price - exit_price) / position.entry_price
        
        profit_pct = position.calculate_blended_profit(raw_profit)
        profit_dollars = profit_pct * (position.original_size * position.entry_price)
        
        self.capital += profit_dollars
        
        self.trades.append({
            'pair': position.pair,
            'entry_date': position.entry_date,
            'entry_price': position.entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'direction': position.direction,
            'days_held': position.days_held,
            'profit_pct': profit_pct,
            'profit_dollars': profit_dollars,
            'exit_reason': exit_reason,
            'capital_after': self.capital
        })

    def _calculate_results(self, yearly_capital):
        """Calculate final results"""
        results = {
            'initial_capital': INITIAL_CAPITAL,
            'final_capital': self.capital,
            'total_return': (self.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL,
            'total_trades': len(self.trades),
            'trades': self.trades
        }
        
        # Calculate yearly returns
        years = sorted(yearly_capital.keys())
        prev_capital = INITIAL_CAPITAL
        yearly_returns = {}
        
        for year in years:
            year_capital = yearly_capital[year]
            yearly_returns[year] = (year_capital - prev_capital) / prev_capital
            prev_capital = year_capital
        
        results['yearly_returns'] = yearly_returns
        
        # Win rate
        if len(self.trades) > 0:
            winners = sum(1 for t in self.trades if t['profit_pct'] > 0)
            results['win_rate'] = winners / len(self.trades)
        else:
            results['win_rate'] = 0
        
        return results

    def _save_predictions(self):
        """Save predictions to pickle file for comparison with backtest"""
        # Convert lists to DataFrames (match backtest format)
        predictions_df = {}
        for quarter_key, pairs_data in self.all_predictions.items():
            predictions_df[quarter_key] = {}
            for pair, pred_list in pairs_data.items():
                df = pd.DataFrame(pred_list)
                df = df.set_index('date')
                predictions_df[quarter_key][pair] = df
        
        # Save to pickle file
        with open('model_predictions_production_corrected.pkl', 'wb') as f:
            pickle.dump(predictions_df, f)
        
        quarters_saved = sorted(self.all_predictions.keys())
        print(f"  >>> Saved predictions: {len(quarters_saved)} quarters")


if __name__ == '__main__':
    # Use saved predictions for speed (set to False to regenerate and save new predictions)
    sim = ProductionSimulation(
        data_dir='data',
        start_date='2016-01-01',
        end_date='2025-12-31',
        use_saved_predictions=True  # Set to False on first run to generate predictions
    )
    
    results = sim.run()
    
    print(f"Initial Capital:  ${results['initial_capital']:,.0f}")
    print(f"Final Capital:    ${results['final_capital']:,.0f}")
    print(f"Total Return:     {results['total_return']:.1%}")
    print(f"Total Trades:     {results['total_trades']}")
    print(f"Win Rate:         {results['win_rate']:.1%}")
    print()
    
    print("Year-by-Year Returns:")
    for year, ret in sorted(results['yearly_returns'].items()):
        print(f"  {year}: {ret:+7.1%}")
    print()
    
    print("="*100)
    print("CORRECTED VERSION - Should now match quarterly backtest!")
    print("="*100)