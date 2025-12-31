"""
BREAKOUT STRATEGY WITH RANGE-BOUND FILTER
==========================================
Test filtering breakout trades using range_bound_3d predictions.

Hypothesis:
- When model predicts range_bound with high confidence, skip breakout trades
- Should reduce false breakouts in choppy markets
- Should improve performance in losing years (2018, 2020, 2021, 2022, 2024)

Test different confidence thresholds:
- No filter (baseline)
- Skip if range_bound > 60%
- Skip if range_bound > 65%
- Skip if range_bound > 70%
- Skip if range_bound > 75%
- Skip if range_bound > 80%
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("BREAKOUT STRATEGY WITH RANGE-BOUND FILTER")
print("="*100)
print()

# Strategy parameters (optimized breakout config)
INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.007
MIN_CONFIDENCE = 0.70
EMERGENCY_STOP_PCT = -0.04
EMERGENCY_STOP_DAYS = 15
TRAILING_TRIGGER = 0.005
TRAILING_PCT = 0.60

# Test these filter thresholds
FILTER_THRESHOLDS = [None, 0.60, 0.65, 0.70, 0.75, 0.80]

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

def calculate_features(df):
    """Calculate technical features"""
    df['return_1d'] = df['close'].pct_change()
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)

    for period in [10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = ranges.max(axis=1).rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    df['momentum_10d'] = df['close'] - df['close'].shift(10)
    df['momentum_20d'] = df['close'] - df['close'].shift(20)

    df['high_20d'] = df['high'].rolling(20).max()
    df['low_20d'] = df['low'].rolling(20).min()
    df['range_20d'] = (df['high_20d'] - df['low_20d']) / df['close']
    df['position_in_range'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 1e-10)

    return df

def create_range_bound_target(df):
    """Create range_bound_3d target"""
    future_high_3d = df['high'].rolling(3).max().shift(-3)
    future_low_3d = df['low'].rolling(3).min().shift(-3)

    df['range_bound_3d'] = (
        (future_high_3d <= df['high_20d'] * 1.002) &
        (future_low_3d >= df['low_20d'] * 0.998)
    ).astype(int)

    return df

FEATURE_COLS = [
    'return_1d', 'return_3d', 'return_5d', 'return_10d',
    'ema_10', 'ema_20', 'ema_50',
    'price_to_ema_10', 'price_to_ema_20', 'price_to_ema_50',
    'macd', 'macd_signal', 'macd_diff',
    'rsi', 'atr', 'atr_pct',
    'volatility_10d', 'volatility_20d',
    'bb_middle', 'bb_upper', 'bb_lower', 'bb_position', 'bb_width',
    'momentum_10d', 'momentum_20d',
    'high_20d', 'low_20d', 'range_20d', 'position_in_range'
]

class Position:
    """Trading position with emergency stop, trailing stop, and target"""

    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.breakout_target = breakout_target
        self.confidence = confidence

        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.profit_pct = None
        self.days_held = 0
        self.max_profit = 0
        self.trailing_stop = None

    def check_exit(self, date, open_price, high, low, close):
        """Check if position should exit"""
        self.days_held += 1

        if self.direction == 'long':
            current_profit = (close - self.entry_price) / self.entry_price
            intraday_high_profit = (high - self.entry_price) / self.entry_price
            hit_target = high >= self.breakout_target
        else:
            current_profit = (self.entry_price - close) / self.entry_price
            intraday_high_profit = (self.entry_price - low) / self.entry_price
            hit_target = low <= self.breakout_target

        self.max_profit = max(self.max_profit, intraday_high_profit)

        # 1. Emergency stop
        if self.days_held >= EMERGENCY_STOP_DAYS and current_profit < EMERGENCY_STOP_PCT:
            return 'emergency_stop', close, current_profit

        # 2. Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > TRAILING_TRIGGER:
                self.trailing_stop = self.entry_price
        else:
            if self.direction == 'long':
                new_stop = self.entry_price + (high - self.entry_price) * TRAILING_PCT
                self.trailing_stop = max(self.trailing_stop, new_stop)
                hit_stop = low <= self.trailing_stop
            else:
                new_stop = self.entry_price - (self.entry_price - low) * TRAILING_PCT
                self.trailing_stop = min(self.trailing_stop, new_stop)
                hit_stop = high >= self.trailing_stop

            if hit_stop:
                return 'trailing_stop', self.trailing_stop, (self.trailing_stop - self.entry_price) / self.entry_price if self.direction == 'long' else (self.entry_price - self.trailing_stop) / self.entry_price

        # 3. Target
        if hit_target:
            return 'target', self.breakout_target, (self.breakout_target - self.entry_price) / self.entry_price if self.direction == 'long' else (self.entry_price - self.breakout_target) / self.entry_price

        return None, None, None

def train_range_bound_model(all_pairs_data):
    """Train range_bound_3d model on 2016-2017"""
    print("Training range_bound_3d filter model...")

    X_train_list = []
    y_train_list = []

    for pair_data in all_pairs_data.values():
        train_data = pair_data[(pair_data['year'] >= 2016) & (pair_data['year'] <= 2017)].copy()
        train_data = train_data.dropna(subset=FEATURE_COLS + ['range_bound_3d'])

        if len(train_data) > 50:
            X_train_list.append(train_data[FEATURE_COLS])
            y_train_list.append(train_data['range_bound_3d'])

    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    model.fit(X_train, y_train, verbose=False)
    print(f"  Trained on {len(y_train)} samples")
    print()

    return model

def backtest_with_filter(breakout_predictions, range_bound_model, all_pairs_data, filter_threshold):
    """
    Backtest breakout strategy with range_bound filter

    filter_threshold: Skip breakout if range_bound confidence > this value
    None = no filter (baseline)
    """
    capital = INITIAL_CAPITAL
    positions = []

    # Get all dates
    all_dates = set()
    for pair_preds in breakout_predictions.values():
        all_dates.update(pair_preds.index)
    all_dates = sorted(list(all_dates))

    equity_curve = []
    equity_dates = []

    for date in all_dates:
        # Update open positions
        for pos in positions:
            if pos.exit_date is None:
                pair_preds = breakout_predictions[pos.pair]
                if date in pair_preds.index:
                    row = pair_preds.loc[date]
                    exit_reason, exit_price, profit_pct = pos.check_exit(
                        date, row['open'], row['high'], row['low'], row['close']
                    )

                    if exit_reason:
                        pos.exit_date = date
                        pos.exit_price = exit_price
                        pos.exit_reason = exit_reason
                        pos.profit_pct = profit_pct

                        profit_dollars = profit_pct * pos.size * pos.entry_price
                        capital += profit_dollars

        # Check for new entries
        for pair, pair_preds in breakout_predictions.items():
            if date not in pair_preds.index:
                continue

            row = pair_preds.loc[date]

            # Check if already in position
            if any(p.pair == pair and p.exit_date is None for p in positions):
                continue

            # Get breakout predictions
            breakout_high_prob = row['breakout_high_prob']
            breakout_low_prob = row['breakout_low_prob']
            max_prob = max(breakout_high_prob, breakout_low_prob)

            if max_prob <= MIN_CONFIDENCE:
                continue

            # Apply range_bound filter
            if filter_threshold is not None:
                # Get range_bound prediction
                pair_data = all_pairs_data[pair]
                date_data = pair_data[pair_data['date'] == pd.Timestamp(date)]

                if len(date_data) > 0:
                    X = date_data[FEATURE_COLS].values
                    if not np.isnan(X).any():
                        range_bound_prob = range_bound_model.predict_proba(X)[0, 1]

                        # Skip if range_bound confidence too high
                        if range_bound_prob > filter_threshold:
                            continue

            # Entry logic (same as breakout strategy)
            price = row['close']
            high_20d = row['high_20d']
            low_20d = row['low_20d']

            risk_amount = capital * RISK_PER_TRADE
            position_size = risk_amount / (price * 0.02)

            if breakout_high_prob > breakout_low_prob:
                direction = 'long'
                target = high_20d * 1.005
            else:
                direction = 'short'
                target = low_20d * 0.995

            position = Position(pair, date, price, direction, position_size, target, max_prob)
            positions.append(position)

        equity_curve.append(capital)
        equity_dates.append(date)

    # Close remaining positions
    for pos in positions:
        if pos.exit_date is None:
            pair_preds = breakout_predictions[pos.pair]
            last_row = pair_preds.iloc[-1]
            exit_price = last_row['close']

            if pos.direction == 'long':
                profit_pct = (exit_price - pos.entry_price) / pos.entry_price
            else:
                profit_pct = (pos.entry_price - exit_price) / pos.entry_price

            pos.exit_date = pair_preds.index[-1]
            pos.exit_price = exit_price
            pos.exit_reason = 'final'
            pos.profit_pct = profit_pct

            profit_dollars = profit_pct * pos.size * pos.entry_price
            capital += profit_dollars

    # Calculate metrics
    closed_positions = [p for p in positions if p.exit_date is not None]

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    num_trades = len(closed_positions)
    win_rate = sum(1 for p in closed_positions if p.profit_pct > 0) / num_trades if num_trades > 0 else 0

    # Max drawdown
    if len(equity_curve) > 0:
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        dd = (equity_series - running_max) / running_max
        max_dd = dd.min()
    else:
        max_dd = 0

    # Trades skipped (if filter applied)
    if filter_threshold is not None:
        # Count how many signals were generated without filter
        total_signals = 0
        for pair, pair_preds in breakout_predictions.items():
            for date in pair_preds.index:
                row = pair_preds.loc[date]
                max_prob = max(row['breakout_high_prob'], row['breakout_low_prob'])
                if max_prob > MIN_CONFIDENCE:
                    total_signals += 1

        trades_skipped = total_signals - num_trades
    else:
        trades_skipped = 0

    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'trades_skipped': trades_skipped,
        'positions': closed_positions
    }

# Load data
print("Loading data...")
all_pairs_data = {}

for pair in PAIRS:
    file_path = os.path.join(DATA_DIR, f'{pair}_1day_oanda.csv')
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['year'] = pd.to_datetime(df['date']).dt.year
    df = df[(df['year'] >= 2016) & (df['year'] <= 2025)].copy()

    df = calculate_features(df)
    df = create_range_bound_target(df)

    all_pairs_data[pair] = df

print(f"Loaded {len(all_pairs_data)} pairs")
print()

# Train range_bound filter model
range_bound_model = train_range_bound_model(all_pairs_data)

# Load breakout predictions
print("Loading breakout predictions...")
try:
    with open('model_predictions.pkl', 'rb') as f:
        all_predictions = pickle.load(f)
    print(f"Loaded predictions for {len(all_predictions)} validation periods")
    print()
except FileNotFoundError:
    print("ERROR: model_predictions.pkl not found!")
    print("Run generate_predictions.py first.")
    exit(1)

# Test different filter thresholds
print("="*100)
print("TESTING FILTER THRESHOLDS")
print("="*100)
print()

results = []

for threshold in FILTER_THRESHOLDS:
    if threshold is None:
        print(f"Testing NO FILTER (baseline)...")
    else:
        print(f"Testing filter threshold: {threshold*100:.0f}% (skip if range_bound > {threshold*100:.0f}%)...")

    # Run backtest for each period
    period_results = []

    for period_name, period_preds in all_predictions.items():
        result = backtest_with_filter(period_preds, range_bound_model, all_pairs_data, threshold)
        period_results.append(result)

    # Aggregate metrics
    avg_return = np.mean([r['total_return'] for r in period_results])
    avg_trades = np.mean([r['num_trades'] for r in period_results])
    avg_win_rate = np.mean([r['win_rate'] for r in period_results])
    avg_dd = np.mean([r['max_dd'] for r in period_results])
    total_skipped = sum([r['trades_skipped'] for r in period_results])

    profitable_periods = sum(1 for r in period_results if r['total_return'] > 0)

    print(f"  Avg Return: {avg_return:+.1%}")
    print(f"  Avg Trades: {avg_trades:.0f} per period")
    print(f"  Avg Win Rate: {avg_win_rate:.0%}")
    print(f"  Avg DD: {avg_dd:.1%}")
    print(f"  Profitable Periods: {profitable_periods}/{len(period_results)}")
    if threshold is not None:
        print(f"  Trades Skipped: {total_skipped} total")
    print()

    results.append({
        'threshold': threshold,
        'avg_return': avg_return,
        'avg_trades': avg_trades,
        'avg_win_rate': avg_win_rate,
        'avg_dd': avg_dd,
        'profitable_periods': profitable_periods,
        'total_periods': len(period_results),
        'trades_skipped': total_skipped
    })

# Summary
print("="*100)
print("RESULTS COMPARISON")
print("="*100)
print()

results_df = pd.DataFrame(results)

print(f"{'Filter':<12} {'Avg Return':>12} {'Trades':>8} {'Win%':>6} {'MaxDD':>8} {'Profitable':>11} {'Skipped':>10}")
print("-" * 85)

for _, row in results_df.iterrows():
    if row['threshold'] is None:
        filter_str = "None"
    else:
        filter_str = f">{row['threshold']*100:.0f}%"

    print(f"{filter_str:<12} {row['avg_return']:>11.1%} {row['avg_trades']:>8.0f} "
          f"{row['avg_win_rate']:>5.0%} {row['avg_dd']:>7.1%} "
          f"{row['profitable_periods']:>4}/{row['total_periods']:<4} "
          f"{row['trades_skipped']:>10.0f}")

print()
print("="*100)
print("RECOMMENDATION")
print("="*100)
print()

baseline = results_df[results_df['threshold'].isna()].iloc[0]
filtered_results = results_df[results_df['threshold'].notna()].sort_values('avg_return', ascending=False)

if len(filtered_results) > 0:
    best_filtered = filtered_results.iloc[0]

    improvement = best_filtered['avg_return'] - baseline['avg_return']

    if improvement > 0.05:
        print(f"SUCCESS: Filter improves performance!")
        print(f"  Best threshold: >{best_filtered['threshold']*100:.0f}%")
        print(f"  Improvement: {improvement:+.1%}")
        print(f"  New avg return: {best_filtered['avg_return']:+.1%}")
        print(f"  Trades reduced: {baseline['avg_trades']:.0f} -> {best_filtered['avg_trades']:.0f}")
        print()
        print("The range_bound filter successfully avoids bad breakout trades!")
    elif improvement > 0:
        print(f"MARGINAL: Small improvement with filter")
        print(f"  Best threshold: >{best_filtered['threshold']*100:.0f}%")
        print(f"  Improvement: {improvement:+.1%}")
        print(f"  May not be worth the added complexity")
    else:
        print("NO IMPROVEMENT: Filter does not help breakout strategy")
        print(f"  Best filtered return: {best_filtered['avg_return']:+.1%}")
        print(f"  Baseline return: {baseline['avg_return']:+.1%}")
        print(f"  Difference: {improvement:+.1%}")
        print()
        print("Stick with unfiltered breakout strategy.")
