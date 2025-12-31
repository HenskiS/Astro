"""
TEST MEAN REVERSION WITH DIFFERENT STOP LEVELS
===============================================
Test range_bound_3d strategy with various stop loss levels.

Goal: Find the sweet spot where we cut losses quickly without getting stopped out too early.

Current issue:
- 70% win rate (good accuracy)
- Avg win: +0.86%
- Avg loss: -2.99% (too large!)

Test stop levels: -1%, -1.5%, -2%, -2.5%, -3%, -4%
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TESTING STOP LOSS LEVELS FOR MEAN REVERSION")
print("="*100)
print()

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
LOSING_YEARS = [2018, 2020, 2021]
WINNING_YEARS = [2019]

INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.005
MIN_CONFIDENCE = 0.70
EDGE_THRESHOLD_LOW = 0.3
EDGE_THRESHOLD_HIGH = 0.7
COOLDOWN_DAYS = 3

# Test these stop levels
STOP_LEVELS = [-0.01, -0.015, -0.02, -0.025, -0.03, -0.04]

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

def train_model(all_pairs_data):
    """Train range_bound_3d model"""
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
    return model

def backtest_with_stop(model, all_pairs_data, test_years, emergency_stop_pct):
    """Backtest with specific stop level"""
    capital = INITIAL_CAPITAL
    trades = []
    open_positions = {}
    last_trade_date = {}

    # Combine all data
    all_data = []
    for pair, pair_data in all_pairs_data.items():
        test_data = pair_data[pair_data['year'].isin(test_years)].copy()
        test_data = test_data.dropna(subset=FEATURE_COLS + ['range_bound_3d'])
        test_data['pair'] = pair
        all_data.append(test_data)

    combined_data = pd.concat(all_data).sort_values('date').reset_index(drop=True)

    for idx, row in combined_data.iterrows():
        pair = row['pair']
        current_date = row['date']
        current_price = row['close']

        # Entry
        if pair not in open_positions:
            if pair in last_trade_date:
                days_since_last = (current_date - last_trade_date[pair]).days
                if days_since_last < COOLDOWN_DAYS:
                    continue

            X = row[FEATURE_COLS].values.reshape(1, -1)
            prob = model.predict_proba(X)[0, 1]

            if prob > MIN_CONFIDENCE:
                position_in_range = row['position_in_range']

                if position_in_range < EDGE_THRESHOLD_LOW:
                    direction = 'long'
                    entry_price = current_price
                    target_price = (row['high_20d'] + row['low_20d']) / 2

                elif position_in_range > EDGE_THRESHOLD_HIGH:
                    direction = 'short'
                    entry_price = current_price
                    target_price = (row['high_20d'] + row['low_20d']) / 2

                else:
                    continue

                risk_amount = capital * RISK_PER_TRADE
                stop_distance = abs(entry_price * 0.02)
                position_size = risk_amount / stop_distance

                open_positions[pair] = {
                    'entry_date': current_date,
                    'entry_price': entry_price,
                    'direction': direction,
                    'target_price': target_price,
                    'position_size': position_size,
                    'max_profit': 0,
                    'days_held': 0
                }

        # Check exits
        if pair in open_positions:
            pos = open_positions[pair]
            pos['days_held'] += 1

            if pos['direction'] == 'long':
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            else:
                pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']

            pos['max_profit'] = max(pos['max_profit'], pnl_pct)

            exit_triggered = False
            exit_reason = None
            exit_price = current_price

            # 1. Target
            if pos['direction'] == 'long' and current_price >= pos['target_price']:
                exit_triggered = True
                exit_reason = 'target'
                exit_price = pos['target_price']
            elif pos['direction'] == 'short' and current_price <= pos['target_price']:
                exit_triggered = True
                exit_reason = 'target'
                exit_price = pos['target_price']

            # 2. Trailing stop (once up >0.5%, lock 50%)
            if not exit_triggered and pos['max_profit'] > 0.005:
                trailing_level = pos['max_profit'] * 0.5
                if pnl_pct < trailing_level:
                    exit_triggered = True
                    exit_reason = 'trailing_stop'

            # 3. Emergency stop (immediate, no day requirement)
            if not exit_triggered and pnl_pct < emergency_stop_pct:
                exit_triggered = True
                exit_reason = 'emergency_stop'

            if exit_triggered:
                if pos['direction'] == 'long':
                    final_pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                else:
                    final_pnl_pct = (pos['entry_price'] - exit_price) / pos['entry_price']

                profit_dollars = final_pnl_pct * pos['position_size'] * pos['entry_price']
                capital += profit_dollars

                trades.append({
                    'pair': pair,
                    'pnl_pct': final_pnl_pct,
                    'exit_reason': exit_reason
                })

                last_trade_date[pair] = current_date
                del open_positions[pair]

    # Close remaining
    for pair, pos in open_positions.items():
        exit_price = combined_data[combined_data['pair'] == pair].iloc[-1]['close']

        if pos['direction'] == 'long':
            final_pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
        else:
            final_pnl_pct = (pos['entry_price'] - exit_price) / pos['entry_price']

        profit_dollars = final_pnl_pct * pos['position_size'] * pos['entry_price']
        capital += profit_dollars

        trades.append({
            'pair': pair,
            'pnl_pct': final_pnl_pct,
            'exit_reason': 'final'
        })

    # Calculate metrics
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
        num_trades = len(trades_df)
        win_rate = (trades_df['pnl_pct'] > 0).sum() / num_trades
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if (trades_df['pnl_pct'] > 0).any() else 0
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if (trades_df['pnl_pct'] < 0).any() else 0

        num_stops = (trades_df['exit_reason'] == 'emergency_stop').sum()

        return {
            'stop_level': emergency_stop_pct,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'num_stops': num_stops,
            'stop_pct': num_stops / num_trades if num_trades > 0 else 0
        }
    else:
        return None

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
    df = df[(df['year'] >= 2016) & (df['year'] <= 2021)].copy()

    df = calculate_features(df)
    df = create_range_bound_target(df)

    all_pairs_data[pair] = df

print(f"Loaded {len(all_pairs_data)} pairs")
print()

# Train model
print("Training model...")
model = train_model(all_pairs_data)
print()

# Test different stop levels
print("="*100)
print("TESTING STOP LEVELS")
print("="*100)
print()

results = []

for stop_level in STOP_LEVELS:
    print(f"Testing {stop_level*100:.1f}% stop...")

    losing_result = backtest_with_stop(model, all_pairs_data, LOSING_YEARS, stop_level)
    winning_result = backtest_with_stop(model, all_pairs_data, WINNING_YEARS, stop_level)

    if losing_result and winning_result:
        edge = losing_result['total_return'] - winning_result['total_return']

        results.append({
            'stop': stop_level,
            'losing_return': losing_result['total_return'],
            'losing_trades': losing_result['num_trades'],
            'losing_win': losing_result['win_rate'],
            'losing_avg_win': losing_result['avg_win'],
            'losing_avg_loss': losing_result['avg_loss'],
            'losing_stops': losing_result['num_stops'],
            'winning_return': winning_result['total_return'],
            'winning_trades': winning_result['num_trades'],
            'winning_win': winning_result['win_rate'],
            'edge': edge
        })

print()
print("="*100)
print("RESULTS")
print("="*100)
print()

results_df = pd.DataFrame(results).sort_values('losing_return', ascending=False)

print(f"{'Stop':<8} {'Losing Yrs':>11} {'Trades':>7} {'Win%':>5} {'Avg W':>7} {'Avg L':>7} {'Stops':>6}   "
      f"{'Win Year':>11} {'Trades':>7} {'Win%':>5}   {'Edge':>7}")
print("-" * 115)

for _, row in results_df.iterrows():
    print(f"{row['stop']*100:>6.1f}%  {row['losing_return']:>10.1%} {row['losing_trades']:>7.0f} {row['losing_win']:>4.0%} "
          f"{row['losing_avg_win']:>6.2%} {row['losing_avg_loss']:>6.2%} {row['losing_stops']:>6.0f}   "
          f"{row['winning_return']:>10.1%} {row['winning_trades']:>7.0f} {row['winning_win']:>4.0%}   "
          f"{row['edge']:>6.1%}")

print()
print("="*100)
print("RECOMMENDATION")
print("="*100)
print()

best = results_df.iloc[0]
print(f"Best stop level: {best['stop']*100:.1f}%")
print(f"  Losing years: {best['losing_return']:+.1%} ({best['losing_trades']:.0f} trades, {best['losing_win']:.0%} win)")
print(f"  Winning year: {best['winning_return']:+.1%} ({best['winning_trades']:.0f} trades, {best['winning_win']:.0%} win)")
print(f"  Edge: {best['edge']:+.1%}")
print()
print(f"  Avg win: {best['losing_avg_win']:+.2%}")
print(f"  Avg loss: {best['losing_avg_loss']:+.2%}")
print(f"  Emergency stops: {best['losing_stops']:.0f} ({best['losing_stops']/best['losing_trades']*100:.0f}% of trades)")
print()

if best['losing_return'] > 0.10 and best['edge'] > 0.05:
    print("SUCCESS: Tighter stop improves profitability!")
    print(f"Mean reversion strategy is viable with {best['stop']*100:.1f}% stop.")
    print()
    print("Next steps:")
    print("  1. Full backtest on 2016-2025")
    print("  2. Test combining with breakout strategy")
elif best['losing_return'] > 0:
    print("MARGINAL: Small positive returns, may not be worth complexity.")
else:
    print("FAILED: Mean reversion still unprofitable even with tighter stops.")
    print("The strategy fundamentally doesn't work - abandon and stick with breakouts.")
