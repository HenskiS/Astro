"""
MEAN REVERSION BACKTEST - Range Bound Strategy
===============================================
Proper backtest with real entry/exit logic for range_bound_3d target.

Strategy:
1. Predict when price will stay range-bound (3-day horizon)
2. Enter when near range edges (position_in_range < 0.3 or > 0.7)
3. Direction: Long if near bottom, short if near top
4. Target: Exit at range middle
5. Stops: Emergency stop (-4% after 15 days), trailing stop (50% of max profit)

Compare performance in:
- Losing years: 2018, 2020, 2021
- Winning year: 2019
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import pickle
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("MEAN REVERSION BACKTEST - RANGE BOUND STRATEGY")
print("="*100)
print()

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
LOSING_YEARS = [2018, 2020, 2021]
WINNING_YEARS = [2019]

# Risk parameters
INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.005  # 0.5% risk per trade
MIN_CONFIDENCE = 0.70   # 70% confidence threshold

# Entry parameters
EDGE_THRESHOLD_LOW = 0.3   # Enter long if position_in_range < 0.3
EDGE_THRESHOLD_HIGH = 0.7  # Enter short if position_in_range > 0.7

# Exit parameters
EMERGENCY_STOP_PCT = -0.04  # -4% stop loss
EMERGENCY_STOP_DAYS = 15    # After 15 days
TRAILING_THRESHOLD = 0.005  # Activate trailing stop once up 0.5%
TRAILING_LOCK_PCT = 0.5     # Lock in 50% of max profit

COOLDOWN_DAYS = 3  # Days between trades on same pair

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
    """Train range_bound_3d model on 2016-2017"""
    print("Training range_bound_3d model on 2016-2017...")

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

def backtest_mean_reversion(model, all_pairs_data, test_years, year_label):
    """
    Backtest mean reversion strategy

    Entry:
    - Model predicts range_bound_3d with >70% confidence
    - Price is near range edge (position_in_range < 0.3 or > 0.7)
    - Direction: Long if near bottom, short if near top

    Exit:
    - Target: Range middle (middle of high_20d and low_20d)
    - Emergency stop: -4% after 15 days
    - Trailing stop: Lock 50% of max profit once up >0.5%
    """
    capital = INITIAL_CAPITAL
    equity_curve = [INITIAL_CAPITAL]
    trades = []
    open_positions = {}
    last_trade_date = {}

    exit_reasons = {
        'target': 0,
        'trailing_stop': 0,
        'emergency_stop': 0,
        'final': 0
    }

    # Combine all pairs data sorted by date
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

        # Check for entry signal
        if pair not in open_positions:
            # Check cooldown
            if pair in last_trade_date:
                days_since_last = (current_date - last_trade_date[pair]).days
                if days_since_last < COOLDOWN_DAYS:
                    continue

            # Get prediction
            X = row[FEATURE_COLS].values.reshape(1, -1)
            prob = model.predict_proba(X)[0, 1]

            if prob > MIN_CONFIDENCE:
                position_in_range = row['position_in_range']

                # Entry: Near bottom (long) or near top (short)
                if position_in_range < EDGE_THRESHOLD_LOW:
                    direction = 'long'
                    entry_price = current_price
                    target_price = (row['high_20d'] + row['low_20d']) / 2  # Range middle

                elif position_in_range > EDGE_THRESHOLD_HIGH:
                    direction = 'short'
                    entry_price = current_price
                    target_price = (row['high_20d'] + row['low_20d']) / 2  # Range middle

                else:
                    continue  # Not at edge, skip

                # Calculate position size
                risk_amount = capital * RISK_PER_TRADE
                stop_distance = abs(entry_price * 0.02)  # Assume 2% risk per position
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

        # Check exits for open positions
        if pair in open_positions:
            pos = open_positions[pair]
            pos['days_held'] += 1

            # Calculate current profit/loss
            if pos['direction'] == 'long':
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            else:  # short
                pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']

            # Update max profit
            pos['max_profit'] = max(pos['max_profit'], pnl_pct)

            exit_triggered = False
            exit_reason = None
            exit_price = current_price

            # 1. Check target hit
            if pos['direction'] == 'long' and current_price >= pos['target_price']:
                exit_triggered = True
                exit_reason = 'target'
                exit_price = pos['target_price']
            elif pos['direction'] == 'short' and current_price <= pos['target_price']:
                exit_triggered = True
                exit_reason = 'target'
                exit_price = pos['target_price']

            # 2. Check trailing stop (once up >0.5%, lock 50% of max profit)
            if not exit_triggered and pos['max_profit'] > TRAILING_THRESHOLD:
                trailing_level = pos['max_profit'] * TRAILING_LOCK_PCT
                if pnl_pct < trailing_level:
                    exit_triggered = True
                    exit_reason = 'trailing_stop'

            # 3. Check emergency stop (-4% after 15 days)
            if not exit_triggered and pos['days_held'] >= EMERGENCY_STOP_DAYS:
                if pnl_pct < EMERGENCY_STOP_PCT:
                    exit_triggered = True
                    exit_reason = 'emergency_stop'

            # Execute exit
            if exit_triggered:
                # Recalculate final P&L
                if pos['direction'] == 'long':
                    final_pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                else:
                    final_pnl_pct = (pos['entry_price'] - exit_price) / pos['entry_price']

                profit_dollars = final_pnl_pct * pos['position_size'] * pos['entry_price']
                capital += profit_dollars

                trades.append({
                    'pair': pair,
                    'entry_date': pos['entry_date'],
                    'exit_date': current_date,
                    'direction': pos['direction'],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': final_pnl_pct,
                    'pnl_dollars': profit_dollars,
                    'days_held': pos['days_held'],
                    'exit_reason': exit_reason,
                    'year': pd.to_datetime(current_date).year
                })

                exit_reasons[exit_reason] += 1
                last_trade_date[pair] = current_date
                del open_positions[pair]

        equity_curve.append(capital)

    # Close any remaining positions at final price
    for pair, pos in open_positions.items():
        exit_price = combined_data[combined_data['pair'] == pair].iloc[-1]['close']
        exit_date = combined_data[combined_data['pair'] == pair].iloc[-1]['date']

        if pos['direction'] == 'long':
            final_pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
        else:
            final_pnl_pct = (pos['entry_price'] - exit_price) / pos['entry_price']

        profit_dollars = final_pnl_pct * pos['position_size'] * pos['entry_price']
        capital += profit_dollars

        trades.append({
            'pair': pair,
            'entry_date': pos['entry_date'],
            'exit_date': exit_date,
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'pnl_pct': final_pnl_pct,
            'pnl_dollars': profit_dollars,
            'days_held': pos['days_held'],
            'exit_reason': 'final',
            'year': pd.to_datetime(exit_date).year
        })

        exit_reasons['final'] += 1

    # Calculate metrics
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
        num_trades = len(trades_df)
        win_rate = (trades_df['pnl_pct'] > 0).sum() / num_trades
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if (trades_df['pnl_pct'] > 0).any() else 0
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if (trades_df['pnl_pct'] < 0).any() else 0

        # Calculate max drawdown
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_dd = drawdown.min()

        print(f"{year_label}:")
        print(f"  Total Return: {total_return:+.1%}")
        print(f"  Final Capital: ${capital:,.0f}")
        print(f"  Total Trades: {num_trades}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Avg Win: {avg_win:+.2%}")
        print(f"  Avg Loss: {avg_loss:+.2%}")
        print(f"  Max Drawdown: {max_dd:.1%}")
        print(f"  Exit Reasons: Target={exit_reasons['target']}, Trail={exit_reasons['trailing_stop']}, "
              f"Emergency={exit_reasons['emergency_stop']}, Final={exit_reasons['final']}")
        print()

        return {
            'label': year_label,
            'capital': capital,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_dd': max_dd,
            'trades': trades_df,
            'exit_reasons': exit_reasons
        }
    else:
        print(f"{year_label}: No trades")
        print()
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
model = train_model(all_pairs_data)

# Test on losing years
print("="*100)
print("BACKTEST RESULTS")
print("="*100)
print()

losing_result = backtest_mean_reversion(model, all_pairs_data, LOSING_YEARS, "Losing Years (2018, 2020, 2021)")

# Test on winning year
winning_result = backtest_mean_reversion(model, all_pairs_data, WINNING_YEARS, "Winning Year (2019)")

# Compare
print("="*100)
print("COMPARISON")
print("="*100)
print()

if losing_result and winning_result:
    edge = losing_result['total_return'] - winning_result['total_return']

    print(f"{'Metric':<20} {'Losing Years':>15} {'Winning Year':>15} {'Difference':>15}")
    print("-" * 70)
    print(f"{'Total Return':<20} {losing_result['total_return']:>14.1%} {winning_result['total_return']:>14.1%} {edge:>14.1%}")
    print(f"{'Trades':<20} {losing_result['num_trades']:>14.0f} {winning_result['num_trades']:>14.0f}")
    print(f"{'Win Rate':<20} {losing_result['win_rate']:>14.1%} {winning_result['win_rate']:>14.1%}")
    print(f"{'Max Drawdown':<20} {losing_result['max_dd']:>14.1%} {winning_result['max_dd']:>14.1%}")
    print()

    if edge > 0.10:
        print(f"✓ COMPLEMENT STRATEGY CONFIRMED!")
        print(f"  Mean reversion performs {edge:+.1%} better in losing years.")
        print(f"  This could offset breakout strategy losses in choppy markets.")
    elif edge > 0:
        print(f"MARGINAL EDGE: {edge:+.1%}")
        print(f"  Small advantage in losing years, may not justify complexity.")
    else:
        print(f"NO EDGE: Mean reversion performs {edge:+.1%} worse in losing years.")
        print(f"  Strategy does not complement breakout approach.")

    print()
    print("="*100)
    print("NEXT STEPS")
    print("="*100)
    print()

    if edge > 0.10:
        print("1. Validate on full 2016-2025 period")
        print("2. Optimize entry thresholds (test 0.2, 0.3, 0.4 for edges)")
        print("3. Test combining with breakout strategy")
        print("4. Consider dynamic position sizing based on regime")
else:
    print("Unable to compare - insufficient trades in one or both periods")
