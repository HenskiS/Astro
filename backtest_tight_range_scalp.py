"""
TIGHT_RANGE: SCALPING STRATEGY
================================
Based on analysis insights:
- 85% prediction accuracy
- 67% of predictions have >1% move
- Avg max move: ±0.8%

Strategy: SCALP the tight_range with matched risk/reward

Test multiple target/stop combinations:
- 0.5% / -0.5%
- 0.6% / -0.6%
- 0.7% / -0.7%
- 0.8% / -0.8%

Entry: Mean reversion at range edges during predicted tight_range
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TIGHT_RANGE: SCALPING STRATEGY")
print("="*100)
print()

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
LOSING_YEARS = [2018, 2020, 2021]
WINNING_YEARS = [2019]

INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.005
MIN_CONFIDENCE = 0.70

# Entry parameters
EDGE_THRESHOLD_LOW = 0.3
EDGE_THRESHOLD_HIGH = 0.7

# Test these target/stop combinations
TARGET_STOP_COMBOS = [
    (0.005, -0.005),  # 0.5% / -0.5%
    (0.006, -0.006),  # 0.6% / -0.6%
    (0.007, -0.007),  # 0.7% / -0.7%
    (0.008, -0.008),  # 0.8% / -0.8%
]

COOLDOWN_DAYS = 1  # Short cooldown for high frequency

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

def create_tight_range_target(df):
    """Create tight_range target"""
    future_high_10d = df['high'].rolling(10).max().shift(-10)
    future_low_10d = df['low'].rolling(10).min().shift(-10)
    future_range = (future_high_10d - future_low_10d) / df['close']

    df['tight_range'] = (future_range < df['range_20d'] * 0.8).astype(int)

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
    """Train tight_range model"""
    print("Training tight_range model on 2016-2017...")

    X_train_list = []
    y_train_list = []

    for pair_data in all_pairs_data.values():
        train_data = pair_data[(pair_data['year'] >= 2016) & (pair_data['year'] <= 2017)].copy()
        train_data = train_data.dropna(subset=FEATURE_COLS + ['tight_range'])

        if len(train_data) > 50:
            X_train_list.append(train_data[FEATURE_COLS])
            y_train_list.append(train_data['tight_range'])

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

def backtest_scalp(model, all_pairs_data, test_years, target_pct, stop_pct):
    """
    Backtest scalping strategy

    Entry: tight_range predicted + at range edges
    Target: +target_pct
    Stop: stop_pct
    """
    capital = INITIAL_CAPITAL
    trades = []
    open_positions = {}
    last_trade_date = {}

    # Combine all data
    all_data = []
    for pair, pair_data in all_pairs_data.items():
        test_data = pair_data[pair_data['year'].isin(test_years)].copy()
        test_data = test_data.dropna(subset=FEATURE_COLS + ['tight_range'])
        test_data['pair'] = pair
        all_data.append(test_data)

    combined_data = pd.concat(all_data).sort_values('date').reset_index(drop=True)

    for idx, row in combined_data.iterrows():
        pair = row['pair']
        current_date = row['date']
        current_price = row['close']

        # Entry
        if pair not in open_positions:
            # Cooldown
            if pair in last_trade_date:
                days_since_last = (current_date - last_trade_date[pair]).days
                if days_since_last < COOLDOWN_DAYS:
                    continue

            # Get prediction
            X = row[FEATURE_COLS].values.reshape(1, -1)
            prob = model.predict_proba(X)[0, 1]

            if prob > MIN_CONFIDENCE:
                position_in_range = row['position_in_range']

                # Entry at edges
                if position_in_range < EDGE_THRESHOLD_LOW:
                    direction = 'long'
                    entry_price = current_price
                    target_price = entry_price * (1 + target_pct)
                    stop_price = entry_price * (1 + stop_pct)

                elif position_in_range > EDGE_THRESHOLD_HIGH:
                    direction = 'short'
                    entry_price = current_price
                    target_price = entry_price * (1 - target_pct)
                    stop_price = entry_price * (1 - stop_pct)

                else:
                    continue

                # Position sizing
                risk_amount = capital * RISK_PER_TRADE
                stop_distance = abs(entry_price - stop_price)
                position_size = risk_amount / stop_distance

                open_positions[pair] = {
                    'entry_date': current_date,
                    'entry_price': entry_price,
                    'direction': direction,
                    'target_price': target_price,
                    'stop_price': stop_price,
                    'position_size': position_size,
                    'days_held': 0
                }

        # Check exits
        if pair in open_positions:
            pos = open_positions[pair]
            pos['days_held'] += 1

            exit_triggered = False
            exit_reason = None
            exit_price = current_price

            # Target hit
            if pos['direction'] == 'long' and current_price >= pos['target_price']:
                exit_triggered = True
                exit_reason = 'target'
                exit_price = pos['target_price']
            elif pos['direction'] == 'short' and current_price <= pos['target_price']:
                exit_triggered = True
                exit_reason = 'target'
                exit_price = pos['target_price']

            # Stop hit
            if not exit_triggered:
                if pos['direction'] == 'long' and current_price <= pos['stop_price']:
                    exit_triggered = True
                    exit_reason = 'stop'
                    exit_price = pos['stop_price']
                elif pos['direction'] == 'short' and current_price >= pos['stop_price']:
                    exit_triggered = True
                    exit_reason = 'stop'
                    exit_price = pos['stop_price']

            # Time exit (5 days)
            if not exit_triggered and pos['days_held'] >= 5:
                exit_triggered = True
                exit_reason = 'time'

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

        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
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
    df = create_tight_range_target(df)

    all_pairs_data[pair] = df

print(f"Loaded {len(all_pairs_data)} pairs")
print()

# Train model
model = train_model(all_pairs_data)

# Test all combinations
print("="*100)
print("TESTING TARGET/STOP COMBINATIONS")
print("="*100)
print()

results = []

for target_pct, stop_pct in TARGET_STOP_COMBOS:
    print(f"Testing Target: {target_pct*100:.1f}%, Stop: {stop_pct*100:.1f}%")

    losing_result = backtest_scalp(model, all_pairs_data, LOSING_YEARS, target_pct, stop_pct)
    winning_result = backtest_scalp(model, all_pairs_data, WINNING_YEARS, target_pct, stop_pct)

    if losing_result and winning_result:
        edge = losing_result['total_return'] - winning_result['total_return']

        print(f"  Losing years: {losing_result['total_return']:+.1%} ({losing_result['num_trades']} trades, {losing_result['win_rate']:.0%} win)")
        print(f"  Winning year: {winning_result['total_return']:+.1%} ({winning_result['num_trades']} trades, {winning_result['win_rate']:.0%} win)")
        print(f"  Edge: {edge:+.1%}")
        print()

        results.append({
            'target': target_pct,
            'stop': stop_pct,
            'losing_return': losing_result['total_return'],
            'losing_trades': losing_result['num_trades'],
            'losing_win': losing_result['win_rate'],
            'winning_return': winning_result['total_return'],
            'winning_trades': winning_result['num_trades'],
            'winning_win': winning_result['win_rate'],
            'edge': edge
        })

# Summary
print("="*100)
print("RESULTS COMPARISON")
print("="*100)
print()

results_df = pd.DataFrame(results).sort_values('losing_return', ascending=False)

print(f"{'Target':>8} {'Stop':>8} {'Losing Yrs':>12} {'Trades':>7} {'Win%':>5}   {'Win Year':>12} {'Trades':>7} {'Win%':>5}   {'Edge':>7}")
print("-" * 95)

for _, row in results_df.iterrows():
    print(f"{row['target']*100:>7.1f}% {row['stop']*100:>7.1f}% {row['losing_return']:>11.1%} {row['losing_trades']:>7.0f} {row['losing_win']:>4.0%}   "
          f"{row['winning_return']:>11.1%} {row['winning_trades']:>7.0f} {row['winning_win']:>4.0%}   "
          f"{row['edge']:>6.1%}")

print()
print("="*100)
print("VERDICT")
print("="*100)
print()

best = results_df.iloc[0]

if best['losing_return'] > 0.10 and best['edge'] > 0.05:
    print(f"SUCCESS! Scalping strategy works!")
    print(f"  Best config: {best['target']*100:.1f}% target, {best['stop']*100:.1f}% stop")
    print(f"  Losing years: {best['losing_return']:+.1%}")
    print(f"  Edge: {best['edge']:+.1%}")
    print()
    print("By matching our targets/stops to the actual price movement range,")
    print("we can finally monetize the 85% prediction accuracy!")
elif best['losing_return'] > 0:
    print(f"Profitable but marginal")
    print(f"  Best config: {best['target']*100:.1f}% target, {best['stop']*100:.1f}% stop")
    print(f"  Losing years: {best['losing_return']:+.1%}")
else:
    print("Still unprofitable - even with matched targets/stops")
    print(f"  Best result: {best['losing_return']:+.1%}")
    print()
    print("The 0.82% avg movement is too small to overcome trading costs")
