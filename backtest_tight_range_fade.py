"""
TIGHT_RANGE: FADE THE MOVE STRATEGY
====================================
New approach: Trade ALL tight_range signals, not just those at edges.

Strategy:
- When tight_range predicted >70%, volatility will contract
- Enter immediately: Long if below middle, Short if above middle
- Target: Reversion to range middle (mean reversion in low vol)
- Stop: -1.5% (tight since low vol)
- Time exit: 10 days

Goal: Capture the 85% accuracy by trading MORE signals, not fewer.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TIGHT_RANGE: FADE THE MOVE STRATEGY")
print("="*100)
print()

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
LOSING_YEARS = [2018, 2020, 2021]
WINNING_YEARS = [2019]

INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.005
MIN_CONFIDENCE = 0.70

# Exit parameters
STOP_LOSS_PCT = -0.015     # -1.5% stop (tight since predicting low vol)
TRAILING_THRESHOLD = 0.003  # Activate at 0.3%
TRAILING_PCT = 0.5         # Lock 50%
MAX_HOLD_DAYS = 10         # Exit after 10 days

COOLDOWN_DAYS = 1  # Short cooldown (1 day)

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

def backtest_fade_strategy(model, all_pairs_data, test_years, year_label):
    """
    Backtest FADE THE MOVE strategy

    Entry:
    - Tight_range predicted >70% (NO edge filter)
    - If price > middle: Short (expect reversion down)
    - If price < middle: Long (expect reversion up)

    Exit:
    - Target: Range middle
    - Stop: -1.5%
    - Trailing: 50% of gains once up 0.3%
    - Time: 10 days max
    """
    capital = INITIAL_CAPITAL
    trades = []
    open_positions = {}
    last_trade_date = {}

    exit_reasons = {
        'target': 0,
        'stop_loss': 0,
        'trailing_stop': 0,
        'time_exit': 0,
        'final': 0
    }

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

        # Entry: On ANY tight_range prediction
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
                # Calculate range middle
                range_middle = (row['high_20d'] + row['low_20d']) / 2

                # Fade the move: If above middle, short; if below middle, long
                if current_price > range_middle:
                    direction = 'short'
                    target_price = range_middle
                elif current_price < range_middle:
                    direction = 'long'
                    target_price = range_middle
                else:
                    # At exact middle, skip
                    continue

                # Position sizing
                risk_amount = capital * RISK_PER_TRADE
                stop_distance = abs(current_price * abs(STOP_LOSS_PCT))
                position_size = risk_amount / stop_distance

                open_positions[pair] = {
                    'entry_date': current_date,
                    'entry_price': current_price,
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

            # Calculate P&L
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

            # 2. Stop loss
            if not exit_triggered and pnl_pct < STOP_LOSS_PCT:
                exit_triggered = True
                exit_reason = 'stop_loss'

            # 3. Trailing stop
            if not exit_triggered and pos['max_profit'] > TRAILING_THRESHOLD:
                trailing_level = pos['max_profit'] * TRAILING_PCT
                if pnl_pct < trailing_level:
                    exit_triggered = True
                    exit_reason = 'trailing_stop'

            # 4. Time exit
            if not exit_triggered and pos['days_held'] >= MAX_HOLD_DAYS:
                exit_triggered = True
                exit_reason = 'time_exit'

            if exit_triggered:
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
                    'pnl_pct': final_pnl_pct,
                    'days_held': pos['days_held'],
                    'exit_reason': exit_reason
                })

                exit_reasons[exit_reason] += 1
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
        exit_reasons['final'] += 1

    # Calculate metrics
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
        num_trades = len(trades_df)
        win_rate = (trades_df['pnl_pct'] > 0).sum() / num_trades
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if (trades_df['pnl_pct'] > 0).any() else 0
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if (trades_df['pnl_pct'] < 0).any() else 0

        print(f"{year_label}:")
        print(f"  Total Return: {total_return:+.1%}")
        print(f"  Trades: {num_trades}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Avg Win: {avg_win:+.2%}")
        print(f"  Avg Loss: {avg_loss:+.2%}")
        print(f"  Risk/Reward: {avg_win/abs(avg_loss):.2f}:1" if avg_loss != 0 else "")
        print(f"  Exits: Target={exit_reasons['target']}, Stop={exit_reasons['stop_loss']}, "
              f"Trail={exit_reasons['trailing_stop']}, Time={exit_reasons['time_exit']}")
        print()

        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
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
    df = create_tight_range_target(df)

    all_pairs_data[pair] = df

print(f"Loaded {len(all_pairs_data)} pairs")
print()

# Train model
model = train_model(all_pairs_data)

# Test
print("="*100)
print("RESULTS")
print("="*100)
print()

losing_result = backtest_fade_strategy(model, all_pairs_data, LOSING_YEARS, "Losing Years (2018, 2020, 2021)")
winning_result = backtest_fade_strategy(model, all_pairs_data, WINNING_YEARS, "Winning Year (2019)")

if losing_result and winning_result:
    edge = losing_result['total_return'] - winning_result['total_return']

    print("="*100)
    print("COMPARISON")
    print("="*100)
    print()
    print(f"Losing years: {losing_result['total_return']:+.1%} ({losing_result['num_trades']} trades, {losing_result['win_rate']:.0%} win)")
    print(f"Winning year: {winning_result['total_return']:+.1%} ({winning_result['num_trades']} trades, {winning_result['win_rate']:.0%} win)")
    print(f"Edge: {edge:+.1%}")
    print()

    if edge > 0.10 and losing_result['total_return'] > 0.10:
        print("SUCCESS! Fade strategy works with tight_range predictions!")
    elif losing_result['total_return'] > 0:
        print("Profitable but marginal edge")
    else:
        print("Still unprofitable - the prediction accuracy doesn't translate to trading profits")
