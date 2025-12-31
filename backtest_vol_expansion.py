"""
VOL_EXPANSION: MOMENTUM STRATEGY
=================================
Test trading volatility EXPANSION predictions.

From comprehensive test:
- Simplified backtest: +111.5% in losing years, +35.9% in winning year (+75.7% edge)
- 1,933 trades in losing years, 57% win rate
- Predicts: Volatility will INCREASE by >20% in next 10 days

Strategy:
- When vol_expansion predicted, volatility is about to spike
- Trade WITH the momentum (not against it)
- Enter breakouts in the direction of the move
- Use trailing stops to ride the volatility wave
- This is trend-following during volatile periods
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("VOL_EXPANSION: MOMENTUM STRATEGY")
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
STOP_LOSS_PCT = -0.02      # -2% stop
TRAILING_THRESHOLD = 0.005  # Activate at 0.5%
TRAILING_PCT = 0.6          # Lock 60% of gains
MAX_HOLD_DAYS = 10

COOLDOWN_DAYS = 2

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

def create_vol_expansion_target(df):
    """Create vol_expansion target"""
    current_vol = df['volatility_10d']
    future_vol = df['volatility_10d'].shift(-10)

    df['vol_expansion'] = (future_vol > current_vol * 1.2).astype(int)

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
    """Train vol_expansion model"""
    print("Training vol_expansion model on 2016-2017...")

    X_train_list = []
    y_train_list = []

    for pair_data in all_pairs_data.values():
        train_data = pair_data[(pair_data['year'] >= 2016) & (pair_data['year'] <= 2017)].copy()
        train_data = train_data.dropna(subset=FEATURE_COLS + ['vol_expansion'])

        if len(train_data) > 50:
            X_train_list.append(train_data[FEATURE_COLS])
            y_train_list.append(train_data['vol_expansion'])

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

def backtest_momentum_strategy(model, all_pairs_data, test_years, year_label):
    """
    Backtest momentum strategy during vol_expansion

    Entry:
    - Vol_expansion predicted >70%
    - Enter in direction of current momentum (return_5d)
    - If return_5d > 0: Long (ride the up move)
    - If return_5d < 0: Short (ride the down move)

    Exit:
    - Stop: -2%
    - Trailing: Lock 60% once up >0.5%
    - Time: 10 days max
    """
    capital = INITIAL_CAPITAL
    trades = []
    open_positions = {}
    last_trade_date = {}

    exit_reasons = {
        'stop_loss': 0,
        'trailing_stop': 0,
        'time_exit': 0,
        'final': 0
    }

    # Combine all data
    all_data = []
    for pair, pair_data in all_pairs_data.items():
        test_data = pair_data[pair_data['year'].isin(test_years)].copy()
        test_data = test_data.dropna(subset=FEATURE_COLS + ['vol_expansion'])
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
                # Trade WITH the momentum
                momentum = row['return_5d']

                if momentum > 0.001:  # Positive momentum
                    direction = 'long'
                elif momentum < -0.001:  # Negative momentum
                    direction = 'short'
                else:
                    continue  # No clear momentum

                # Position sizing
                risk_amount = capital * RISK_PER_TRADE
                stop_distance = abs(current_price * abs(STOP_LOSS_PCT))
                position_size = risk_amount / stop_distance

                open_positions[pair] = {
                    'entry_date': current_date,
                    'entry_price': current_price,
                    'direction': direction,
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

            # 1. Stop loss
            if pnl_pct < STOP_LOSS_PCT:
                exit_triggered = True
                exit_reason = 'stop_loss'

            # 2. Trailing stop
            if not exit_triggered and pos['max_profit'] > TRAILING_THRESHOLD:
                trailing_level = pos['max_profit'] * TRAILING_PCT
                if pnl_pct < trailing_level:
                    exit_triggered = True
                    exit_reason = 'trailing_stop'

            # 3. Time exit
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
                    'pnl_pct': final_pnl_pct,
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
        if avg_loss != 0:
            print(f"  Risk/Reward: {avg_win/abs(avg_loss):.2f}:1")
        print(f"  Exits: Stop={exit_reasons['stop_loss']}, Trail={exit_reasons['trailing_stop']}, Time={exit_reasons['time_exit']}")
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
    df = create_vol_expansion_target(df)

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

losing_result = backtest_momentum_strategy(model, all_pairs_data, LOSING_YEARS, "Losing Years (2018, 2020, 2021)")
winning_result = backtest_momentum_strategy(model, all_pairs_data, WINNING_YEARS, "Winning Year (2019)")

if losing_result and winning_result:
    edge = losing_result['total_return'] - winning_result['total_return']

    print("="*100)
    print("VERDICT")
    print("="*100)
    print()
    print(f"Losing years: {losing_result['total_return']:+.1%} ({losing_result['num_trades']} trades, {losing_result['win_rate']:.0%} win)")
    print(f"Winning year: {winning_result['total_return']:+.1%} ({winning_result['num_trades']} trades, {winning_result['win_rate']:.0%} win)")
    print(f"Edge: {edge:+.1%}")
    print()

    if losing_result['total_return'] > 0.10 and edge > 0.05:
        print("SUCCESS! Vol_expansion momentum strategy works!")
        print("Trading WITH volatility expansion is profitable!")
    elif losing_result['total_return'] > 0:
        print("Profitable but marginal edge")
    else:
        print("Unprofitable - even momentum during vol expansion doesn't work")
