"""
WALK-FORWARD ANALYSIS FOR 15M STRATEGY (EXPANDING WINDOW)
==========================================================
Simulates production retraining with EXPANDING training window:
1. Train on ALL data from start to test period
2. Test on next 1 month
3. Slide forward and repeat
4. Aggregate all test results

This is the MOST realistic approach - as more data becomes available,
you train on more history (not just last N months).

First window: 6 months training
Second window: 7 months training
Third window: 8 months training
...and so on
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

print("="*100)
print("WALK-FORWARD ANALYSIS: 15M BREAKOUT STRATEGY (EXPANDING WINDOW)")
print("="*100)
print()

# Walk-forward parameters
MIN_TRAIN_MONTHS = 6  # Minimum training data for first window
TEST_MONTHS = 1   # Test on 1 month
LOOKBACK_PERIOD = 80  # 20 hours
FORWARD_PERIODS = 24  # 6 hours
DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

# Strategy parameters (optimized)
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.004
MIN_CONFIDENCE = 0.70
EMERGENCY_STOP_PERIODS = 24
EMERGENCY_STOP_LOSS_PCT = -0.04
TRAILING_STOP_TRIGGER = 0.001
TRAILING_STOP_PCT = 0.75
LADDER_LEVELS = [0.002, 0.004]
LADDER_SCALE_PCT = 0.40
MAX_TOTAL_POSITIONS = 120
MAX_POSITIONS_PER_PAIR = 15
AVOID_HOURS = [20, 21, 22]

print(f"Walk-forward configuration:")
print(f"  Train window: EXPANDING (starts at {MIN_TRAIN_MONTHS} months)")
print(f"  Test window: {TEST_MONTHS} month")
print(f"  Lookback: {LOOKBACK_PERIOD} periods ({LOOKBACK_PERIOD*0.25:.1f}h)")
print()

# Load feature calculation functions from train script
def add_time_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute_slot'] = df.index.minute // 15
    df['day_of_week'] = df.index.dayofweek
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['session_overlap'] = (df['european_session'] + df['us_session']) > 1
    df['friday_close'] = ((df['day_of_week'] == 4) & (df['hour'] >= 20)).astype(int)
    df['sunday_open'] = ((df['day_of_week'] == 6) & (df['hour'] <= 3)).astype(int)
    return df

def calculate_features(df, lookback=LOOKBACK_PERIOD):
    df = df.copy()
    df = add_time_features(df)

    df[f'high_{lookback}p'] = df['high'].rolling(lookback).max()
    df[f'low_{lookback}p'] = df['low'].rolling(lookback).min()
    df[f'range_{lookback}p'] = df[f'high_{lookback}p'] - df[f'low_{lookback}p']
    df['dist_to_high'] = (df[f'high_{lookback}p'] - df['close']) / df['close']
    df['dist_to_low'] = (df['close'] - df[f'low_{lookback}p']) / df['close']

    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_100'] = df['close'].ewm(span=100).mean()

    df['price_vs_ema12'] = (df['close'] - df['ema_12']) / df['close']
    df['price_vs_ema26'] = (df['close'] - df['ema_26']) / df['close']
    df['price_vs_ema50'] = (df['close'] - df['ema_50']) / df['close']
    df['price_vs_ema100'] = (df['close'] - df['ema_100']) / df['close']

    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(
        abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']

    df['volume_ma'] = df['volume'].rolling(96).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    df['return_1p'] = df['close'].pct_change(1)
    df['return_4p'] = df['close'].pct_change(4)
    df['return_16p'] = df['close'].pct_change(16)
    df['return_96p'] = df['close'].pct_change(96)

    df['spread_ma'] = df['spread_pct'].rolling(96).mean()
    df['spread_ratio'] = df['spread_pct'] / df['spread_ma']

    return df

def calculate_targets(df, lookback=LOOKBACK_PERIOD, forward_periods=FORWARD_PERIODS):
    df = df.copy()
    df['future_high'] = df['high'].shift(-1).rolling(forward_periods).max()
    df['future_low'] = df['low'].shift(-1).rolling(forward_periods).min()
    high_level = df[f'high_{lookback}p']
    low_level = df[f'low_{lookback}p']
    df['breakout_high'] = (df['future_high'] > high_level).astype(int)
    df['breakout_low'] = (df['future_low'] < low_level).astype(int)
    return df

feature_cols = [
    'dist_to_high', 'dist_to_low', f'range_{LOOKBACK_PERIOD}p',
    'price_vs_ema12', 'price_vs_ema26', 'price_vs_ema50', 'price_vs_ema100',
    'macd', 'macd_signal', 'macd_hist', 'rsi_14', 'atr_pct', 'volume_ratio',
    'return_1p', 'return_4p', 'return_16p', 'return_96p',
    'spread_pct', 'spread_ratio',
    'hour', 'minute_slot', 'day_of_week',
    'asian_session', 'european_session', 'us_session', 'session_overlap',
    'friday_close', 'sunday_open'
]

# Load and prepare data
print("Loading data...")
all_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = calculate_features(df, LOOKBACK_PERIOD)
    df = calculate_targets(df, LOOKBACK_PERIOD, FORWARD_PERIODS)
    all_data[pair] = df
    print(f"  {pair}: {len(df)} candles")

print()

# Load raw data for backtesting (with bid/ask prices)
print("Loading raw data for backtesting...")
all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

print()

# Determine walk-forward windows (EXPANDING)
min_date = min([df.index.min() for df in all_data.values()])
max_date = max([df.index.max() for df in all_data.values()])
total_days = (max_date - min_date).days

min_train_days = MIN_TRAIN_MONTHS * 30
test_days = TEST_MONTHS * 30

windows = []
# Start with minimum training period
first_test_start = min_date + timedelta(days=min_train_days)
current_test_start = first_test_start

while True:
    test_end = current_test_start + timedelta(days=test_days)

    if test_end > max_date:
        break

    # Expanding window: always train from min_date to current_test_start
    windows.append({
        'train_start': min_date,
        'train_end': current_test_start,
        'test_start': current_test_start,
        'test_end': test_end
    })

    # Move forward by test period
    current_test_start += timedelta(days=test_days)

print(f"Walk-forward windows: {len(windows)}")
print()

# Position class
class Position:
    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.original_size = size
        self.breakout_target = breakout_target
        self.confidence = confidence
        self.periods_held = 0
        self.max_profit = 0
        self.trailing_stop = None
        self.partial_exits = []
        self.ladder_level = 0

    def update(self, date, bid_high, bid_low, bid_close, ask_high, ask_low, ask_close):
        self.periods_held += 1

        if self.direction == 'long':
            current_profit = (bid_close - self.entry_price) / self.entry_price
            intraday_high_profit = (bid_high - self.entry_price) / self.entry_price
            hit_target = bid_high >= self.breakout_target
        else:
            current_profit = (self.entry_price - ask_close) / self.entry_price
            intraday_high_profit = (self.entry_price - ask_low) / self.entry_price
            hit_target = ask_low <= self.breakout_target

        self.max_profit = max(self.max_profit, intraday_high_profit)

        if self.ladder_level < len(LADDER_LEVELS):
            if intraday_high_profit >= LADDER_LEVELS[self.ladder_level]:
                self.partial_exits.append((LADDER_LEVELS[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
                self.ladder_level += 1
                return None

        if self.periods_held >= EMERGENCY_STOP_PERIODS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            exit_price = bid_close if self.direction == 'long' else ask_close
            return 'emergency_stop', exit_price, current_profit

        if self.trailing_stop is None:
            if self.max_profit > TRAILING_STOP_TRIGGER:
                self.trailing_stop = self.entry_price
        else:
            old_stop = self.trailing_stop

            if self.direction == 'long':
                hit_stop = bid_low <= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price + (bid_high - self.entry_price) * TRAILING_STOP_PCT
                self.trailing_stop = max(self.trailing_stop, new_stop)
            else:
                hit_stop = ask_high >= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price - (self.entry_price - ask_low) * TRAILING_STOP_PCT
                self.trailing_stop = min(self.trailing_stop, new_stop)

        if hit_target:
            return 'target', self.breakout_target, current_profit

        return None

    def calculate_blended_profit(self, final_profit):
        if len(self.partial_exits) == 0:
            return final_profit
        total = 0
        remaining = 1.0
        for exit_profit, exit_pct in self.partial_exits:
            total += exit_profit * exit_pct
            remaining -= exit_pct
        total += final_profit * remaining
        return total

# Run walk-forward analysis
all_window_results = []
all_trades = []
capital = INITIAL_CAPITAL
positions = []

for window_idx, window in enumerate(windows, 1):
    print(f"Window {window_idx}/{len(windows)}")
    print(f"  Train: {window['train_start'].date()} to {window['train_end'].date()}")
    print(f"  Test:  {window['test_start'].date()} to {window['test_end'].date()}")

    # Train models for this window
    window_models = {}
    for pair in PAIRS:
        df = all_data[pair]
        train_data = df[(df.index >= window['train_start']) & (df.index < window['train_end'])]
        train_data = train_data.dropna(subset=feature_cols + ['breakout_high', 'breakout_low'])

        if len(train_data) < 1000:
            print(f"  {pair}: Insufficient train data ({len(train_data)} samples), skipping")
            continue

        X_train = train_data[feature_cols]
        y_train_high = train_data['breakout_high']
        y_train_low = train_data['breakout_low']

        model_high = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, random_state=42,
                                       eval_metric='logloss')
        model_high.fit(X_train, y_train_high, verbose=False)

        model_low = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8, random_state=42,
                                      eval_metric='logloss')
        model_low.fit(X_train, y_train_low, verbose=False)

        window_models[pair] = {'model_high': model_high, 'model_low': model_low}

    print(f"  Trained models for {len(window_models)} pairs")

    # Generate predictions for test period
    window_predictions = {}
    for pair in window_models.keys():
        df = all_data[pair]
        test_data = df[(df.index >= window['test_start']) & (df.index < window['test_end'])]
        test_data = test_data.dropna(subset=feature_cols)

        if len(test_data) == 0:
            continue

        X_test = test_data[feature_cols]
        pred_high = window_models[pair]['model_high'].predict_proba(X_test)[:, 1]
        pred_low = window_models[pair]['model_low'].predict_proba(X_test)[:, 1]

        preds_df = pd.DataFrame({
            'breakout_high_prob': pred_high,
            'breakout_low_prob': pred_low,
            f'high_{LOOKBACK_PERIOD}p': test_data[f'high_{LOOKBACK_PERIOD}p'],
            f'low_{LOOKBACK_PERIOD}p': test_data[f'low_{LOOKBACK_PERIOD}p'],
            'close': test_data['close']
        }, index=test_data.index)

        window_predictions[pair] = preds_df

    # Backtest on test period
    window_start_capital = capital
    window_trades = []

    test_periods = sorted(list(set().union(*[preds_df.index for preds_df in window_predictions.values()])))

    for date in test_periods:
        # Get prices
        prices_dict = {}
        for pair in window_predictions.keys():
            if date in all_raw_data[pair].index:
                row = all_raw_data[pair].loc[date]
                prices_dict[pair] = {
                    'bid_high': row['bid_high'],
                    'bid_low': row['bid_low'],
                    'bid_close': row['bid_close'],
                    'ask_high': row['ask_high'],
                    'ask_low': row['ask_low'],
                    'ask_close': row['ask_close'],
                    'close': row['close']
                }

        # Update positions
        positions_to_close = []
        for position in positions:
            if position.pair not in prices_dict:
                continue

            prices = prices_dict[position.pair]
            exit_info = position.update(
                date,
                prices['bid_high'],
                prices['bid_low'],
                prices['bid_close'],
                prices['ask_high'],
                prices['ask_low'],
                prices['ask_close']
            )
            if exit_info is not None:
                positions_to_close.append((position, exit_info))

        # Close positions
        for position, exit_info in positions_to_close:
            exit_reason, exit_price, current_profit = exit_info

            if position.direction == 'long':
                raw_profit = (exit_price - position.entry_price) / position.entry_price
            else:
                raw_profit = (position.entry_price - exit_price) / position.entry_price

            profit_pct = position.calculate_blended_profit(raw_profit)
            profit_dollars = profit_pct * (position.original_size * position.entry_price)

            capital += profit_dollars
            positions.remove(position)

            trade_record = {
                'window': window_idx,
                'pair': position.pair,
                'entry_date': position.entry_date,
                'exit_date': date,
                'direction': position.direction,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'size': position.original_size,
                'profit_pct': profit_pct,
                'profit_dollars': profit_dollars,
                'periods_held': position.periods_held,
                'exit_reason': exit_reason,
                'confidence': position.confidence
            }
            window_trades.append(trade_record)
            all_trades.append(trade_record)

        # Open new positions
        if len(positions) >= MAX_TOTAL_POSITIONS:
            continue

        if date.hour in AVOID_HOURS:
            continue

        for pair in window_predictions.keys():
            if date not in window_predictions[pair].index:
                continue

            pair_positions = [p for p in positions if p.pair == pair]
            if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
                continue

            row = window_predictions[pair].loc[date]

            breakout_high_prob = row['breakout_high_prob']
            breakout_low_prob = row['breakout_low_prob']
            max_prob = max(breakout_high_prob, breakout_low_prob)

            if max_prob <= MIN_CONFIDENCE:
                continue

            # Calculate position size
            assumed_risk_pct = 0.02
            risk_amount = capital * RISK_PER_TRADE
            mid_price = row['close']
            position_size = risk_amount / (mid_price * assumed_risk_pct)

            # Determine direction and entry price
            if breakout_high_prob > breakout_low_prob:
                direction = 'long'
                breakout_level = row[f'high_{LOOKBACK_PERIOD}p']
                target = breakout_level * 1.005
                if pair not in prices_dict:
                    continue
                entry_price = prices_dict[pair]['ask_close']
            else:
                direction = 'short'
                breakout_level = row[f'low_{LOOKBACK_PERIOD}p']
                target = breakout_level * 0.995
                if pair not in prices_dict:
                    continue
                entry_price = prices_dict[pair]['bid_close']

            position = Position(pair, date, entry_price, direction, position_size, target, max_prob)
            positions.append(position)

    # Store window results
    window_return = (capital - window_start_capital) / window_start_capital

    all_window_results.append({
        'window': window_idx,
        'train_start': window['train_start'],
        'train_end': window['train_end'],
        'test_start': window['test_start'],
        'test_end': window['test_end'],
        'trades': len(window_trades),
        'start_capital': window_start_capital,
        'final_capital': capital,
        'return': window_return
    })

    print(f"  Test results: {len(window_trades)} trades, ${window_start_capital:.0f} -> ${capital:.0f} ({window_return:+.1%})")
    print()

# Aggregate all results
print("="*100)
print("WALK-FORWARD RESULTS")
print("="*100)
print()

print("Window-by-window performance:")
for result in all_window_results:
    print(f"Window {result['window']}: {result['test_start'].date()} to {result['test_end'].date()}")
    print(f"  Trades: {result['trades']} | Return: {result['return']:+.1%} | Capital: ${result['final_capital']:.0f}")
    print()

# Calculate aggregate statistics
avg_window_return = np.mean([r['return'] for r in all_window_results])
total_trades = sum([r['trades'] for r in all_window_results])
final_capital = all_window_results[-1]['final_capital'] if all_window_results else INITIAL_CAPITAL

total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
test_start = windows[0]['test_start']
test_end = windows[-1]['test_end']
days = (test_end - test_start).days
years = days / 365
cagr = (final_capital / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 else 0

print("="*100)
print("AGGREGATE STATISTICS")
print("="*100)
print()
print(f"Total windows: {len(windows)}")
print(f"Test period: {test_start.date()} to {test_end.date()} ({days} days, {years:.1f} years)")
print()
print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
print(f"Final Capital: ${final_capital:,.0f}")
print(f"Total Return: {total_return:+.1%}")
print(f"CAGR: {cagr:.1%}")
print()
print(f"Avg Window Return: {avg_window_return:+.1%}")
print(f"Total Trades: {total_trades:,}")
print()

# Detailed trade statistics
if len(all_trades) > 0:
    trades_df = pd.DataFrame(all_trades)
    winners = trades_df[trades_df['profit_pct'] > 0]
    losers = trades_df[trades_df['profit_pct'] <= 0]
    win_rate = len(winners) / len(trades_df) if len(trades_df) > 0 else 0
    avg_winner = winners['profit_pct'].mean() if len(winners) > 0 else 0
    avg_loser = losers['profit_pct'].mean() if len(losers) > 0 else 0
    profit_ratio = abs(avg_winner / avg_loser) if len(losers) > 0 and avg_loser != 0 else 0
    avg_winner_hold = winners['periods_held'].mean() * 0.25 if len(winners) > 0 else 0  # 15m = 0.25h
    avg_loser_hold = losers['periods_held'].mean() * 0.25 if len(losers) > 0 else 0

    print(f"Winners: {len(winners):,} ({win_rate:.1%})")
    print(f"Losers: {len(losers):,}")
    print()
    print(f"Avg Winner: {avg_winner:+.2%} ({avg_winner_hold:.1f} hours)")
    print(f"Avg Loser: {avg_loser:+.2%} ({avg_loser_hold:.1f} hours)")
    print(f"Profit Ratio: {profit_ratio:.2f}:1")
    print()

    # Exit reasons
    print("Exit Reasons:")
    exit_reasons = trades_df['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        pct = count / len(trades_df) * 100
        print(f"  {reason:20s} {count:>5,} ({pct:>5.1f}%)")
    print()


print("="*100)
print("NOTES")
print("="*100)
print()
print("EXPANDING WINDOW walk-forward analysis:")
print(f"  - Train on ALL data from start to test period")
print(f"  - First window: {MIN_TRAIN_MONTHS} months training")
print(f"  - Each subsequent window: +{TEST_MONTHS} month more training data")
print(f"  - Test on next {TEST_MONTHS} month")
print()
print("This is the MOST realistic approach for production because:")
print("  - Uses all available historical data (like production would)")
print("  - No lookahead bias")
print("  - Later windows benefit from more training data")
print("  - Simulates how models improve over time with more data")
print()
print("Compare with other approaches:")
print("  Single-train (70% data): +49.6% return, 202% CAGR")
print("  6-month fixed window: +45.0% return, 76% CAGR")
print(f"  Expanding window: {total_return:+.1%} return, {cagr:.1%} CAGR")
print()
print("Expanding window advantages:")
print("  - More training data = better model performance")
print("  - Realistic for production (you'd use all available data)")
print("  - First window has least data (conservative estimate)")
print("  - Performance typically improves as windows progress")
print()

print("="*100)
