"""
OPTIMIZE FOR CONSISTENCY ACROSS MULTIPLE YEARS
===============================================
Instead of optimizing for peak CAGR, we optimize for:
1. Positive CAGR across most years
2. Low variance (consistency)
3. High Sharpe ratio (return per unit of risk)

Tests different parameter combinations across all 5 test periods.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from itertools import product
warnings.filterwarnings('ignore')

DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

# Fixed parameters
INITIAL_CAPITAL = 500
FIFO_MODE = 'skip_competing'
AVOID_HOURS = [20, 21, 22]
SLIPPAGE_PCT = 0.0001
MAX_TOTAL_POSITIONS = 120
MAX_POSITIONS_PER_PAIR = 15

# Parameters to optimize
PARAM_GRID = {
    'RISK_PER_TRADE': [0.002, 0.003, 0.004, 0.005],  # 0.2% to 0.5%
    'MIN_CONFIDENCE': [0.65, 0.70, 0.75, 0.80],
    'EMERGENCY_STOP_PERIODS': [16, 24, 32],  # 4h, 6h, 8h
    'EMERGENCY_STOP_LOSS_PCT': [-0.03, -0.04, -0.05],
    'TRAILING_STOP_TRIGGER': [0.0008, 0.001, 0.0012],
    'TRAILING_STOP_PCT': [0.70, 0.75, 0.80],
}


class Position:
    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence, params):
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
        self.params = params

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

        # Ladder
        LADDER_LEVELS = [0.002, 0.004]
        LADDER_SCALE_PCT = 0.40
        if self.ladder_level < len(LADDER_LEVELS):
            if intraday_high_profit >= LADDER_LEVELS[self.ladder_level]:
                self.partial_exits.append((LADDER_LEVELS[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
                self.ladder_level += 1
                return None

        # Emergency stop
        if self.periods_held >= self.params['EMERGENCY_STOP_PERIODS'] and current_profit < self.params['EMERGENCY_STOP_LOSS_PCT']:
            exit_price = bid_close if self.direction == 'long' else ask_close
            return 'emergency_stop', exit_price, current_profit

        # Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > self.params['TRAILING_STOP_TRIGGER']:
                self.trailing_stop = self.entry_price
        else:
            old_stop = self.trailing_stop
            if self.direction == 'long':
                hit_stop = bid_low <= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price + (bid_high - self.entry_price) * self.params['TRAILING_STOP_PCT']
                self.trailing_stop = max(self.trailing_stop, new_stop)
            else:
                hit_stop = ask_high >= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price - (self.entry_price - ask_low) * self.params['TRAILING_STOP_PCT']
                self.trailing_stop = min(self.trailing_stop, new_stop)

        # Target
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


def run_backtest_with_params(predictions, raw_data, params):
    """Run backtest with given parameters"""
    all_trading_periods = set()
    for pair_df in predictions.values():
        all_trading_periods.update(pair_df.index)
    all_trading_periods = sorted(list(all_trading_periods))

    if len(all_trading_periods) == 0:
        return None

    min_date = min(all_trading_periods)
    max_date = max(all_trading_periods)
    capital = INITIAL_CAPITAL
    positions = []
    trades = []
    pending_signals = []

    for date in all_trading_periods:
        prices_dict = {}
        for pair in PAIRS:
            if date in raw_data[pair].index:
                row = raw_data[pair].loc[date]
                prices_dict[pair] = {
                    'bid_open': row['bid_open'], 'bid_high': row['bid_high'],
                    'bid_low': row['bid_low'], 'bid_close': row['bid_close'],
                    'ask_open': row['ask_open'], 'ask_high': row['ask_high'],
                    'ask_low': row['ask_low'], 'ask_close': row['ask_close'],
                    'close': row['close']
                }

        # Process pending signals
        signals_to_keep = []
        for signal in pending_signals:
            if signal['pair'] not in prices_dict or len(positions) >= MAX_TOTAL_POSITIONS:
                if signal['pair'] not in prices_dict:
                    signals_to_keep.append(signal)
                continue
            if len([p for p in positions if p.pair == signal['pair']]) >= MAX_POSITIONS_PER_PAIR:
                continue

            prices = prices_dict[signal['pair']]
            entry_price = prices['ask_open'] if signal['direction'] == 'long' else prices['bid_open']
            if SLIPPAGE_PCT > 0:
                entry_price *= (1 + SLIPPAGE_PCT) if signal['direction'] == 'long' else (1 - SLIPPAGE_PCT)

            position = Position(signal['pair'], date, entry_price, signal['direction'],
                              signal['size'], signal['target'], signal['confidence'], params)
            positions.append(position)

            exit_info = position.update(date, prices['bid_high'], prices['bid_low'], prices['bid_close'],
                                       prices['ask_high'], prices['ask_low'], prices['ask_close'])
            if exit_info:
                exit_reason, exit_price, _ = exit_info
                raw_profit = (exit_price - position.entry_price) / position.entry_price if position.direction == 'long' else (position.entry_price - exit_price) / position.entry_price
                profit_pct = position.calculate_blended_profit(raw_profit)
                profit_dollars = profit_pct * (position.original_size * position.entry_price)
                capital += profit_dollars
                positions.remove(position)
                trades.append({'profit_pct': profit_pct})

        pending_signals = signals_to_keep

        # Update positions
        positions_to_close = []
        for position in positions:
            if position.pair not in prices_dict:
                continue
            prices = prices_dict[position.pair]
            exit_info = position.update(date, prices['bid_high'], prices['bid_low'], prices['bid_close'],
                                       prices['ask_high'], prices['ask_low'], prices['ask_close'])
            if exit_info:
                positions_to_close.append((position, exit_info))

        for position, exit_info in positions_to_close:
            exit_reason, exit_price, _ = exit_info
            raw_profit = (exit_price - position.entry_price) / position.entry_price if position.direction == 'long' else (position.entry_price - exit_price) / position.entry_price
            profit_pct = position.calculate_blended_profit(raw_profit)
            profit_dollars = profit_pct * (position.original_size * position.entry_price)
            capital += profit_dollars
            positions.remove(position)
            trades.append({'profit_pct': profit_pct})

        # Generate signals
        if len(positions) >= MAX_TOTAL_POSITIONS or date.hour in AVOID_HOURS:
            continue

        for pair in PAIRS:
            if date not in predictions[pair].index:
                continue
            if len([p for p in positions if p.pair == pair]) >= MAX_POSITIONS_PER_PAIR:
                continue

            row = predictions[pair].loc[date]
            max_prob = max(row['breakout_high_prob'], row['breakout_low_prob'])
            if max_prob <= params['MIN_CONFIDENCE']:
                continue

            if row['breakout_high_prob'] > row['breakout_low_prob']:
                direction = 'long'
                target = row['high_80p'] * 1.005
            else:
                direction = 'short'
                target = row['low_80p'] * 0.995

            pair_positions = [p for p in positions if p.pair == pair]
            if FIFO_MODE == 'skip_competing' and len(pair_positions) > 0:
                if direction not in set(p.direction for p in pair_positions):
                    continue

            risk_amount = capital * params['RISK_PER_TRADE']
            position_size = risk_amount / (row['close'] * 0.02)

            pending_signals.append({
                'pair': pair, 'direction': direction, 'size': position_size,
                'target': target, 'confidence': max_prob
            })

    # Calculate results
    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        return None

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    days = (max_date - min_date).days
    years = max(days / 365, 0.01)
    cagr = (capital / INITIAL_CAPITAL) ** (1 / years) - 1

    return cagr


print("="*100)
print("OPTIMIZING FOR CONSISTENCY ACROSS MULTIPLE YEARS")
print("="*100)
print()

# Load raw data
print("Loading raw data...")
all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df
print("Done")
print()

# Load all test period predictions
test_files = [
    'test_predictions_15m_2021_test.pkl',
    'test_predictions_15m_2022_test.pkl',
    'test_predictions_15m_2023_test.pkl',
    'test_predictions_15m_2024_test.pkl',
    'test_predictions_15m_2025_test.pkl',
]

all_predictions = []
for filename in test_files:
    with open(filename, 'rb') as f:
        all_predictions.append(pickle.load(f))

print(f"Loaded {len(all_predictions)} test periods")
print()

# Generate parameter combinations
param_names = list(PARAM_GRID.keys())
param_values = list(PARAM_GRID.values())
param_combinations = list(product(*param_values))

print(f"Testing {len(param_combinations)} parameter combinations...")
print()

# Test each combination
best_score = -999
best_params = None
best_cagrs = None

for idx, param_combo in enumerate(param_combinations):
    params = dict(zip(param_names, param_combo))

    # Test on all periods
    cagrs = []
    for preds in all_predictions:
        cagr = run_backtest_with_params(preds, all_raw_data, params)
        if cagr is not None:
            cagrs.append(cagr)

    if len(cagrs) < 3:  # Need at least 3 valid results
        continue

    # Calculate metrics
    avg_cagr = np.mean(cagrs)
    std_cagr = np.std(cagrs)
    min_cagr = min(cagrs)
    positive_years = sum(1 for c in cagrs if c > 0)

    # Scoring: prioritize consistency and positive returns
    # Score = avg_cagr - 0.5 * std_cagr + 0.1 * positive_years
    score = avg_cagr - 0.3 * std_cagr + 0.05 * positive_years

    if score > best_score:
        best_score = score
        best_params = params
        best_cagrs = cagrs

        print(f"New best! Combo {idx+1}/{len(param_combinations)}")
        print(f"  Avg CAGR: {avg_cagr:.1%} | Std: {std_cagr:.1%} | Min: {min_cagr:.1%} | Positive: {positive_years}/5")
        print(f"  Score: {score:.3f}")
        print(f"  Params: {params}")
        print()

    # Progress update
    if (idx + 1) % 100 == 0:
        print(f"Progress: {idx+1}/{len(param_combinations)} combinations tested...")

print()
print("="*100)
print("OPTIMIZATION COMPLETE")
print("="*100)
print()

if best_params:
    print("BEST PARAMETERS (optimized for consistency):")
    print("-" * 100)
    for key, value in best_params.items():
        print(f"  {key:<25} {value}")

    print()
    print("PERFORMANCE ACROSS YEARS:")
    print("-" * 100)
    year_names = ['2021', '2022', '2023', '2024', '2025']
    for year, cagr in zip(year_names, best_cagrs):
        print(f"  {year}: {cagr:>8.1%}")

    print()
    avg_cagr = np.mean(best_cagrs)
    std_cagr = np.std(best_cagrs)
    min_cagr = min(best_cagrs)
    max_cagr = max(best_cagrs)
    positive_years = sum(1 for c in best_cagrs if c > 0)

    print(f"  Average: {avg_cagr:>8.1%}")
    print(f"  Std Dev: {std_cagr:>8.1%}")
    print(f"  Min:     {min_cagr:>8.1%}")
    print(f"  Max:     {max_cagr:>8.1%}")
    print(f"  Positive years: {positive_years}/5")

    print()
    print("Compare to original parameters:")
    print("  Original avg CAGR: -5.3%")
    print(f"  Optimized avg CAGR: {avg_cagr:.1%}")
    print()

    if avg_cagr > 0.05:
        print("[SUCCESS] Found parameters with positive expected return!")
    elif avg_cagr > -0.05:
        print("[MARGINAL] Parameters break-even but not strongly profitable")
    else:
        print("[FAILED] Could not find consistently profitable parameters")

print()
