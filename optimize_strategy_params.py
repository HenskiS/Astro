"""
OPTIMIZE STRATEGY PARAMETERS
=============================
Systematically test combinations of:
1. Confidence threshold
2. Ladder levels and scale
3. Active management rules (time-based exits)
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from itertools import product
warnings.filterwarnings('ignore')

print("="*100)
print("STRATEGY PARAMETER OPTIMIZATION")
print("="*100)
print()

# Load data
print("Loading data...")
with open('model_predictions_quarterly.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_1day_with_spreads.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

print("Data loaded")
print()

# Fixed parameters
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.007
MAX_TOTAL_POSITIONS = 90
MAX_POSITIONS_PER_PAIR = 12

# Emergency stop parameters
EMERGENCY_STOP_LOSS_PCT = -0.04
EMERGENCY_STOP_DAYS = 15

# Trailing stop parameters (fixed for now)
TRAILING_STOP_TRIGGER = 0.005
TRAILING_STOP_PCT = 0.60


class Position:
    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence,
                 ladder_levels, ladder_scale_pct, early_stop_days=None, early_stop_loss_pct=None):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.original_size = size
        self.breakout_target = breakout_target
        self.confidence = confidence
        self.days_held = 0
        self.max_profit = 0
        self.trailing_stop = None
        self.partial_exits = []
        self.ladder_level = 0
        self.ladder_levels = ladder_levels
        self.ladder_scale_pct = ladder_scale_pct
        self.early_stop_days = early_stop_days
        self.early_stop_loss_pct = early_stop_loss_pct

    def update(self, date, high, low, close):
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

        # Check ladder
        if self.ladder_level < len(self.ladder_levels):
            if intraday_high_profit >= self.ladder_levels[self.ladder_level]:
                self.partial_exits.append((self.ladder_levels[self.ladder_level], self.ladder_scale_pct))
                self.size *= (1 - self.ladder_scale_pct)
                self.ladder_level += 1
                return None

        # Early stop (tighter than emergency stop, for cutting losers faster)
        if self.early_stop_days is not None and self.early_stop_loss_pct is not None:
            if self.days_held >= self.early_stop_days and current_profit < self.early_stop_loss_pct:
                return 'early_stop', close, current_profit

        # Emergency stop
        if self.days_held >= EMERGENCY_STOP_DAYS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            return 'emergency_stop', close, current_profit

        # Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > TRAILING_STOP_TRIGGER:
                self.trailing_stop = self.entry_price
        else:
            old_stop = self.trailing_stop

            if self.direction == 'long':
                hit_stop = low <= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price + (high - self.entry_price) * TRAILING_STOP_PCT
                self.trailing_stop = max(self.trailing_stop, new_stop)
            else:
                hit_stop = high >= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price - (self.entry_price - low) * TRAILING_STOP_PCT
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


def run_backtest_with_params(min_confidence, ladder_levels, ladder_scale_pct,
                              early_stop_days=None, early_stop_loss_pct=None):
    """Run full backtest with specific parameter set"""
    capital = INITIAL_CAPITAL
    carried_positions = []
    total_trades = 0

    for quarter_name, quarter_preds in sorted(all_predictions.items()):
        # Get trading days
        all_trading_days = set()
        prediction_dates = set()

        for pair_df in quarter_preds.values():
            prediction_dates.update(pair_df.index)

        for pair, pair_df in all_raw_data.items():
            if len(prediction_dates) > 0:
                min_date = min(prediction_dates)
                max_date = max(prediction_dates)
                trading_days = pair_df[(pair_df.index >= min_date) & (pair_df.index <= max_date)].index
                all_trading_days.update(trading_days)

        all_trading_days = sorted(list(all_trading_days))

        for date in all_trading_days:
            # Get prices
            prices_dict = {}
            for pair, pair_df in all_raw_data.items():
                if date in pair_df.index:
                    row = pair_df.loc[date]
                    prices_dict[pair] = {'high': row['high'], 'low': row['low'], 'close': row['close']}

            # Update positions
            positions_to_close = []
            for position in carried_positions:
                if position.pair not in prices_dict:
                    continue
                high = prices_dict[position.pair]['high']
                low = prices_dict[position.pair]['low']
                close = prices_dict[position.pair]['close']
                exit_info = position.update(date, high, low, close)
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
                carried_positions.remove(position)
                total_trades += 1

            # Open new positions
            if date not in prediction_dates:
                continue

            if len(carried_positions) >= MAX_TOTAL_POSITIONS:
                continue

            for pair, pair_df in quarter_preds.items():
                if date not in pair_df.index:
                    continue

                pair_positions = [p for p in carried_positions if p.pair == pair]
                if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
                    continue

                row = pair_df.loc[date]

                breakout_high_prob = row['breakout_high_prob']
                breakout_low_prob = row['breakout_low_prob']
                max_prob = max(breakout_high_prob, breakout_low_prob)

                if max_prob <= min_confidence:
                    continue

                assumed_risk_pct = 0.02
                risk_amount = capital * RISK_PER_TRADE
                price = row['close']
                position_size = risk_amount / (price * assumed_risk_pct)

                if breakout_high_prob > breakout_low_prob:
                    direction = 'long'
                    breakout_level = row['high_20d']
                    target = breakout_level * 1.005
                else:
                    direction = 'short'
                    breakout_level = row['low_20d']
                    target = breakout_level * 0.995

                position = Position(pair, date, price, direction, position_size, target, max_prob,
                                    ladder_levels, ladder_scale_pct, early_stop_days, early_stop_loss_pct)
                carried_positions.append(position)

    return capital, total_trades


# PHASE 1: Optimize confidence threshold
print("="*100)
print("PHASE 1: OPTIMIZING CONFIDENCE THRESHOLD")
print("="*100)
print()

baseline_ladder = [0.008, 0.015]
baseline_scale = 0.33

confidence_tests = [0.65, 0.70, 0.75, 0.80, 0.85]
confidence_results = []

print("Testing confidence thresholds...")
for conf in confidence_tests:
    final_cap, trades = run_backtest_with_params(conf, baseline_ladder, baseline_scale)
    total_return = (final_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    confidence_results.append({
        'confidence': conf,
        'final_capital': final_cap,
        'total_return': total_return,
        'total_trades': trades
    })
    print(f"  Confidence {conf:.2f}: ${final_cap:>8,.0f} ({total_return:>7.1%}) | {trades:>5,} trades")

best_conf = max(confidence_results, key=lambda x: x['final_capital'])
print()
print(f"Best confidence threshold: {best_conf['confidence']:.2f} -> ${best_conf['final_capital']:,.0f}")
print()

# PHASE 2: Optimize ladder parameters
print("="*100)
print("PHASE 2: OPTIMIZING LADDER LEVELS")
print("="*100)
print()

best_confidence = best_conf['confidence']

# Test different ladder configurations
ladder_configs = [
    ([0.008, 0.015], 0.33),  # Current
    ([0.006, 0.012], 0.33),  # Earlier exits
    ([0.010, 0.020], 0.33),  # Later exits
    ([0.008, 0.015], 0.25),  # Smaller scale
    ([0.008, 0.015], 0.40),  # Larger scale
    ([0.005, 0.010, 0.015], 0.25),  # 3 levels
    ([0.008], 0.50),  # Single level, half out
]

ladder_results = []

print("Testing ladder configurations...")
for levels, scale in ladder_configs:
    final_cap, trades = run_backtest_with_params(best_confidence, levels, scale)
    total_return = (final_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    ladder_results.append({
        'levels': levels,
        'scale': scale,
        'final_capital': final_cap,
        'total_return': total_return,
        'total_trades': trades
    })
    levels_str = '/'.join([f"{l:.3f}" for l in levels])
    print(f"  Levels [{levels_str}] @ {scale:.0%}: ${final_cap:>8,.0f} ({total_return:>7.1%}) | {trades:>5,} trades")

best_ladder = max(ladder_results, key=lambda x: x['final_capital'])
print()
levels_str = '/'.join([f"{l:.3f}" for l in best_ladder['levels']])
print(f"Best ladder config: [{levels_str}] @ {best_ladder['scale']:.0%} -> ${best_ladder['final_capital']:,.0f}")
print()

# PHASE 3: Optimize early stop (active management)
print("="*100)
print("PHASE 3: OPTIMIZING EARLY STOP (ACTIVE MANAGEMENT)")
print("="*100)
print()

best_ladder_levels = best_ladder['levels']
best_ladder_scale = best_ladder['scale']

# Test early stop configurations
early_stop_configs = [
    (None, None),  # No early stop (baseline)
    (5, -0.015),   # Very aggressive: 5 days, -1.5%
    (7, -0.02),    # Aggressive: 7 days, -2%
    (10, -0.025),  # Moderate: 10 days, -2.5%
    (10, -0.03),   # Moderate: 10 days, -3%
]

early_stop_results = []

print("Testing early stop configurations...")
for days, loss_pct in early_stop_configs:
    final_cap, trades = run_backtest_with_params(best_confidence, best_ladder_levels, best_ladder_scale,
                                                  days, loss_pct)
    total_return = (final_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    early_stop_results.append({
        'days': days,
        'loss_pct': loss_pct,
        'final_capital': final_cap,
        'total_return': total_return,
        'total_trades': trades
    })
    if days is None:
        print(f"  No early stop:            ${final_cap:>8,.0f} ({total_return:>7.1%}) | {trades:>5,} trades")
    else:
        print(f"  {days:>2} days @ {loss_pct:>+6.1%}:        ${final_cap:>8,.0f} ({total_return:>7.1%}) | {trades:>5,} trades")

best_early_stop = max(early_stop_results, key=lambda x: x['final_capital'])
print()
if best_early_stop['days'] is None:
    print(f"Best early stop: None -> ${best_early_stop['final_capital']:,.0f}")
else:
    print(f"Best early stop: {best_early_stop['days']} days @ {best_early_stop['loss_pct']:+.1%} -> ${best_early_stop['final_capital']:,.0f}")
print()

# FINAL SUMMARY
print("="*100)
print("OPTIMIZATION SUMMARY")
print("="*100)
print()
print("BASELINE (current settings):")
print(f"  Confidence: 0.70")
print(f"  Ladder: [0.008, 0.015] @ 33%")
print(f"  Early stop: None")
baseline_cap = confidence_results[1]['final_capital']  # 0.70 confidence
print(f"  Result: ${baseline_cap:,.0f}")
print()
print("OPTIMIZED SETTINGS:")
print(f"  Confidence: {best_confidence:.2f}")
levels_str = ', '.join([f"{l:.3f}" for l in best_ladder_levels])
print(f"  Ladder: [{levels_str}] @ {best_ladder_scale:.0%}")
if best_early_stop['days'] is None:
    print(f"  Early stop: None")
else:
    print(f"  Early stop: {best_early_stop['days']} days @ {best_early_stop['loss_pct']:+.1%}")
optimized_cap = best_early_stop['final_capital']
print(f"  Result: ${optimized_cap:,.0f}")
print()
improvement = (optimized_cap - baseline_cap) / baseline_cap
print(f"Improvement: {improvement:+.1%} (${optimized_cap - baseline_cap:+,.0f})")
print()
print("="*100)
