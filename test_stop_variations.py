"""
TEST STOP LOSS VARIATIONS
==========================
Test the three promising stop loss improvements:
1. Day 10 / -2.5% early stop
2. Dynamic stops by confidence level
3. Reduce emergency stop days (15 -> 12)
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TESTING STOP LOSS VARIATIONS")
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

# Fixed parameters (optimized)
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.007
MIN_CONFIDENCE = 0.65
MAX_TOTAL_POSITIONS = 90
MAX_POSITIONS_PER_PAIR = 12
TRAILING_STOP_TRIGGER = 0.005
TRAILING_STOP_PCT = 0.60
LADDER_LEVELS = [0.008, 0.015]
LADDER_SCALE_PCT = 0.40


class Position:
    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence,
                 emergency_stop_days, emergency_stop_loss_pct,
                 early_stop_days=None, early_stop_loss_pct=None,
                 dynamic_stop=False):
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

        # Stop loss parameters
        self.emergency_stop_days = emergency_stop_days

        # Dynamic stop based on confidence
        if dynamic_stop:
            if confidence < 0.70:
                self.emergency_stop_loss_pct = -0.035  # -3.5% for low confidence
            elif confidence < 0.80:
                self.emergency_stop_loss_pct = -0.040  # -4.0% for medium confidence
            else:
                self.emergency_stop_loss_pct = -0.045  # -4.5% for high confidence
        else:
            self.emergency_stop_loss_pct = emergency_stop_loss_pct

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
        if self.ladder_level < len(LADDER_LEVELS):
            if intraday_high_profit >= LADDER_LEVELS[self.ladder_level]:
                self.partial_exits.append((LADDER_LEVELS[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
                self.ladder_level += 1
                return None

        # Early stop (tighter, kicks in earlier)
        if self.early_stop_days is not None and self.early_stop_loss_pct is not None:
            if self.days_held >= self.early_stop_days and current_profit < self.early_stop_loss_pct:
                return 'early_stop', close, current_profit

        # Emergency stop
        if self.days_held >= self.emergency_stop_days and current_profit < self.emergency_stop_loss_pct:
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


def run_backtest_with_stops(emergency_days, emergency_loss_pct,
                             early_stop_days=None, early_stop_loss_pct=None,
                             dynamic_stop=False):
    """Run full backtest with specific stop parameters"""
    capital = INITIAL_CAPITAL
    carried_positions = []
    total_trades = 0
    trades = []
    equity_curve = [INITIAL_CAPITAL]  # Track equity for drawdown calculation

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
                equity_curve.append(capital)  # Track equity after each trade

                trades.append({
                    'exit_reason': exit_reason,
                    'profit_pct': profit_pct,
                    'days_held': position.days_held
                })

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

                if max_prob <= MIN_CONFIDENCE:
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
                                    emergency_days, emergency_loss_pct,
                                    early_stop_days, early_stop_loss_pct,
                                    dynamic_stop)
                carried_positions.append(position)

    # Calculate max drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    for equity in equity_curve:
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

    return capital, total_trades, trades, max_dd


print("="*100)
print("TEST 1: DAY 10 / -2.5% EARLY STOP")
print("="*100)
print()

# Baseline (current)
baseline_cap, baseline_trades, baseline_trade_list, baseline_dd = run_backtest_with_stops(15, -0.04)
baseline_return = (baseline_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL

print(f"Baseline (15d/-4%):           ${baseline_cap:>10,.0f} ({baseline_return:>7.1%}) | DD: {baseline_dd:>6.1%} | {baseline_trades:>5,} trades")

# Test early stop variations
early_stop_tests = [
    (10, -0.025, "Day 10 @ -2.5%"),
    (10, -0.030, "Day 10 @ -3.0%"),
    (12, -0.025, "Day 12 @ -2.5%"),
    (12, -0.030, "Day 12 @ -3.0%"),
]

print()
print("Early Stop Tests:")
for days, loss, label in early_stop_tests:
    cap, trades, trade_list, max_dd = run_backtest_with_stops(15, -0.04, days, loss)
    total_return = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    improvement = cap - baseline_cap
    dd_improvement = max_dd - baseline_dd

    print(f"  {label:20s} ${cap:>10,.0f} ({total_return:>7.1%}) | DD: {max_dd:>6.1%} ({dd_improvement:>+5.1%}) | {improvement:>+8,.0f}")

print()

# Analyze exit reasons for best early stop
best_early_cap, best_early_trades, best_early_trade_list, best_early_dd = run_backtest_with_stops(15, -0.04, 10, -0.025)
trade_df = pd.DataFrame(best_early_trade_list)
print("Exit reason breakdown (10d/-2.5% early stop):")
for reason in trade_df['exit_reason'].unique():
    count = (trade_df['exit_reason'] == reason).sum()
    avg_pl = trade_df[trade_df['exit_reason'] == reason]['profit_pct'].mean()
    print(f"  {reason:20s} {count:>6,} trades ({count/len(trade_df):>5.1%}) | Avg P/L: {avg_pl:>+7.2%}")

print()

print("="*100)
print("TEST 2: DYNAMIC STOPS BY CONFIDENCE")
print("="*100)
print()

print("Testing dynamic stops (confidence-based thresholds):")
print("  Confidence 0.65-0.70: -3.5% stop (more defensive)")
print("  Confidence 0.70-0.80: -4.0% stop (standard)")
print("  Confidence 0.80+:     -4.5% stop (more patient)")
print()

dynamic_cap, dynamic_trades, dynamic_trade_list, dynamic_dd = run_backtest_with_stops(15, -0.04, dynamic_stop=True)
dynamic_return = (dynamic_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
dynamic_improvement = dynamic_cap - baseline_cap
dynamic_dd_improvement = dynamic_dd - baseline_dd

print(f"Dynamic stops:  ${dynamic_cap:>10,.0f} ({dynamic_return:>7.1%}) | DD: {dynamic_dd:>6.1%} ({dynamic_dd_improvement:>+5.1%}) | {dynamic_improvement:>+8,.0f}")
print()

print("="*100)
print("TEST 3: REDUCE EMERGENCY STOP DAYS")
print("="*100)
print()

print("Testing shorter emergency stop periods:")
days_tests = [
    (15, "15 days (current)"),
    (14, "14 days"),
    (13, "13 days"),
    (12, "12 days"),
    (11, "11 days"),
    (10, "10 days"),
]

for days, label in days_tests:
    cap, trades, trade_list, max_dd = run_backtest_with_stops(days, -0.04)
    total_return = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    improvement = cap - baseline_cap
    dd_improvement = max_dd - baseline_dd

    print(f"  {label:20s} ${cap:>10,.0f} ({total_return:>7.1%}) | DD: {max_dd:>6.1%} ({dd_improvement:>+5.1%}) | {improvement:>+8,.0f}")

print()

print("="*100)
print("TEST 4: COMBINATIONS")
print("="*100)
print()

print("Testing combinations of improvements:")
print()

# Combination tests
combos = [
    (12, -0.04, None, None, False, "12 days @ -4%"),
    (12, -0.04, 10, -0.025, False, "12d/-4% + early 10d/-2.5%"),
    (15, -0.04, 10, -0.025, True, "Dynamic stops + early 10d/-2.5%"),
    (12, -0.04, 10, -0.025, True, "12d/-4% + early 10d/-2.5% + dynamic"),
]

for emerg_days, emerg_loss, early_days, early_loss, dynamic, label in combos:
    cap, trades, trade_list, max_dd = run_backtest_with_stops(emerg_days, emerg_loss, early_days, early_loss, dynamic)
    total_return = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    improvement = cap - baseline_cap
    dd_improvement = max_dd - baseline_dd

    print(f"  {label:45s} ${cap:>10,.0f} ({total_return:>7.1%}) | DD: {max_dd:>6.1%} ({dd_improvement:>+5.1%}) | {improvement:>+8,.0f}")

print()

print("="*100)
print("SUMMARY")
print("="*100)
print()

# Find best configuration
all_tests = [
    (baseline_cap, baseline_dd, "Baseline (15d/-4%)"),
    (run_backtest_with_stops(15, -0.04, 10, -0.025)[0], run_backtest_with_stops(15, -0.04, 10, -0.025)[3], "Early 10d/-2.5%"),
    (run_backtest_with_stops(15, -0.04, dynamic_stop=True)[0], run_backtest_with_stops(15, -0.04, dynamic_stop=True)[3], "Dynamic stops"),
    (run_backtest_with_stops(12, -0.04)[0], run_backtest_with_stops(12, -0.04)[3], "12 days @ -4%"),
    (run_backtest_with_stops(12, -0.04, 10, -0.025, False)[0], run_backtest_with_stops(12, -0.04, 10, -0.025, False)[3], "12d/-4% + early 10d/-2.5%"),
    (run_backtest_with_stops(15, -0.04, 10, -0.025, True)[0], run_backtest_with_stops(15, -0.04, 10, -0.025, True)[3], "Dynamic + early 10d/-2.5%"),
    (run_backtest_with_stops(12, -0.04, 10, -0.025, True)[0], run_backtest_with_stops(12, -0.04, 10, -0.025, True)[3], "Full combo"),
]

sorted_tests = sorted(all_tests, key=lambda x: x[0], reverse=True)

print("Rankings (by final capital):")
for i, (cap, dd, label) in enumerate(sorted_tests, 1):
    ret = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    improvement = cap - baseline_cap
    dd_improvement = dd - baseline_dd
    print(f"  {i}. {label:45s} ${cap:>10,.0f} ({ret:>7.1%}) | DD: {dd:>6.1%} ({dd_improvement:>+5.1%}) | {improvement:>+8,.0f}")

print()

best_cap, best_dd, best_label = sorted_tests[0]
best_improvement = best_cap - baseline_cap
best_improvement_pct = best_improvement / baseline_cap
best_dd_improvement = best_dd - baseline_dd

print(f"Best configuration: {best_label}")
print(f"  Final capital: ${best_cap:,.0f}")
print(f"  Max drawdown: {best_dd:.1%}")
print(f"  Return improvement: ${best_improvement:+,.0f} ({best_improvement_pct:+.1%})")
print(f"  Drawdown improvement: {best_dd_improvement:+.1%}")
print()

if best_improvement > 100:
    print("RECOMMENDATION: Adopt the best configuration")
elif best_improvement > 0:
    print("RECOMMENDATION: Marginal improvement - consider adopting if benefits outweigh complexity")
else:
    print("RECOMMENDATION: Keep baseline configuration (current settings are optimal)")

print()
print("="*100)
