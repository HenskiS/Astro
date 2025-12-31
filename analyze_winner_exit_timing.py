"""
WINNER EXIT TIMING ANALYSIS
============================
Show how sample composition changes over time.
Demonstrate that negative averages are due to fast winners exiting early.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

print("="*100)
print("WINNER EXIT TIMING ANALYSIS - Why Are Averages Negative?")
print("="*100)
print()

# Load and run backtest (simplified version)
INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.007
MIN_CONFIDENCE = 0.70
EMERGENCY_STOP_LOSS_PCT = -0.04
EMERGENCY_STOP_DAYS = 15
TRAILING_STOP_TRIGGER = 0.005
TRAILING_STOP_PCT = 0.60


class SimplePosition:
    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.breakout_target = breakout_target
        self.confidence = confidence
        self.days_held = 0
        self.max_profit = 0
        self.max_adverse_excursion = 0
        self.trailing_stop = None
        self.daily_progression = []

    def update(self, date, high, low, close):
        self.days_held += 1

        if self.direction == 'long':
            current_profit = (close - self.entry_price) / self.entry_price
            intraday_high_profit = (high - self.entry_price) / self.entry_price
            intraday_low_profit = (low - self.entry_price) / self.entry_price
            hit_target = high >= self.breakout_target
        else:
            current_profit = (self.entry_price - close) / self.entry_price
            intraday_high_profit = (self.entry_price - low) / self.entry_price
            intraday_low_profit = (self.entry_price - high) / self.entry_price
            hit_target = low <= self.breakout_target

        self.daily_progression.append({
            'day': self.days_held,
            'date': date,
            'profit': current_profit,
            'high_profit': intraday_high_profit,
            'low_profit': intraday_low_profit
        })

        self.max_profit = max(self.max_profit, intraday_high_profit)
        self.max_adverse_excursion = min(self.max_adverse_excursion, intraday_low_profit)

        # Emergency stop
        if self.days_held >= EMERGENCY_STOP_DAYS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            return 'emergency_stop', close, current_profit

        # Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > TRAILING_STOP_TRIGGER:
                self.trailing_stop = self.entry_price
        else:
            if self.direction == 'long':
                new_stop = self.entry_price + (high - self.entry_price) * TRAILING_STOP_PCT
                self.trailing_stop = max(self.trailing_stop, new_stop)
                hit_stop = low <= self.trailing_stop
            else:
                new_stop = self.entry_price - (self.entry_price - low) * TRAILING_STOP_PCT
                self.trailing_stop = min(self.trailing_stop, new_stop)
                hit_stop = high >= self.trailing_stop

            if hit_stop:
                return 'trailing_stop', self.trailing_stop, current_profit

        # Target hit
        if hit_target:
            return 'target', self.breakout_target, current_profit

        return None, close, current_profit


def run_backtest(period_predictions):
    capital = INITIAL_CAPITAL
    positions = []
    closed_positions = []

    all_dates = set()
    for pair_df in period_predictions.values():
        dates = pd.to_datetime(pair_df.index).tz_localize(None)
        all_dates.update(dates)
    all_dates = sorted(list(all_dates))

    for date in all_dates:
        prices_dict = {}
        for pair, pair_df in period_predictions.items():
            date_with_tz = pd.Timestamp(date).tz_localize('UTC')
            if date_with_tz in pair_df.index:
                row = pair_df.loc[date_with_tz]
                prices_dict[pair] = {'high': row['high'], 'low': row['low'], 'close': row['close']}

        positions_to_close = []
        for position in positions:
            if position.pair not in prices_dict:
                continue
            high = prices_dict[position.pair]['high']
            low = prices_dict[position.pair]['low']
            close = prices_dict[position.pair]['close']
            exit_reason, exit_price, current_profit = position.update(date, high, low, close)
            if exit_reason is not None:
                positions_to_close.append((position, exit_price, exit_reason))

        for position, exit_price, reason in positions_to_close:
            if position.direction == 'long':
                profit_pct = (exit_price - position.entry_price) / position.entry_price
            else:
                profit_pct = (position.entry_price - exit_price) / position.entry_price

            position.exit_date = date
            position.exit_price = exit_price
            position.profit_pct = profit_pct
            position.exit_reason = reason
            closed_positions.append(position)
            positions.remove(position)

        for pair, pair_df in period_predictions.items():
            date_with_tz = pd.Timestamp(date).tz_localize('UTC')
            if date_with_tz not in pair_df.index:
                continue
            row = pair_df.loc[date_with_tz]
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

            position = SimplePosition(pair, date, price, direction, position_size, target, max_prob)
            positions.append(position)

    return closed_positions


# Load data and run
print("Loading predictions and running backtest...")
with open('model_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

all_positions = []
for period_name, period_preds in all_predictions.items():
    positions = run_backtest(period_preds)
    all_positions.extend(positions)

winners = [p for p in all_positions if p.profit_pct > 0]
print(f"Analyzing {len(winners)} winning trades")
print()

# Categorize winners by exit day
exit_day_groups = {
    '1-3 days': [w for w in winners if w.days_held <= 3],
    '4-7 days': [w for w in winners if 4 <= w.days_held <= 7],
    '8-14 days': [w for w in winners if 8 <= w.days_held <= 14],
    '15+ days': [w for w in winners if w.days_held >= 15]
}

print("="*100)
print("WINNERS BY EXIT TIMING")
print("="*100)
print()

for group_name, trades in exit_day_groups.items():
    if len(trades) > 0:
        avg_profit = np.mean([t.profit_pct for t in trades])
        avg_exit_day = np.mean([t.days_held for t in trades])
        pct_of_winners = len(trades) / len(winners)

        print(f"{group_name}:")
        print(f"  Count: {len(trades)} trades ({pct_of_winners:.1%} of all winners)")
        print(f"  Avg profit: {avg_profit:+.2%}")
        print(f"  Avg exit day: {avg_exit_day:.1f}")
        print()

# Now show P&L progression for each group
print("="*100)
print("P&L PROGRESSION BY EXIT GROUP")
print("="*100)
print()

print("This shows WHY the average is negative - fast winners exit early!")
print()

for check_day in [1, 3, 5, 7, 10]:
    print(f"DAY {check_day} P&L:")
    print()

    for group_name, trades in exit_day_groups.items():
        # Get trades that WERE STILL OPEN at this day (hadn't exited yet)
        trades_still_open = [t for t in trades if len(t.daily_progression) >= check_day]

        if len(trades_still_open) > 0:
            avg_pl = np.mean([t.daily_progression[check_day-1]['profit'] for t in trades_still_open])
            still_in_sample = len(trades_still_open) / len(trades)

            print(f"  {group_name:<15}: {avg_pl:>7.2%}  ({len(trades_still_open):5} trades, "
                  f"{still_in_sample:5.1%} of group still open)")

    print()

# Show the composition shift
print("="*100)
print("SAMPLE COMPOSITION OVER TIME")
print("="*100)
print()

print("This shows how the sample changes as fast winners exit:")
print()

for check_day in [1, 3, 5, 7, 10, 15, 20]:
    print(f"DAY {check_day}:")

    total_still_open = 0
    for group_name, trades in exit_day_groups.items():
        trades_still_open = [t for t in trades if len(t.daily_progression) >= check_day]
        total_still_open += len(trades_still_open)

    if total_still_open > 0:
        for group_name, trades in exit_day_groups.items():
            trades_still_open = [t for t in trades if len(t.daily_progression) >= check_day]
            pct_of_sample = len(trades_still_open) / total_still_open

            print(f"  {group_name:<15}: {len(trades_still_open):5} trades ({pct_of_sample:5.1%} of remaining sample)")

        # Calculate average P&L
        all_still_open = []
        for trades in exit_day_groups.values():
            all_still_open.extend([t for t in trades if len(t.daily_progression) >= check_day])

        avg_pl = np.mean([t.daily_progression[check_day-1]['profit'] for t in all_still_open])
        print(f"  {'AVERAGE P&L':<15}: {avg_pl:>7.2%}")

    print()

# Show final profits by group
print("="*100)
print("THE KEY INSIGHT")
print("="*100)
print()

print("FAST WINNERS (exit by day 3):")
fast = exit_day_groups['1-3 days']
if len(fast) > 0:
    print(f"  {len(fast)} trades ({len(fast)/len(winners):.1%} of all winners)")
    print(f"  Final profit: {np.mean([t.profit_pct for t in fast]):+.2%}")
    print(f"  P&L at day 1: {np.mean([t.daily_progression[0]['profit'] for t in fast if len(t.daily_progression) >= 1]):+.2%}")
    print(f"  P&L at day 2: {np.mean([t.daily_progression[1]['profit'] for t in fast if len(t.daily_progression) >= 2]):+.2%}")
    print(f"  These traders are POSITIVE early and exit quickly!")
    print()

print("SLOW WINNERS (exit after day 15):")
slow = exit_day_groups['15+ days']
if len(slow) > 0:
    print(f"  {len(slow)} trades ({len(slow)/len(winners):.1%} of all winners)")
    print(f"  Final profit: {np.mean([t.profit_pct for t in slow]):+.2%}")
    print(f"  P&L at day 5: {np.mean([t.daily_progression[4]['profit'] for t in slow if len(t.daily_progression) >= 5]):+.2%}")
    print(f"  P&L at day 10: {np.mean([t.daily_progression[9]['profit'] for t in slow if len(t.daily_progression) >= 10]):+.2%}")
    print(f"  P&L at day 15: {np.mean([t.daily_progression[14]['profit'] for t in slow if len(t.daily_progression) >= 15]):+.2%}")
    print(f"  These trades are DOWN for a long time before recovering!")
    print()

print("CONCLUSION:")
print("The 'average' at each day is dominated by SLOW winners (fast ones already exited).")
print("This creates SURVIVOR BIAS - we're only seeing the strugglers still in the game!")
print()
