"""
MONTHLY PERFORMANCE ANALYSIS
=============================
Analyze if losing months cluster together, suggesting we could use cooldowns
or regime filters to avoid bad trading periods.

Key Questions:
1. Do losses cluster in certain months/periods?
2. Can we predict bad months based on recent performance?
3. Would cooldowns after losing months improve results?
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import defaultdict
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*100)
print("MONTHLY PERFORMANCE ANALYSIS")
print("="*100)
print()

# Strategy Parameters
INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.007
MIN_CONFIDENCE = 0.70
EMERGENCY_STOP_LOSS_PCT = -0.04
EMERGENCY_STOP_DAYS = 15
TRAILING_STOP_TRIGGER = 0.005
TRAILING_STOP_PCT = 0.60

# Active management
SCALE_OUT_PROFIT_TRIGGER = 0.003
WEAK_MOMENTUM_THRESHOLD = 0.0015
SCALE_OUT_PCT = 0.50
MOMENTUM_LOOKBACK = 3


class ActivePosition:
    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence):
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
        self.max_adverse_excursion = 0
        self.trailing_stop = None
        self.daily_progression = []
        self.scaled_out = False
        self.scale_out_profit = 0

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
            'profit': current_profit,
            'high_profit': intraday_high_profit,
            'low_profit': intraday_low_profit
        })

        self.max_profit = max(self.max_profit, intraday_high_profit)
        self.max_adverse_excursion = min(self.max_adverse_excursion, intraday_low_profit)

        # Active management: scale out on weak momentum
        if not self.scaled_out and self.days_held >= MOMENTUM_LOOKBACK:
            if intraday_high_profit >= SCALE_OUT_PROFIT_TRIGGER:
                recent_profits = [d['profit'] for d in self.daily_progression[-MOMENTUM_LOOKBACK:]]
                momentum = np.mean([recent_profits[i] - recent_profits[i-1]
                                   for i in range(1, len(recent_profits))])

                if momentum < WEAK_MOMENTUM_THRESHOLD:
                    self.scaled_out = True
                    self.scale_out_profit = intraday_high_profit
                    self.size *= (1 - SCALE_OUT_PCT)

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

    def calculate_blended_profit(self, final_profit):
        if self.scaled_out:
            return (self.scale_out_profit * SCALE_OUT_PCT) + (final_profit * (1 - SCALE_OUT_PCT))
        else:
            return final_profit


def run_backtest(period_predictions):
    capital = INITIAL_CAPITAL
    positions = []
    closed_positions = []

    all_dates = set()
    for pair_df in period_predictions.values():
        dates = pd.to_datetime(pair_df.index).tz_localize(None)
        all_dates.update(dates)
    all_dates = sorted(list(all_dates))

    if len(all_dates) == 0:
        return [], []

    equity_curve = []
    equity_dates = []

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
                raw_profit = (exit_price - position.entry_price) / position.entry_price
            else:
                raw_profit = (position.entry_price - exit_price) / position.entry_price

            profit_pct = position.calculate_blended_profit(raw_profit)

            if position.scaled_out:
                scale_out_dollars = position.scale_out_profit * (position.original_size * SCALE_OUT_PCT * position.entry_price)
                remaining_dollars = raw_profit * (position.size * position.entry_price)
                profit_dollars = scale_out_dollars + remaining_dollars
            else:
                profit_dollars = profit_pct * (position.original_size * position.entry_price)

            capital += profit_dollars

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

            position = ActivePosition(pair, date, price, direction, position_size, target, max_prob)
            positions.append(position)

        equity_curve.append(capital)
        equity_dates.append(date)

    return closed_positions, list(zip(equity_dates, equity_curve))


# Load and run backtest
print("Loading predictions and running backtest...")
with open('model_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

all_positions = []
all_equity = []

for period_name, period_preds in all_predictions.items():
    print(f"  Processing {period_name}...")
    positions, equity = run_backtest(period_preds)
    all_positions.extend(positions)
    all_equity.extend(equity)

print(f"Collected {len(all_positions)} trades")
print()

# Create monthly performance DataFrame
monthly_data = []

for year_month in pd.period_range(start='2016-01', end='2025-12', freq='M'):
    year = year_month.year
    month = year_month.month

    # Get trades that EXITED in this month
    month_positions = [p for p in all_positions
                      if pd.Timestamp(p.exit_date).year == year
                      and pd.Timestamp(p.exit_date).month == month]

    if len(month_positions) == 0:
        continue

    # Calculate monthly metrics
    monthly_return = sum(p.profit_pct for p in month_positions) / len(month_positions)
    num_trades = len(month_positions)
    win_rate = sum(1 for p in month_positions if p.profit_pct > 0) / num_trades

    monthly_data.append({
        'year': year,
        'month': month,
        'year_month': f"{year}-{month:02d}",
        'return': monthly_return,
        'trades': num_trades,
        'win_rate': win_rate,
        'winners': sum(1 for p in month_positions if p.profit_pct > 0),
        'losers': sum(1 for p in month_positions if p.profit_pct <= 0)
    })

df_monthly = pd.DataFrame(monthly_data)

# Calculate statistics
print("="*100)
print("MONTHLY PERFORMANCE OVERVIEW")
print("="*100)
print()

print(f"Total months analyzed: {len(df_monthly)}")
print(f"Winning months: {sum(df_monthly['return'] > 0)} ({sum(df_monthly['return'] > 0)/len(df_monthly):.1%})")
print(f"Losing months: {sum(df_monthly['return'] < 0)} ({sum(df_monthly['return'] < 0)/len(df_monthly):.1%})")
print()

print(f"Average monthly return: {df_monthly['return'].mean():+.2%}")
print(f"Median monthly return: {df_monthly['return'].median():+.2%}")
print(f"Best month: {df_monthly['return'].max():+.2%}")
print(f"Worst month: {df_monthly['return'].min():+.2%}")
print(f"Std dev: {df_monthly['return'].std():.2%}")
print()

# Show all months
print("="*100)
print("MONTHLY BREAKDOWN")
print("="*100)
print()

print(f"{'Year-Month':<12} {'Return':>10} {'Trades':>8} {'Win Rate':>10} {'W/L':>8}")
print("-" * 65)

for _, row in df_monthly.iterrows():
    print(f"{row['year_month']:<12} {row['return']:>9.2%} {row['trades']:>8} "
          f"{row['win_rate']:>9.0%} {row['winners']:>3}/{row['losers']:<3}")

print()

# Analyze clustering of losing months
print("="*100)
print("LOSING MONTH CLUSTERING ANALYSIS")
print("="*100)
print()

# Find consecutive losing months
df_monthly['is_losing'] = df_monthly['return'] < 0
losing_streaks = []
current_streak = 0

for is_losing in df_monthly['is_losing']:
    if is_losing:
        current_streak += 1
    else:
        if current_streak > 0:
            losing_streaks.append(current_streak)
        current_streak = 0

if current_streak > 0:
    losing_streaks.append(current_streak)

print(f"Consecutive losing month streaks:")
print(f"  Single losing months: {losing_streaks.count(1)}")
print(f"  2-month losing streaks: {losing_streaks.count(2)}")
print(f"  3-month losing streaks: {losing_streaks.count(3)}")
print(f"  4+ month losing streaks: {sum(1 for x in losing_streaks if x >= 4)}")
print()

# Show the actual streaks
print("Losing streaks detail:")
streak_count = 0
current_streak_months = []

for idx, row in df_monthly.iterrows():
    if row['is_losing']:
        current_streak_months.append((row['year_month'], row['return']))
    else:
        if len(current_streak_months) > 0:
            streak_count += 1
            print(f"  Streak {streak_count} ({len(current_streak_months)} months):")
            for ym, ret in current_streak_months:
                print(f"    {ym}: {ret:+.2%}")
            current_streak_months = []

if len(current_streak_months) > 0:
    streak_count += 1
    print(f"  Streak {streak_count} ({len(current_streak_months)} months):")
    for ym, ret in current_streak_months:
        print(f"    {ym}: {ret:+.2%}")

print()

# Test cooldown strategies
print("="*100)
print("COOLDOWN STRATEGY TESTING")
print("="*100)
print()

# Strategy 1: Skip 1 month after losing month
print("STRATEGY 1: Skip trading for 1 month after any losing month")
months_skipped = 0
trades_skipped = 0
profit_given_up = 0

for idx in range(1, len(df_monthly)):
    prev_month = df_monthly.iloc[idx-1]
    this_month = df_monthly.iloc[idx]

    if prev_month['return'] < 0:  # Previous month was losing
        months_skipped += 1
        trades_skipped += this_month['trades']
        profit_given_up += this_month['return'] * this_month['trades']

print(f"  Would skip {months_skipped} months ({months_skipped/len(df_monthly):.1%} of time)")
print(f"  Would skip {trades_skipped} trades")
print(f"  Would give up: {profit_given_up/trades_skipped:+.2%} avg per trade")
print(f"  Net impact: {profit_given_up:+.2%} total")
print()

# Strategy 2: Skip 1 month after 2 consecutive losing months
print("STRATEGY 2: Skip trading for 1 month after 2 consecutive losing months")
months_skipped = 0
trades_skipped = 0
profit_given_up = 0

for idx in range(2, len(df_monthly)):
    prev_month = df_monthly.iloc[idx-1]
    prev_prev_month = df_monthly.iloc[idx-2]
    this_month = df_monthly.iloc[idx]

    if prev_month['return'] < 0 and prev_prev_month['return'] < 0:
        months_skipped += 1
        trades_skipped += this_month['trades']
        profit_given_up += this_month['return'] * this_month['trades']

print(f"  Would skip {months_skipped} months ({months_skipped/len(df_monthly):.1%} of time)")
print(f"  Would skip {trades_skipped} trades")
if trades_skipped > 0:
    print(f"  Would give up: {profit_given_up/trades_skipped:+.2%} avg per trade")
    print(f"  Net impact: {profit_given_up:+.2%} total")
print()

# Strategy 3: Reduce position size by 50% after losing month
print("STRATEGY 3: Reduce position size by 50% for 1 month after any losing month")
total_impact = 0

for idx in range(1, len(df_monthly)):
    prev_month = df_monthly.iloc[idx-1]
    this_month = df_monthly.iloc[idx]

    if prev_month['return'] < 0:
        # Would have traded at 50% size
        # Impact = (current return) * 0.5 * num_trades - (current return) * num_trades
        # Impact = current return * num_trades * -0.5
        impact = this_month['return'] * this_month['trades'] * -0.5
        total_impact += impact

print(f"  Net impact: {total_impact:+.2%} total return")
print(f"  (Negative means we lose money, positive means we save money)")
print()

# Analyze by season/calendar patterns
print("="*100)
print("SEASONAL PATTERNS")
print("="*100)
print()

for month_num in range(1, 13):
    month_name = datetime(2020, month_num, 1).strftime('%B')
    month_data = df_monthly[df_monthly['month'] == month_num]

    if len(month_data) > 0:
        avg_return = month_data['return'].mean()
        winning_pct = sum(month_data['return'] > 0) / len(month_data)

        print(f"{month_name:<12}: {avg_return:>7.2%} avg return, "
              f"{winning_pct:>5.0%} winning months ({len(month_data)} samples)")

print()

print("="*100)
print("ANALYSIS COMPLETE")
print("="*100)
