"""
EQUITY CURVE COMPARISON
=======================
Compare baseline vs optimal ladder strategy equity curves
"""
import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*100)
print("GENERATING EQUITY CURVES")
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

# Ladder parameters
LADDER_LEVELS = [0.008, 0.015]
LADDER_SCALE_PCT = 0.33


class BaselinePosition:
    """Baseline position without ladder"""
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
        self.trailing_stop = None

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

        if self.days_held >= EMERGENCY_STOP_DAYS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            return 'emergency_stop', close, current_profit

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

        if hit_target:
            return 'target', self.breakout_target, current_profit

        return None


class LadderPosition:
    """Position with ladder"""
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
        self.trailing_stop = None
        self.partial_exits = []
        self.ladder_level = 0

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

        # Ladder
        if self.ladder_level < len(LADDER_LEVELS):
            if intraday_high_profit >= LADDER_LEVELS[self.ladder_level]:
                self.partial_exits.append((LADDER_LEVELS[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
                self.ladder_level += 1
                return None

        if self.days_held >= EMERGENCY_STOP_DAYS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            return 'emergency_stop', close, current_profit

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


def run_backtest(period_predictions, use_ladder=False):
    """Run backtest with or without ladder"""
    capital = INITIAL_CAPITAL
    positions = []
    equity_curve = []
    equity_dates = []

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
            exit_info = position.update(date, high, low, close)
            if exit_info is not None:
                positions_to_close.append((position, exit_info))

        for position, exit_info in positions_to_close:
            exit_reason, exit_price, current_profit = exit_info

            if position.direction == 'long':
                raw_profit = (exit_price - position.entry_price) / position.entry_price
            else:
                raw_profit = (position.entry_price - exit_price) / position.entry_price

            if use_ladder and hasattr(position, 'calculate_blended_profit'):
                profit_pct = position.calculate_blended_profit(raw_profit)
                profit_dollars = profit_pct * (position.original_size * position.entry_price)
            else:
                profit_pct = raw_profit
                profit_dollars = profit_pct * (position.size * position.entry_price)

            capital += profit_dollars
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

            if use_ladder:
                position = LadderPosition(pair, date, price, direction, position_size, target, max_prob)
            else:
                position = BaselinePosition(pair, date, price, direction, position_size, target, max_prob)
            positions.append(position)

        equity_curve.append(capital)
        equity_dates.append(date)

    return equity_dates, equity_curve


# Load predictions
print("Loading predictions...")
with open('model_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

# Combine all periods
print("Running baseline backtest...")
baseline_dates = []
baseline_equity = []

for period_name, period_preds in all_predictions.items():
    dates, equity = run_backtest(period_preds, use_ladder=False)
    baseline_dates.extend(dates)
    baseline_equity.extend(equity)

print("Running optimal ladder backtest...")
ladder_dates = []
ladder_equity = []

for period_name, period_preds in all_predictions.items():
    dates, equity = run_backtest(period_preds, use_ladder=True)
    ladder_dates.extend(dates)
    ladder_equity.extend(equity)

print()
print("Creating plots...")

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Breakout Strategy: Baseline vs Optimal Ladder', fontsize=16, fontweight='bold')

# Plot 1: Equity Curves
ax1 = axes[0]
ax1.plot(baseline_dates, baseline_equity, label='Baseline', linewidth=2, color='#1f77b4', alpha=0.8)
ax1.plot(ladder_dates, ladder_equity, label='Optimal Ladder (0.8%, 1.5%)', linewidth=2, color='#2ca02c', alpha=0.8)
ax1.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
ax1.set_title('Equity Growth Comparison', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Format y-axis
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Plot 2: Drawdown
ax2 = axes[1]

baseline_series = pd.Series(baseline_equity, index=baseline_dates)
baseline_peak = baseline_series.expanding().max()
baseline_dd = (baseline_series - baseline_peak) / baseline_peak * 100

ladder_series = pd.Series(ladder_equity, index=ladder_dates)
ladder_peak = ladder_series.expanding().max()
ladder_dd = (ladder_series - ladder_peak) / ladder_peak * 100

ax2.plot(baseline_dates, baseline_dd, label='Baseline DD', linewidth=2, color='#1f77b4', alpha=0.8)
ax2.plot(ladder_dates, ladder_dd, label='Optimal Ladder DD', linewidth=2, color='#2ca02c', alpha=0.8)
ax2.fill_between(baseline_dates, baseline_dd, 0, alpha=0.2, color='#1f77b4')
ax2.fill_between(ladder_dates, ladder_dd, 0, alpha=0.2, color='#2ca02c')
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_title('Drawdown Comparison', fontsize=13, fontweight='bold')
ax2.legend(loc='lower left', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# Plot 3: Returns (%)
ax3 = axes[2]

baseline_pct = [(x - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100 for x in baseline_equity]
ladder_pct = [(x - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100 for x in ladder_equity]

ax3.plot(baseline_dates, baseline_pct, label='Baseline', linewidth=2, color='#1f77b4', alpha=0.8)
ax3.plot(ladder_dates, ladder_pct, label='Optimal Ladder', linewidth=2, color='#2ca02c', alpha=0.8)
ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
ax3.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
ax3.set_title('Cumulative Returns', fontsize=13, fontweight='bold')
ax3.legend(loc='upper left', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Format x-axis for all plots
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()

# Save figure
output_file = 'equity_curve_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved plot to {output_file}")

# Display summary statistics
print()
print("="*100)
print("SUMMARY STATISTICS")
print("="*100)
print()

baseline_final = baseline_equity[-1]
ladder_final = ladder_equity[-1]

baseline_return = (baseline_final - INITIAL_CAPITAL) / INITIAL_CAPITAL
ladder_return = (ladder_final - INITIAL_CAPITAL) / INITIAL_CAPITAL

baseline_max_dd = baseline_dd.min()
ladder_max_dd = ladder_dd.min()

print(f"{'Metric':<30} {'Baseline':>20} {'Optimal Ladder':>20} {'Improvement':>15}")
print("-" * 90)
print(f"{'Initial Capital':<30} ${INITIAL_CAPITAL:>19,} ${INITIAL_CAPITAL:>19,} {'':>15}")
print(f"{'Final Equity':<30} ${baseline_final:>19,.0f} ${ladder_final:>19,.0f} ${ladder_final - baseline_final:>14,.0f}")
print(f"{'Total Return':<30} {baseline_return:>19.1%} {ladder_return:>19.1%} {ladder_return - baseline_return:>14.1%}")
print(f"{'Max Drawdown':<30} {baseline_max_dd:>19.1%} {ladder_max_dd:>19.1%} {ladder_max_dd - baseline_max_dd:>14.1%}")
print(f"{'Return/DD Ratio':<30} {abs(baseline_return/baseline_max_dd*100):>19.2f} {abs(ladder_return/ladder_max_dd*100):>19.2f} {abs(ladder_return/ladder_max_dd*100) - abs(baseline_return/baseline_max_dd*100):>14.2f}")

print()
print("="*100)
print("Chart saved to equity_curve_comparison.png")
print("="*100)
