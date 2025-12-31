"""
COMPOUNDING EQUITY CURVE - REALISTIC SCENARIO
==============================================
Show equity curve with continuous compounding across validation periods.

Walk-forward structure (models retrain):
- 2016-2017: Train on 2010-2015, test on 2016-2017
- 2018-2019: Train on 2010-2017, test on 2018-2019
- 2020-2021: Train on 2010-2019, test on 2020-2021
- 2022-2023: Train on 2010-2021, test on 2022-2023
- 2024-2025: Train on 2010-2023, test on 2024-2025

But capital compounds continuously (no resets).
"""
import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

print("="*100)
print("COMPOUNDING EQUITY CURVE - REALISTIC SCENARIO")
print("="*100)
print()

# Strategy Parameters
INITIAL_CAPITAL = 500
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


def run_period_backtest(period_predictions, starting_capital, use_ladder=False):
    """Run backtest for one period with specified starting capital"""
    capital = starting_capital
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

    return equity_dates, equity_curve, capital


# Load predictions
print("Loading predictions...")
with open('model_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

# Run compounding backtests
print("Running baseline with compounding...")
baseline_dates = []
baseline_equity = []
baseline_capital = INITIAL_CAPITAL

for period_name, period_preds in all_predictions.items():
    print(f"  {period_name}: Starting capital ${baseline_capital:,.0f}")
    dates, equity, ending_capital = run_period_backtest(period_preds, baseline_capital, use_ladder=False)
    baseline_dates.extend(dates)
    baseline_equity.extend(equity)
    baseline_capital = ending_capital  # Compound for next period

print()
print("Running optimal ladder with compounding...")
ladder_dates = []
ladder_equity = []
ladder_capital = INITIAL_CAPITAL

for period_name, period_preds in all_predictions.items():
    print(f"  {period_name}: Starting capital ${ladder_capital:,.0f}")
    dates, equity, ending_capital = run_period_backtest(period_preds, ladder_capital, use_ladder=True)
    ladder_dates.extend(dates)
    ladder_equity.extend(equity)
    ladder_capital = ending_capital  # Compound for next period

print()
print("Creating plots...")

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Compounding Equity Curve: Baseline vs Optimal Ladder', fontsize=16, fontweight='bold')

# Plot 1: Equity Curves (log scale for better visualization)
ax1 = axes[0]
ax1.plot(baseline_dates, baseline_equity, label='Baseline', linewidth=2.5, color='#1f77b4', alpha=0.9)
ax1.plot(ladder_dates, ladder_equity, label='Optimal Ladder (0.8%, 1.5%)', linewidth=2.5, color='#2ca02c', alpha=0.9)
ax1.set_ylabel('Equity ($, log scale)', fontsize=12, fontweight='bold')
ax1.set_title('Compounding Capital Growth', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3, which='both')
ax1.axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Starting Capital')

# Add period markers
period_starts = ['2016-01-01', '2018-01-01', '2020-01-01', '2022-01-01', '2024-01-01']
for start in period_starts:
    ax1.axvline(x=pd.Timestamp(start), color='gray', linestyle=':', alpha=0.4, linewidth=1)

# Format y-axis
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K' if x < 1000000 else f'${x/1000000:.1f}M'))

# Plot 2: Drawdown
ax2 = axes[1]

baseline_series = pd.Series(baseline_equity, index=baseline_dates)
baseline_peak = baseline_series.expanding().max()
baseline_dd = (baseline_series - baseline_peak) / baseline_peak * 100

ladder_series = pd.Series(ladder_equity, index=ladder_dates)
ladder_peak = ladder_series.expanding().max()
ladder_dd = (ladder_series - ladder_peak) / ladder_peak * 100

ax2.plot(baseline_dates, baseline_dd, label='Baseline DD', linewidth=2.5, color='#1f77b4', alpha=0.9)
ax2.plot(ladder_dates, ladder_dd, label='Optimal Ladder DD', linewidth=2.5, color='#2ca02c', alpha=0.9)
ax2.fill_between(baseline_dates, baseline_dd, 0, alpha=0.2, color='#1f77b4')
ax2.fill_between(ladder_dates, ladder_dd, 0, alpha=0.2, color='#2ca02c')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_title('Drawdown from Peak', fontsize=13, fontweight='bold')
ax2.legend(loc='lower left', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# Add period markers
for start in period_starts:
    ax2.axvline(x=pd.Timestamp(start), color='gray', linestyle=':', alpha=0.4, linewidth=1)

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()

# Save figure
output_file = 'compounding_equity_curve.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved plot to {output_file}")

# Summary statistics
print()
print("="*100)
print("COMPOUNDING PERFORMANCE SUMMARY")
print("="*100)
print()

baseline_final = baseline_equity[-1]
ladder_final = ladder_equity[-1]
baseline_total_return = (baseline_final - INITIAL_CAPITAL) / INITIAL_CAPITAL
ladder_total_return = (ladder_final - INITIAL_CAPITAL) / INITIAL_CAPITAL

baseline_max_dd = baseline_dd.min()
ladder_max_dd = ladder_dd.min()

# Annualized returns (10 years)
years = 10
baseline_cagr = (baseline_final / INITIAL_CAPITAL) ** (1/years) - 1
ladder_cagr = (ladder_final / INITIAL_CAPITAL) ** (1/years) - 1

print(f"{'Metric':<35} {'Baseline':>20} {'Optimal Ladder':>20} {'Difference':>18}")
print("-" * 98)
print(f"{'Starting Capital':<35} ${INITIAL_CAPITAL:>19,} ${INITIAL_CAPITAL:>19,} {'':>18}")
print(f"{'Final Capital':<35} ${baseline_final:>19,.0f} ${ladder_final:>19,.0f} ${ladder_final - baseline_final:>17,.0f}")
print(f"{'Total Return (10 years)':<35} {baseline_total_return:>19.1%} {ladder_total_return:>19.1%} {ladder_total_return - baseline_total_return:>17.1%}")
print(f"{'CAGR (annualized)':<35} {baseline_cagr:>19.1%} {ladder_cagr:>19.1%} {ladder_cagr - baseline_cagr:>17.1%}")
print(f"{'Max Drawdown':<35} {baseline_max_dd:>19.1%} {ladder_max_dd:>19.1%} {ladder_max_dd - baseline_max_dd:>17.1%}")
print(f"{'Calmar Ratio (CAGR/MaxDD)':<35} {abs(baseline_cagr/baseline_max_dd*100):>19.2f} {abs(ladder_cagr/ladder_max_dd*100):>19.2f} {abs(ladder_cagr/ladder_max_dd*100) - abs(baseline_cagr/baseline_max_dd*100):>17.2f}")

print()
print(f"Starting with $100,000 in 2016:")
print(f"  Baseline strategy:      ${baseline_final:>12,.0f}  ({baseline_total_return:>6.1%} total)")
print(f"  Optimal Ladder strategy: ${ladder_final:>12,.0f}  ({ladder_total_return:>6.1%} total)")
print(f"  Advantage:              ${ladder_final - baseline_final:>12,.0f}  ({ladder_total_return - baseline_total_return:>6.1%} extra)")

print()
print("="*100)
print("REALISTIC COMPOUNDING CHART SAVED")
print("="*100)
print()
print("Note: Vertical dotted lines show when models retrain (start of each 2-year period)")
print("Capital compounds continuously across all periods (realistic scenario)")
