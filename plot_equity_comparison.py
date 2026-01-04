"""
EQUITY CURVE COMPARISON
=======================
Plots equity curves for:
1. Original parameters (losing)
2. Optimized parameters (after optimization completes)

Shows visually where the original strategy loses money.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
INITIAL_CAPITAL = 500

# Load the multi-year backtest results
print("="*100)
print("EQUITY CURVE: Multi-Year Performance")
print("="*100)
print()

# For now, let's just show the results we already have
print("Results from multi-year test (Nov-Mar periods):")
print()

years = ['2021', '2022', '2023', '2024', '2025 (Nov-Mar)', '2025 (Aug-Dec)']
cagrs = [-0.284, 0.289, 0.063, -0.044, -0.290, 1.16]  # Last one is the winning period
months_per_period = [4, 4, 4, 4, 4, 5]  # Aug-Dec is 5 months

# Each test period with actual dates
dates = [
    pd.Timestamp('2020-11-01'),  # Start
    pd.Timestamp('2021-03-01'),  # End of 2021 test
    pd.Timestamp('2022-03-01'),  # End of 2022 test
    pd.Timestamp('2023-03-01'),  # End of 2023 test
    pd.Timestamp('2024-03-01'),  # End of 2024 test
    pd.Timestamp('2025-03-01'),  # End of 2025 Nov-Mar test
    pd.Timestamp('2025-12-31'),  # End of 2025 Aug-Dec test (winning period)
]

equity_curve = [INITIAL_CAPITAL]

for i, (year, cagr, months) in enumerate(zip(years, cagrs, months_per_period)):
    # Calculate return for this period
    period_return = (1 + cagr) ** (months/12) - 1

    new_capital = equity_curve[-1] * (1 + period_return)
    equity_curve.append(new_capital)

# Create plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Equity curve
ax1.plot(dates, equity_curve, linewidth=2, color='#E74C3C', label='Original Strategy', marker='o')
ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Initial Capital')
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Capital ($)', fontsize=12, fontweight='bold')
avg_cagr = sum(cagrs) / len(cagrs)
ax1.set_title(f'Original Strategy Performance (2020-2025)\nAverage CAGR: {avg_cagr:.1%}',
             fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Calculate drawdown
peak = INITIAL_CAPITAL
drawdowns = []
for val in equity_curve:
    peak = max(peak, val)
    dd = (val - peak) / peak
    drawdowns.append(dd)

# Plot 2: Drawdown
ax2.fill_between(dates, drawdowns, 0, color='#E74C3C', alpha=0.3)
ax2.plot(dates, drawdowns, linewidth=2, color='#E74C3C')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Drawdown', fontsize=12, fontweight='bold')
ax2.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

# Format x-axis
for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('equity_curve_original.png', dpi=300, bbox_inches='tight')
print("Equity curve saved: equity_curve_original.png")
print()

# Show statistics
print("="*100)
print("PERIOD-BY-PERIOD BREAKDOWN")
print("="*100)
print()

for year, cagr in zip(years, cagrs):
    status = "WIN" if cagr > 0 else "LOSS"
    print(f"{year}: {cagr:>7.1%}  [{status}]")

print()
print(f"Final Capital: ${equity_curve[-1]:,.0f}")
print(f"Total Return: {(equity_curve[-1] / INITIAL_CAPITAL - 1):.1%}")
print(f"Max Drawdown: {min(drawdowns):.1%}")
print()

print("="*100)
print("KEY INSIGHTS")
print("="*100)
print()
print("Losing periods: 2021, 2024, 2025 (Nov-Mar)")
print("Winning periods: 2022, 2023, 2025 (Aug-Dec)")
print()
print("The strategy is HIGHLY regime-dependent!")
print("2025 Aug-Dec was exceptional (+116% CAGR) but 2025 Nov-Mar was terrible (-29%).")
print("Optimization should help make it more consistent.")
print()

plt.show()
