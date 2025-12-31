"""
PLOT QUARTERLY BACKTEST EQUITY CURVE
=====================================
Plot equity curve from the quarterly backtest trade CSVs
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("PLOTTING QUARTERLY BACKTEST EQUITY CURVE")
print("="*100)
print()

INITIAL_CAPITAL = 500

# Load all trade files
print("Loading trade files...")
trade_files = sorted(glob.glob('trades/backtest_*.csv'))
all_trades = []

for file in trade_files:
    df = pd.read_csv(file)
    if len(df) > 0:
        all_trades.append(df)

if len(all_trades) == 0:
    print("ERROR: No trade files found!")
    exit(1)

trades_df = pd.concat(all_trades, ignore_index=True)
print(f"Loaded {len(trades_df)} trades from {len(trade_files)} quarters")

# Convert dates
trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

# Build equity curve from capital_after (which already includes all trades)
equity_df = pd.DataFrame({
    'date': trades_df['exit_date'],
    'equity': trades_df['capital_after']
})

# Sort and keep last value for each date (in case multiple trades on same day)
equity_df = equity_df.sort_values('date')
equity_df = equity_df.groupby('date', as_index=False).last()

# Prepend starting capital
start_date = equity_df['date'].iloc[0] - pd.Timedelta(days=1)
start_row = pd.DataFrame({'date': [start_date], 'equity': [INITIAL_CAPITAL]})
equity_df = pd.concat([start_row, equity_df], ignore_index=True)

print()
print(f"Equity curve spans: {equity_df['date'].min().date()} to {equity_df['date'].max().date()}")

# Calculate drawdown (as ratio, not percentage)
equity_df['peak'] = equity_df['equity'].expanding().max()
equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']

# Calculate statistics
final_equity = equity_df['equity'].iloc[-1]
total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
max_drawdown = equity_df['drawdown'].min()
years = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days / 365.25
cagr = (final_equity / INITIAL_CAPITAL) ** (1/years) - 1

print()
print("="*100)
print("PERFORMANCE SUMMARY")
print("="*100)
print(f"Starting Capital:    ${INITIAL_CAPITAL:,.0f}")
print(f"Ending Capital:      ${final_equity:,.0f}")
print(f"Total Return:        {total_return:.1%}")
print(f"CAGR:                {cagr:.1%}")
print(f"Max Drawdown:        {max_drawdown:.1%}")
print(f"Calmar Ratio:        {abs(cagr/max_drawdown*100):.2f}")
print(f"Total Trades:        {len(trades_df)}")
print(f"Win Rate:            {(trades_df['profit_pct'] > 0).sum() / len(trades_df):.1%}")
print("="*100)
print()

# Create plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f'Quarterly Backtest - Fixed Lookahead\nCAGR: {cagr:.1%} | Max DD: {max_drawdown:.1%} | Calmar: {abs(cagr/max_drawdown*100):.2f}',
             fontsize=16, fontweight='bold')

# Plot 1: Equity Curve (log scale)
ax1 = axes[0]
ax1.plot(equity_df['date'], equity_df['equity'], linewidth=2.5, color='#2ca02c', alpha=0.9, label='Equity')
ax1.set_ylabel('Equity ($, log scale)', fontsize=12, fontweight='bold')
ax1.set_title('Capital Growth with Compounding', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3, which='both')
ax1.axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Starting Capital')

# Add year markers
for year in range(2016, 2026):
    ax1.axvline(x=pd.Timestamp(f'{year}-01-01'), color='gray', linestyle=':', alpha=0.3, linewidth=1)

# Format y-axis
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K' if x < 1000000 else f'${x/1000000:.1f}M'))

# Plot 2: Drawdown
ax2 = axes[1]
drawdown_pct = equity_df['drawdown'] * 100  # Convert to percentage for plotting
ax2.plot(equity_df['date'], drawdown_pct, linewidth=2.5, color='#d62728', alpha=0.9, label='Drawdown')
ax2.fill_between(equity_df['date'], drawdown_pct, 0, alpha=0.3, color='#d62728')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_title('Drawdown from Peak', fontsize=13, fontweight='bold')
ax2.legend(loc='lower left', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# Add year markers
for year in range(2016, 2026):
    ax2.axvline(x=pd.Timestamp(f'{year}-01-01'), color='gray', linestyle=':', alpha=0.3, linewidth=1)

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()

# Save figure
output_file = 'quarterly_backtest_equity.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Chart saved to {output_file}")
print()
