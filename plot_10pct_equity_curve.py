"""
PLOT 10% SIZING EQUITY CURVE
==============================
Visualize the equity growth with 10% position sizing
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
POSITION_PCT = 0.10


def strategy_with_equity(predictions, raw_data):
    """Trail after 1x target hit with equity tracking"""
    all_dates = sorted(set([d for p in predictions.values() for d in p.index]))

    capital = INITIAL_CAPITAL
    positions = []
    equity_curve = [{'date': all_dates[0], 'capital': INITIAL_CAPITAL}]
    trades = []  # Track all closed trades

    for date in all_dates:
        # Update existing positions
        positions_to_close = []
        for pos in positions:
            pair = pos['pair']
            if date not in raw_data[pair].index:
                continue

            row = raw_data[pair].loc[date]
            pos['bars_held'] += 1

            # Emergency stop loss (5% from entry)
            if pos['direction'] == 'long':
                emergency_stop = pos['entry_price'] * 0.95
                if row['bid_low'] <= emergency_stop:
                    exit_price = emergency_stop
                    profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                    positions_to_close.append((pos, profit_pct, 'emergency_stop'))
                    continue
            else:  # short
                emergency_stop = pos['entry_price'] * 1.05
                if row['ask_high'] >= emergency_stop:
                    exit_price = emergency_stop
                    profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                    positions_to_close.append((pos, profit_pct, 'emergency_stop'))
                    continue

            if pos['direction'] == 'long':
                if not pos.get('trailing_active') and row['bid_high'] >= pos['initial_target']:
                    pos['trailing_active'] = True
                    pos['trailing_stop'] = pos['initial_target']
                    pos['peak_price'] = row['bid_high']
                    continue

                if pos.get('trailing_active'):
                    if row['bid_high'] > pos['peak_price']:
                        pos['peak_price'] = row['bid_high']
                        new_stop = pos['initial_target'] + 0.75 * (pos['peak_price'] - pos['initial_target'])
                        pos['trailing_stop'] = max(pos['trailing_stop'], new_stop)

                    if row['bid_low'] <= pos['trailing_stop']:
                        exit_price = pos['trailing_stop']
                        profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                        positions_to_close.append((pos, profit_pct, 'trailing_stop'))
                        continue

            else:
                if not pos.get('trailing_active') and row['ask_low'] <= pos['initial_target']:
                    pos['trailing_active'] = True
                    pos['trailing_stop'] = pos['initial_target']
                    pos['peak_price'] = row['ask_low']
                    continue

                if pos.get('trailing_active'):
                    if row['ask_low'] < pos['peak_price']:
                        pos['peak_price'] = row['ask_low']
                        new_stop = pos['initial_target'] - 0.75 * (pos['initial_target'] - pos['peak_price'])
                        pos['trailing_stop'] = min(pos['trailing_stop'], new_stop)

                    if row['ask_high'] >= pos['trailing_stop']:
                        exit_price = pos['trailing_stop']
                        profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                        positions_to_close.append((pos, profit_pct, 'trailing_stop'))
                        continue

            if pos['bars_held'] >= 24:
                if pos['direction'] == 'long':
                    exit_price = row['bid_close']
                    profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                else:
                    exit_price = row['ask_close']
                    profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                positions_to_close.append((pos, profit_pct, 'time_exit'))

        # Close positions
        for pos, profit_pct, exit_reason in positions_to_close:
            profit_dollars = profit_pct * pos['size']
            capital += profit_dollars
            positions.remove(pos)

            # Record trade
            trades.append({
                'pair': pos['pair'],
                'direction': pos['direction'],
                'entry_date': pos.get('entry_date', None),
                'exit_date': date,
                'bars_held': pos['bars_held'],
                'profit_pct': profit_pct,
                'profit_dollars': profit_dollars,
                'exit_reason': exit_reason,
                'confidence': pos['confidence']
            })

        # Look for new signals
        for pair in PAIRS:
            if date not in predictions[pair].index:
                continue
            if date not in raw_data[pair].index:
                continue

            pred = predictions[pair].loc[date]
            max_prob = max(pred['breakout_high_prob'], pred['breakout_low_prob'])

            if max_prob <= 0.80:
                continue

            row = raw_data[pair].loc[date]
            if pred['breakout_high_prob'] > pred['breakout_low_prob']:
                direction = 'long'
                entry_price = row['ask_open']
                initial_target = pred['high_80p']
            else:
                direction = 'short'
                entry_price = row['bid_open']
                initial_target = pred['low_80p']

            position_size = capital * POSITION_PCT

            positions.append({
                'pair': pair,
                'direction': direction,
                'entry_price': entry_price,
                'entry_date': date,
                'initial_target': initial_target,
                'size': position_size,
                'bars_held': 0,
                'confidence': max_prob,
                'trailing_active': False
            })

        # Record equity (capital + unrealized P&L from open positions)
        unrealized_pnl = 0
        for pos in positions:
            if date in raw_data[pos['pair']].index:
                row = raw_data[pos['pair']].loc[date]
                if pos['direction'] == 'long':
                    current_price = row['bid_close']
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                else:
                    current_price = row['ask_close']
                    pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']
                unrealized_pnl += pnl_pct * pos['size']

        total_equity = capital + unrealized_pnl
        equity_curve.append({'date': date, 'capital': total_equity})

    return pd.DataFrame(equity_curve), capital, pd.DataFrame(trades)


# Load data
print("Loading data...")
all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

# Load continuous predictions (no gaps!)
print("Loading continuous predictions...")
with open('test_predictions_15m_continuous.pkl', 'rb') as f:
    combined_preds = pickle.load(f)

for pair in PAIRS:
    if pair in combined_preds:
        print(f"  {pair}: {len(combined_preds[pair]):,} predictions ({combined_preds[pair].index.min().date()} to {combined_preds[pair].index.max().date()})")

# Run ONE continuous backtest with all predictions
print("\nRunning continuous backtest...")
combined_equity, final_capital, trades_df = strategy_with_equity(combined_preds, all_raw_data)
print(f"Final Capital: ${final_capital:.2f}")

# Print trade statistics
print("\n" + "="*100)
print("TRADE STATISTICS")
print("="*100)
print()

if len(trades_df) > 0:
    winners = trades_df[trades_df['profit_pct'] > 0]
    losers = trades_df[trades_df['profit_pct'] <= 0]

    print(f"Total trades: {len(trades_df):,}")
    print(f"Winners: {len(winners):,} ({100*len(winners)/len(trades_df):.1f}%)")
    print(f"Losers:  {len(losers):,} ({100*len(losers)/len(trades_df):.1f}%)")
    print()

    print(f"Average win:  {winners['profit_pct'].mean():.2%} (${winners['profit_dollars'].mean():.2f})")
    print(f"Average loss: {losers['profit_pct'].mean():.2%} (${losers['profit_dollars'].mean():.2f})")
    print(f"Largest win:  {winners['profit_pct'].max():.2%} (${winners['profit_dollars'].max():.2f})")
    print(f"Largest loss: {losers['profit_pct'].min():.2%} (${losers['profit_dollars'].min():.2f})")
    print()

    print(f"Average bars held: {trades_df['bars_held'].mean():.1f}")
    print()

    # Exit reasons
    print("Exit reasons:")
    for reason, count in trades_df['exit_reason'].value_counts().items():
        print(f"  {reason}: {count:,} ({100*count/len(trades_df):.1f}%)")
    print()

    # Profit factor
    total_wins = winners['profit_dollars'].sum()
    total_losses = abs(losers['profit_dollars'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Expectancy per trade: ${trades_df['profit_dollars'].mean():.2f}")
    print()
else:
    print("No trades executed")
    print()

# Calculate drawdown
combined_equity['peak'] = combined_equity['capital'].cummax()
combined_equity['drawdown'] = (combined_equity['capital'] - combined_equity['peak']) / combined_equity['peak']

# Calculate statistics for title
total_return = (final_capital / INITIAL_CAPITAL - 1)
max_dd = combined_equity['drawdown'].min()
date_range = (combined_equity['date'].max() - combined_equity['date'].min()).days / 365.25
annual_return = (1 + total_return) ** (1 / date_range) - 1 if date_range > 0 else 0

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Equity curve
ax1.plot(combined_equity['date'], combined_equity['capital'], linewidth=1.5, color='#2ECC71', label='10% Sizing Strategy')
ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Initial Capital')
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Capital ($)', fontsize=12, fontweight='bold')
ax1.set_title(f'Trailing Stop Strategy - 10% Position Sizing\n{annual_return:.1%} Avg Annual Return, {max_dd:.1%} Max DD',
             fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 2: Drawdown
ax2.fill_between(combined_equity['date'], combined_equity['drawdown'], 0, color='#E74C3C', alpha=0.3)
ax2.plot(combined_equity['date'], combined_equity['drawdown'], linewidth=1, color='#E74C3C')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Drawdown', fontsize=12, fontweight='bold')
ax2.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

# Format x-axis
for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('equity_curve_10pct_sizing.png', dpi=300, bbox_inches='tight')
print("\nEquity curve saved: equity_curve_10pct_sizing.png")

# Statistics
print("\n" + "="*100)
print("STATISTICS")
print("="*100)
print()
print(f"Initial Capital:      ${INITIAL_CAPITAL:,.2f}")
print(f"Final Capital:        ${combined_equity['capital'].iloc[-1]:,.2f}")
print(f"Total Return:         {total_return:.1%}")
print(f"Annual Return (CAGR): {annual_return:.1%}")
print(f"Max Drawdown:         {max_dd:.2%}")
print(f"Peak Capital:         ${combined_equity['peak'].max():,.2f}")
print(f"Time Period:          {combined_equity['date'].min().date()} to {combined_equity['date'].max().date()}")
print(f"Duration:             {date_range:.1f} years")
print()

# Year by year breakdown
print("YEAR BY YEAR PERFORMANCE")
print("-" * 100)
combined_equity['year'] = pd.to_datetime(combined_equity['date']).dt.year
yearly_stats = []
for year in sorted(combined_equity['year'].unique()):
    year_data = combined_equity[combined_equity['year'] == year]
    year_start_capital = year_data['capital'].iloc[0]
    year_end_capital = year_data['capital'].iloc[-1]
    year_return = (year_end_capital / year_start_capital - 1)
    year_max_dd = year_data['drawdown'].min()
    yearly_stats.append({
        'year': year,
        'start': year_start_capital,
        'end': year_end_capital,
        'return': year_return,
        'max_dd': year_max_dd
    })
    print(f"{year}: ${year_start_capital:>7.2f} -> ${year_end_capital:>7.2f} ({year_return:>+6.1%}) | Max DD: {year_max_dd:>6.2%}")
print()

plt.show()
