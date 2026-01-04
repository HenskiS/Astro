"""
VERIFY BACKTEST REALISM
=======================
Check for potential sources of bias:
1. Trade statistics (win rate, avg win/loss)
2. Position stacking (max concurrent positions)
3. Drawdown calculation accuracy
4. Entry/exit price realism
5. Trade frequency and size
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
INITIAL_CAPITAL = 500
POSITION_PCT = 0.10

print("="*100)
print("VERIFYING BACKTEST REALISM")
print("="*100)
print()

# Load data
all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

# Load predictions
with open('test_predictions_15m_continuous.pkl', 'rb') as f:
    predictions = pickle.load(f)

print("Running backtest with detailed tracking...")
print()

# Strategy with detailed trade tracking
def strategy_with_tracking(predictions, raw_data):
    """Trail after 1x target hit with detailed logging"""
    all_dates = sorted(set([d for p in predictions.values() for d in p.index]))

    capital = INITIAL_CAPITAL
    positions = []
    equity_curve = [{'date': all_dates[0], 'capital': INITIAL_CAPITAL}]
    trades = []
    max_concurrent = 0

    for date in all_dates:
        # Track max concurrent positions
        max_concurrent = max(max_concurrent, len(positions))

        # Update existing positions
        positions_to_close = []
        for pos in positions:
            pair = pos['pair']
            if date not in raw_data[pair].index:
                continue

            row = raw_data[pair].loc[date]
            pos['bars_held'] += 1

            if pos['direction'] == 'long':
                # Check if target hit
                if not pos.get('trailing_active') and row['bid_high'] >= pos['initial_target']:
                    pos['trailing_active'] = True
                    pos['trailing_stop'] = pos['initial_target']
                    pos['peak_price'] = row['bid_high']
                    continue

                # Trail stop
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

            else:  # short
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

            # Time exit
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

            trades.append({
                'entry_date': pos['entry_date'],
                'exit_date': date,
                'pair': pos['pair'],
                'direction': pos['direction'],
                'entry_price': pos['entry_price'],
                'size': pos['size'],
                'profit_pct': profit_pct,
                'profit_dollars': profit_dollars,
                'bars_held': pos['bars_held'],
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

        # Record equity (capital + unrealized P&L)
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

    return pd.DataFrame(equity_curve), capital, pd.DataFrame(trades), max_concurrent

equity_df, final_capital, trades_df, max_concurrent = strategy_with_tracking(predictions, all_raw_data)

print("="*100)
print("TRADE STATISTICS")
print("="*100)
print()

print(f"Total trades: {len(trades_df)}")
print(f"Max concurrent positions: {max_concurrent}")
print()

if len(trades_df) > 0:
    winners = trades_df[trades_df['profit_pct'] > 0]
    losers = trades_df[trades_df['profit_pct'] <= 0]

    print(f"Winners: {len(winners)} ({100*len(winners)/len(trades_df):.1f}%)")
    print(f"Losers:  {len(losers)} ({100*len(losers)/len(trades_df):.1f}%)")
    print()

    print(f"Avg win:  {winners['profit_pct'].mean():.2%} (${winners['profit_dollars'].mean():.2f})")
    print(f"Avg loss: {losers['profit_pct'].mean():.2%} (${losers['profit_dollars'].mean():.2f})")
    print(f"Largest win:  {winners['profit_pct'].max():.2%} (${winners['profit_dollars'].max():.2f})")
    print(f"Largest loss: {losers['profit_pct'].min():.2%} (${losers['profit_dollars'].min():.2f})")
    print()

    print(f"Avg bars held: {trades_df['bars_held'].mean():.1f}")
    print(f"Avg position size: ${trades_df['size'].mean():.2f}")
    print()

    # Exit reasons
    print("Exit reasons:")
    for reason, count in trades_df['exit_reason'].value_counts().items():
        print(f"  {reason}: {count} ({100*count/len(trades_df):.1f}%)")
    print()

    # Profit factor
    total_wins = winners['profit_dollars'].sum()
    total_losses = abs(losers['profit_dollars'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    print(f"Profit factor: {profit_factor:.2f}")
    print()

print("="*100)
print("DRAWDOWN ANALYSIS")
print("="*100)
print()

equity_df['peak'] = equity_df['capital'].cummax()
equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak']

print(f"Max drawdown: {equity_df['drawdown'].min():.2%}")
print(f"Max drawdown duration: {equity_df[equity_df['drawdown'] < 0]['drawdown'].count()} periods")

# Find drawdown periods
in_dd = False
dd_periods = []
dd_start = None
for idx, row in equity_df.iterrows():
    if row['drawdown'] < -0.001 and not in_dd:  # Start of DD
        in_dd = True
        dd_start = row['date']
    elif row['drawdown'] >= -0.001 and in_dd:  # End of DD
        in_dd = False
        if dd_start is not None:
            dd_periods.append({
                'start': dd_start,
                'end': row['date'],
                'duration': (row['date'] - dd_start).days
            })

print(f"\nDrawdown periods > 0.1%: {len(dd_periods)}")
if len(dd_periods) > 0:
    print(f"Longest drawdown: {max([p['duration'] for p in dd_periods])} days")
print()

print("="*100)
print("POSITION STACKING ANALYSIS")
print("="*100)
print()

# Analyze concurrent positions over time
concurrent_positions = []
for date in sorted(set([d for p in predictions.values() for d in p.index])):
    positions_open = len([t for _, t in trades_df.iterrows()
                         if t['entry_date'] <= date <= t['exit_date']])
    if positions_open > 0:
        concurrent_positions.append(positions_open)

if concurrent_positions:
    print(f"Max concurrent positions: {max(concurrent_positions)}")
    print(f"Avg concurrent positions: {np.mean(concurrent_positions):.2f}")
    print(f"Median concurrent positions: {np.median(concurrent_positions):.0f}")
    print()

print("="*100)
print("SUSPICIOUS PATTERN CHECKS")
print("="*100)
print()

# Check 1: Are we making money on almost every trade?
if len(trades_df) > 0:
    win_rate = len(winners) / len(trades_df)
    if win_rate > 0.80:
        print(f"[WARNING] Win rate is {win_rate:.1%} - suspiciously high!")
    else:
        print(f"[OK] Win rate is {win_rate:.1%} - reasonable")

# Check 2: Is drawdown too low?
max_dd = abs(equity_df['drawdown'].min())
if max_dd < 0.01:
    print(f"[WARNING] Max drawdown is {max_dd:.2%} - suspiciously low!")
    print("         Even the best strategies have drawdowns > 5%")
else:
    print(f"[OK] Max drawdown is {max_dd:.2%}")

# Check 3: Are we taking too many concurrent positions?
if max_concurrent > 8:
    print(f"[WARNING] Max concurrent positions is {max_concurrent}")
    print("         This could amplify returns unrealistically")
else:
    print(f"[OK] Max concurrent positions is {max_concurrent}")

# Check 4: Sharpe ratio check
returns = equity_df['capital'].pct_change().dropna()
sharpe = returns.mean() / returns.std() * np.sqrt(252 * 96)  # 15-min bars
if sharpe > 5:
    print(f"[WARNING] Sharpe ratio is {sharpe:.2f} - suspiciously high!")
else:
    print(f"[OK] Sharpe ratio is {sharpe:.2f}")

print()
print("="*100)
print("FINAL VERDICT")
print("="*100)
print()

suspicious_count = 0
if len(trades_df) > 0:
    if len(winners) / len(trades_df) > 0.80:
        suspicious_count += 1
if abs(equity_df['drawdown'].min()) < 0.01:
    suspicious_count += 1
if max_concurrent > 8:
    suspicious_count += 1
if sharpe > 5:
    suspicious_count += 1

if suspicious_count == 0:
    print("[OK] No major red flags detected")
elif suspicious_count <= 2:
    print("[CAUTION] Some suspicious patterns detected")
    print("         Review the strategy carefully")
else:
    print("[CRITICAL] Multiple red flags detected!")
    print("           Likely source of bias in the backtest")

print()
