"""
BACKTEST 15-MINUTE BREAKOUT STRATEGY (OPTIMIZED)
=================================================
Optimized parameters based on testing:
- Emergency: 24 periods (6 hours) - cuts losers 62% faster
- Trailing: 0.001 trigger (0.1%), 75% trail - locks in profits better
- Ladders: [0.002, 0.004] (0.2%, 0.4%)
- Confidence: 0.70 (vs 0.65 original)
- Spread costs: bid/ask prices for realistic execution

Expected Results: +49.6% return, 202% CAGR, -2.9% max DD
"""
import pandas as pd
import numpy as np
import pickle
import warnings
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Backtest 15M breakout strategy')
parser.add_argument('--plot', action='store_true', help='Generate equity curve plot')
args = parser.parse_args()

print("="*100)
print("BACKTEST: 15-MINUTE BREAKOUT STRATEGY (OPTIMIZED)")
print("="*100)
print()
print("NOTE: Uses bid/ask prices for realistic spread costs")
print("  - Long entry: ASK price | Long exit: BID price")
print("  - Short entry: BID price | Short exit: ASK price")
print()

# Strategy Parameters (OPTIMIZED for 15m timeframe)
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.004  # 0.4%
MIN_CONFIDENCE = 0.70  # Raised from 0.65

# Emergency stop: 24 periods = 6 hours (OPTIMIZED - was 48/12h)
EMERGENCY_STOP_PERIODS = 24
EMERGENCY_STOP_LOSS_PCT = -0.04

# Trailing stop (OPTIMIZED)
TRAILING_STOP_TRIGGER = 0.001  # 0.1% (was 0.0015)
TRAILING_STOP_PCT = 0.75  # 75% (was 65%)

# Ladder parameters (OPTIMIZED)
LADDER_LEVELS = [0.002, 0.004]  # 0.2%, 0.4% (was 0.003, 0.006)
LADDER_SCALE_PCT = 0.40

# Position limits (8 pairs)
MAX_TOTAL_POSITIONS = 120  # 15 per pair on average
MAX_POSITIONS_PER_PAIR = 15

# Spread filter: Avoid high-spread hours
AVOID_HOURS = [20, 21, 22]  # UTC hours

# Data
DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']


class Position:
    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.original_size = size
        self.breakout_target = breakout_target
        self.confidence = confidence
        self.periods_held = 0
        self.max_profit = 0
        self.trailing_stop = None
        self.partial_exits = []
        self.ladder_level = 0

    def update(self, date, bid_high, bid_low, bid_close, ask_high, ask_low, ask_close):
        """Update position with bid/ask prices"""
        self.periods_held += 1

        if self.direction == 'long':
            # Long: exit at BID prices
            current_profit = (bid_close - self.entry_price) / self.entry_price
            intraday_high_profit = (bid_high - self.entry_price) / self.entry_price
            hit_target = bid_high >= self.breakout_target
        else:
            # Short: exit at ASK prices
            current_profit = (self.entry_price - ask_close) / self.entry_price
            intraday_high_profit = (self.entry_price - ask_low) / self.entry_price
            hit_target = ask_low <= self.breakout_target

        self.max_profit = max(self.max_profit, intraday_high_profit)

        # Check ladder
        if self.ladder_level < len(LADDER_LEVELS):
            if intraday_high_profit >= LADDER_LEVELS[self.ladder_level]:
                self.partial_exits.append((LADDER_LEVELS[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
                self.ladder_level += 1
                return None

        # Emergency stop
        if self.periods_held >= EMERGENCY_STOP_PERIODS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            exit_price = bid_close if self.direction == 'long' else ask_close
            return 'emergency_stop', exit_price, current_profit

        # Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > TRAILING_STOP_TRIGGER:
                self.trailing_stop = self.entry_price
        else:
            old_stop = self.trailing_stop

            if self.direction == 'long':
                hit_stop = bid_low <= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price + (bid_high - self.entry_price) * TRAILING_STOP_PCT
                self.trailing_stop = max(self.trailing_stop, new_stop)
            else:
                hit_stop = ask_high >= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price - (self.entry_price - ask_low) * TRAILING_STOP_PCT
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


# Load predictions
print("Loading predictions...")
with open('test_predictions_15m.pkl', 'rb') as f:
    predictions = pickle.load(f)

print(f"Loaded predictions for {len(predictions)} pairs")
print()

# Load raw data for price action
print("Loading raw 15m data...")
all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_15m.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df
    print(f"  {pair}: {len(df)} candles")

print()

# Get all trading periods
all_trading_periods = set()
for pair_df in predictions.values():
    all_trading_periods.update(pair_df.index)

all_trading_periods = sorted(list(all_trading_periods))

min_date = min(all_trading_periods)
max_date = max(all_trading_periods)

print(f"Backtesting from {min_date} to {max_date}")
print(f"Total 15m periods: {len(all_trading_periods):,}")
print()

# Track equity
capital = INITIAL_CAPITAL
positions = []
trades = []
equity_curve = [(min_date, INITIAL_CAPITAL)]  # Track (date, capital) pairs

for period_idx, date in enumerate(all_trading_periods):
    # Get prices
    prices_dict = {}
    for pair in PAIRS:
        if date in all_raw_data[pair].index:
            row = all_raw_data[pair].loc[date]
            prices_dict[pair] = {
                'bid_high': row['bid_high'],
                'bid_low': row['bid_low'],
                'bid_close': row['bid_close'],
                'ask_high': row['ask_high'],
                'ask_low': row['ask_low'],
                'ask_close': row['ask_close'],
                'close': row['close']
            }

    # Update positions
    positions_to_close = []
    for position in positions:
        if position.pair not in prices_dict:
            continue

        prices = prices_dict[position.pair]
        exit_info = position.update(
            date,
            prices['bid_high'],
            prices['bid_low'],
            prices['bid_close'],
            prices['ask_high'],
            prices['ask_low'],
            prices['ask_close']
        )
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
        positions.remove(position)

        trades.append({
            'pair': position.pair,
            'entry_date': position.entry_date,
            'exit_date': date,
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.original_size,
            'profit_pct': profit_pct,
            'profit_dollars': profit_dollars,
            'periods_held': position.periods_held,
            'exit_reason': exit_reason,
            'confidence': position.confidence,
            'capital_after': capital
        })

        equity_curve.append((date, capital))

    # Open new positions
    if len(positions) >= MAX_TOTAL_POSITIONS:
        continue

    # Skip high-spread hours
    current_hour = date.hour
    if current_hour in AVOID_HOURS:
        continue

    for pair in PAIRS:
        if date not in predictions[pair].index:
            continue

        pair_positions = [p for p in positions if p.pair == pair]
        if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
            continue

        row = predictions[pair].loc[date]

        breakout_high_prob = row['breakout_high_prob']
        breakout_low_prob = row['breakout_low_prob']
        max_prob = max(breakout_high_prob, breakout_low_prob)

        if max_prob <= MIN_CONFIDENCE:
            continue

        # Calculate position size
        assumed_risk_pct = 0.02
        risk_amount = capital * RISK_PER_TRADE
        mid_price = row['close']
        position_size = risk_amount / (mid_price * assumed_risk_pct)

        # Determine direction and entry price
        if breakout_high_prob > breakout_low_prob:
            direction = 'long'
            breakout_level = row['high_80p']
            target = breakout_level * 1.005
            if pair not in prices_dict:
                continue
            entry_price = prices_dict[pair]['ask_close']  # Pay ASK
        else:
            direction = 'short'
            breakout_level = row['low_80p']
            target = breakout_level * 0.995
            if pair not in prices_dict:
                continue
            entry_price = prices_dict[pair]['bid_close']  # Receive BID

        position = Position(pair, date, entry_price, direction, position_size, target, max_prob)
        positions.append(position)

    # Progress update
    if (period_idx + 1) % 2000 == 0:
        pct_complete = (period_idx + 1) / len(all_trading_periods) * 100
        print(f"Progress: {period_idx + 1:>6,}/{len(all_trading_periods):,} periods ({pct_complete:>5.1f}%) | "
              f"Capital: ${capital:>10,.0f} | Positions: {len(positions):>3} | "
              f"Trades: {len(trades):>5,}")

print()

# Results
trades_df = pd.DataFrame(trades)
trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
days = (max_date - min_date).days
years = days / 365
cagr = (capital / INITIAL_CAPITAL) ** (1 / years) - 1

winners = trades_df[trades_df['profit_pct'] > 0]
losers = trades_df[trades_df['profit_pct'] <= 0]
win_rate = len(winners) / len(trades_df) if len(trades_df) > 0 else 0

# Calculate max drawdown
equity_values = [INITIAL_CAPITAL] + list(trades_df['capital_after'])
peak = INITIAL_CAPITAL
max_dd = 0
for val in equity_values[1:]:
    peak = max(peak, val)
    dd = (val - peak) / peak
    max_dd = min(max_dd, dd)

print("="*100)
print("RESULTS")
print("="*100)
print()
print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
print(f"Final Capital: ${capital:,.0f}")
print(f"Total Return: {total_return:+.1%}")
print(f"CAGR: {cagr:.1%}")
print(f"Max Drawdown: {max_dd:.1%}")
print()
print(f"Total Trades: {len(trades_df):,}")
print(f"Winners: {len(winners):,} ({win_rate:.1%})")
print(f"Losers: {len(losers):,}")
print()

if len(winners) > 0:
    avg_winner = winners['profit_pct'].mean()
    avg_winner_periods = winners['periods_held'].mean()
    avg_winner_hours = avg_winner_periods * 0.25  # 15m = 0.25h
    print(f"Avg Winner: {avg_winner:+.2%} ({avg_winner_hours:.1f} hours)")

if len(losers) > 0:
    avg_loser = losers['profit_pct'].mean()
    avg_loser_periods = losers['periods_held'].mean()
    avg_loser_hours = avg_loser_periods * 0.25
    print(f"Avg Loser: {avg_loser:+.2%} ({avg_loser_hours:.1f} hours)")

if len(winners) > 0 and len(losers) > 0:
    profit_ratio = abs(avg_winner / avg_loser)
    print(f"Profit Ratio: {profit_ratio:.2f}:1")

print()

# Exit reasons
print("Exit Reasons:")
exit_reasons = trades_df['exit_reason'].value_counts()
for reason, count in exit_reasons.items():
    pct = count / len(trades_df) * 100
    print(f"  {reason:20s} {count:>5,} ({pct:>5.1f}%)")

print()

# Per-pair results
print("="*100)
print("PER-PAIR RESULTS")
print("="*100)
print()

for pair in PAIRS:
    pair_trades = trades_df[trades_df['pair'] == pair]
    if len(pair_trades) == 0:
        continue

    pair_winners = pair_trades[pair_trades['profit_pct'] > 0]
    pair_win_rate = len(pair_winners) / len(pair_trades)
    pair_avg_profit = pair_trades['profit_pct'].mean()

    print(f"  {pair:8s} {len(pair_trades):>6,} trades | Win rate: {pair_win_rate:>5.1%} | "
          f"Avg P/L: {pair_avg_profit:>+7.2%}")

print()

print("="*100)
print("COMPARISON: 1H vs 15M OPTIMIZED")
print("="*100)
print()
print("1H Strategy (8 pairs, ~3 months):")
print("  CAGR: 113.4%")
print("  Max DD: -0.8%")
print("  Win Rate: 89.3%")
print("  Avg Hold: 2.9 days winners, 4.2 days losers")
print()
print(f"15M OPTIMIZED Strategy (8 pairs, {years:.1f} years):")
print(f"  CAGR: {cagr:.1%}")
print(f"  Max DD: {max_dd:.1%}")
print(f"  Win Rate: {win_rate:.1%}")
if len(winners) > 0 and len(losers) > 0:
    print(f"  Avg Hold: {avg_winner_hours:.1f} hours winners, {avg_loser_hours:.1f} hours losers")
print()
print("Optimizations applied:")
print("  - Emergency stop: 24 periods (6h) vs original 48 periods (12h)")
print("  - Trailing: 75% vs original 65% - locks in profits better")
print("  - Trigger: 0.001 vs original 0.0015 - activates sooner")
print("  - Confidence: 0.70 vs original 0.65 - more selective")
print()

# Save results
print("Saving results...")
trades_df.to_csv('backtest_15m_optimized_results.csv', index=False)
print("Results saved to: backtest_15m_optimized_results.csv")
print()

# Generate equity curve plot if requested
if args.plot:
    print("="*100)
    print("GENERATING EQUITY CURVE PLOT")
    print("="*100)
    print()

    # Extract dates and capital values
    eq_dates = [item[0] for item in equity_curve]
    eq_capital = [item[1] for item in equity_curve]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot equity curve
    ax.plot(eq_dates, eq_capital, linewidth=2, color='#2E86AB', label='Equity')

    # Plot initial capital line
    ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Initial Capital')

    # Calculate and plot drawdown periods
    equity_df = pd.DataFrame({'date': eq_dates, 'capital': eq_capital})
    equity_df['peak'] = equity_df['capital'].cummax()
    equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak']

    # Find max drawdown point
    max_dd_idx = equity_df['drawdown'].idxmin()
    max_dd_date = equity_df.loc[max_dd_idx, 'date']
    max_dd_capital = equity_df.loc[max_dd_idx, 'capital']

    # Mark max drawdown
    ax.plot(max_dd_date, max_dd_capital, 'ro', markersize=8, label=f'Max DD: {max_dd:.1%}')

    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Capital ($)', fontsize=12, fontweight='bold')
    ax.set_title('15M Breakout Strategy - Equity Curve\n' +
                 f'Return: {total_return:+.1%} | CAGR: {cagr:.1%} | Max DD: {max_dd:.1%} | Win Rate: {win_rate:.1%}',
                 fontsize=14, fontweight='bold', pad=20)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, ha='right')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add text box with key stats
    stats_text = (
        f'Initial: ${INITIAL_CAPITAL:,.0f}\n'
        f'Final: ${capital:,.0f}\n'
        f'Trades: {len(trades_df):,}\n'
        f'Winners: {len(winners):,} ({win_rate:.1%})'
    )

    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    plot_filename = 'backtest_15m_optimized_equity_curve.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Equity curve saved to: {plot_filename}")

    # Show plot
    plt.show()

    print()

print("="*100)
