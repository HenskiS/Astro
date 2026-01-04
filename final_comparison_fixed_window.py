"""
FINAL COMPARISON: Including Fixed-Window Walk-Forward
======================================================
Compares all approaches including the new fixed-window walk-forward
that uses the SAME training data amount as the original.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.004
MIN_CONFIDENCE = 0.70
EMERGENCY_STOP_PERIODS = 24
EMERGENCY_STOP_LOSS_PCT = -0.04
TRAILING_STOP_TRIGGER = 0.001
TRAILING_STOP_PCT = 0.75
LADDER_LEVELS = [0.002, 0.004]
LADDER_SCALE_PCT = 0.40
MAX_TOTAL_POSITIONS = 120
MAX_POSITIONS_PER_PAIR = 15
FIFO_MODE = 'skip_competing'
AVOID_HOURS = [20, 21, 22]
SLIPPAGE_PCT = 0.0001
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
        self.periods_held += 1
        if self.direction == 'long':
            current_profit = (bid_close - self.entry_price) / self.entry_price
            intraday_high_profit = (bid_high - self.entry_price) / self.entry_price
            hit_target = bid_high >= self.breakout_target
        else:
            current_profit = (self.entry_price - ask_close) / self.entry_price
            intraday_high_profit = (self.entry_price - ask_low) / self.entry_price
            hit_target = ask_low <= self.breakout_target

        self.max_profit = max(self.max_profit, intraday_high_profit)

        if self.ladder_level < len(LADDER_LEVELS):
            if intraday_high_profit >= LADDER_LEVELS[self.ladder_level]:
                self.partial_exits.append((LADDER_LEVELS[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
                self.ladder_level += 1
                return None

        if self.periods_held >= EMERGENCY_STOP_PERIODS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            exit_price = bid_close if self.direction == 'long' else ask_close
            return 'emergency_stop', exit_price, current_profit

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


def run_backtest(predictions, raw_data, name):
    """Run backtest"""
    all_trading_periods = set()
    for pair_df in predictions.values():
        all_trading_periods.update(pair_df.index)
    all_trading_periods = sorted(list(all_trading_periods))

    if len(all_trading_periods) == 0:
        return None

    min_date = min(all_trading_periods)
    max_date = max(all_trading_periods)

    capital = INITIAL_CAPITAL
    positions = []
    trades = []
    pending_signals = []

    for date in all_trading_periods:
        prices_dict = {}
        for pair in PAIRS:
            if date in raw_data[pair].index:
                row = raw_data[pair].loc[date]
                prices_dict[pair] = {
                    'bid_open': row['bid_open'], 'bid_high': row['bid_high'],
                    'bid_low': row['bid_low'], 'bid_close': row['bid_close'],
                    'ask_open': row['ask_open'], 'ask_high': row['ask_high'],
                    'ask_low': row['ask_low'], 'ask_close': row['ask_close'],
                    'close': row['close']
                }

        # Process pending signals
        signals_to_keep = []
        for signal in pending_signals:
            if signal['pair'] not in prices_dict:
                signals_to_keep.append(signal)
                continue
            if len(positions) >= MAX_TOTAL_POSITIONS:
                continue
            if len([p for p in positions if p.pair == signal['pair']]) >= MAX_POSITIONS_PER_PAIR:
                continue

            prices = prices_dict[signal['pair']]
            entry_price = prices['ask_open'] if signal['direction'] == 'long' else prices['bid_open']
            if SLIPPAGE_PCT > 0:
                entry_price *= (1 + SLIPPAGE_PCT) if signal['direction'] == 'long' else (1 - SLIPPAGE_PCT)

            position = Position(signal['pair'], date, entry_price, signal['direction'],
                              signal['size'], signal['target'], signal['confidence'])
            positions.append(position)

            exit_info = position.update(date, prices['bid_high'], prices['bid_low'], prices['bid_close'],
                                       prices['ask_high'], prices['ask_low'], prices['ask_close'])
            if exit_info:
                exit_reason, exit_price, _ = exit_info
                raw_profit = (exit_price - position.entry_price) / position.entry_price if position.direction == 'long' else (position.entry_price - exit_price) / position.entry_price
                profit_pct = position.calculate_blended_profit(raw_profit)
                profit_dollars = profit_pct * (position.original_size * position.entry_price)
                capital += profit_dollars
                positions.remove(position)
                trades.append({'profit_pct': profit_pct, 'profit_dollars': profit_dollars})

        pending_signals = signals_to_keep

        # Update positions
        positions_to_close = []
        for position in positions:
            if position.pair not in prices_dict:
                continue
            prices = prices_dict[position.pair]
            exit_info = position.update(date, prices['bid_high'], prices['bid_low'], prices['bid_close'],
                                       prices['ask_high'], prices['ask_low'], prices['ask_close'])
            if exit_info:
                positions_to_close.append((position, exit_info))

        for position, exit_info in positions_to_close:
            exit_reason, exit_price, _ = exit_info
            raw_profit = (exit_price - position.entry_price) / position.entry_price if position.direction == 'long' else (position.entry_price - exit_price) / position.entry_price
            profit_pct = position.calculate_blended_profit(raw_profit)
            profit_dollars = profit_pct * (position.original_size * position.entry_price)
            capital += profit_dollars
            positions.remove(position)
            trades.append({'profit_pct': profit_pct, 'profit_dollars': profit_dollars})

        # Generate signals
        if len(positions) >= MAX_TOTAL_POSITIONS or date.hour in AVOID_HOURS:
            continue

        for pair in PAIRS:
            if date not in predictions[pair].index:
                continue
            if len([p for p in positions if p.pair == pair]) >= MAX_POSITIONS_PER_PAIR:
                continue

            row = predictions[pair].loc[date]
            max_prob = max(row['breakout_high_prob'], row['breakout_low_prob'])
            if max_prob <= MIN_CONFIDENCE:
                continue

            if row['breakout_high_prob'] > row['breakout_low_prob']:
                direction = 'long'
                target = row['high_80p'] * 1.005
            else:
                direction = 'short'
                target = row['low_80p'] * 0.995

            pair_positions = [p for p in positions if p.pair == pair]
            if FIFO_MODE == 'skip_competing' and len(pair_positions) > 0:
                if direction not in set(p.direction for p in pair_positions):
                    continue

            risk_amount = capital * RISK_PER_TRADE
            position_size = risk_amount / (row['close'] * 0.02)

            pending_signals.append({
                'pair': pair, 'direction': direction, 'size': position_size,
                'target': target, 'confidence': max_prob
            })

    # Calculate results
    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        return None

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    days = (max_date - min_date).days
    years = max(days / 365, 0.01)
    cagr = (capital / INITIAL_CAPITAL) ** (1 / years) - 1

    winners = trades_df[trades_df['profit_pct'] > 0]
    losers = trades_df[trades_df['profit_pct'] <= 0]
    win_rate = len(winners) / len(trades_df)

    equity_values = [INITIAL_CAPITAL] + list(trades_df['profit_dollars'].cumsum() + INITIAL_CAPITAL)
    peak = INITIAL_CAPITAL
    max_dd = 0
    for val in equity_values[1:]:
        peak = max(peak, val)
        dd = (val - peak) / peak
        max_dd = min(max_dd, dd)

    return {
        'name': name,
        'period': f"{min_date.date()} to {max_date.date()}",
        'days': days,
        'final_capital': capital,
        'cagr': cagr,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'total_trades': len(trades_df)
    }


print("="*100)
print("COMPREHENSIVE BACKTEST COMPARISON")
print("="*100)
print()

# Load raw data
print("Loading raw data...")
all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

# Load all prediction sets
print("Loading predictions...")
with open('test_predictions_15m.pkl', 'rb') as f:
    original_preds = pickle.load(f)
    print("  [OK] Original")

with open('test_predictions_15m_walkforward_fixed.pkl', 'rb') as f:
    wf_fixed_preds = pickle.load(f)
    print("  [OK] Walk-forward (fixed window)")

print()
print("="*100)
print("RUNNING BACKTESTS")
print("="*100)
print()

results = []

# Original
print("1. ORIGINAL (70/30 split, train once)")
print("-" * 100)
r = run_backtest(original_preds, all_raw_data, "Original")
if r:
    print(f"  Period: {r['period']} ({r['days']} days)")
    print(f"  CAGR: {r['cagr']:.1%} | Max DD: {r['max_dd']:.1%} | Win Rate: {r['win_rate']:.1%} | Trades: {r['total_trades']:,}")
    results.append(r)
print()

# Walk-forward fixed
print("2. WALK-FORWARD (fixed 10-month training window, retrain every 2 weeks)")
print("-" * 100)
r = run_backtest(wf_fixed_preds, all_raw_data, "Walk-Forward Fixed")
if r:
    print(f"  Period: {r['period']} ({r['days']} days)")
    print(f"  CAGR: {r['cagr']:.1%} | Max DD: {r['max_dd']:.1%} | Win Rate: {r['win_rate']:.1%} | Trades: {r['total_trades']:,}")
    results.append(r)
print()

# Summary
print("="*100)
print("SUMMARY")
print("="*100)
print()

if len(results) == 2:
    print(f"{'Method':<50} {'Period (days)':<20} {'CAGR':<12} {'Max DD':<12} {'Trades':<10}")
    print("-" * 104)
    for r in results:
        print(f"{r['name']:<50} {r['days']:>15}  {r['cagr']:>10.1%}  {r['max_dd']:>10.1%}  {r['total_trades']:>8,}")

    print()
    print("="*100)
    print("ANALYSIS")
    print("="*100)
    print()

    orig = results[0]
    wf_fixed = results[1]

    # Check overlap
    print("Comparing the two approaches:")
    print(f"  Original: {orig['period']}")
    print(f"  Walk-forward fixed: {wf_fixed['period']}")
    print()

    if wf_fixed['days'] < orig['days']:
        print("  Note: Walk-forward covers a SHORTER period (starts later)")
        print("  This is expected - need 10 months of data before first trade")
    print()

    cagr_diff = abs(orig['cagr'] - wf_fixed['cagr'])
    print(f"  CAGR difference: {cagr_diff:.1%}")
    print()

    if cagr_diff < 0.30:
        print("[GOOD] CAGRs are reasonably similar (<30% difference)")
        print("  Your original results are realistic!")
        print("  The walk-forward confirms the strategy works with periodic retraining.")
    else:
        print("[NOTABLE] CAGRs differ by >30%")
        print("  Possible reasons:")
        print("  - Different test periods (original may have caught favorable period)")
        print("  - Walk-forward tests MORE periods (more representative)")
        print("  - Model performance varies with market regimes")

    print()
    print("RECOMMENDATION:")
    print(f"  For production expectations, use: {wf_fixed['cagr']:.1%} CAGR")
    print("  This accounts for periodic retraining with same training data amount.")

print()
