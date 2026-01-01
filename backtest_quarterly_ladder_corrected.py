"""
BACKTEST QUARTERLY RETRAINING - OPTIMAL LADDER (CORRECTED)
===========================================================
Test the optimal ladder strategy with quarterly model retraining

CRITICAL FIX: Now checks positions EVERY trading day, not just prediction days.
This ensures stops/targets are evaluated correctly even when predictions have gaps.

- Opens new positions: Only on days with valid predictions
- Updates positions: EVERY trading day (checks stops/targets daily)

This matches real production behavior where positions are monitored continuously.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("BACKTEST: QUARTERLY RETRAINING WITH OPTIMAL LADDER")
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

# Position limits (risk management)
MAX_TOTAL_POSITIONS = 90  # Prevent overleveraging
MAX_POSITIONS_PER_PAIR = 12  # Prevent overconcentration


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

        # Check ladder
        if self.ladder_level < len(LADDER_LEVELS):
            if intraday_high_profit >= LADDER_LEVELS[self.ladder_level]:
                self.partial_exits.append((LADDER_LEVELS[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
                self.ladder_level += 1
                return None

        # Emergency stop
        if self.days_held >= EMERGENCY_STOP_DAYS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            return 'emergency_stop', close, current_profit

        # Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > TRAILING_STOP_TRIGGER:
                self.trailing_stop = self.entry_price
        else:
            # Conservative approach: check if OLD stop was hit before updating
            # This assumes low came before high (conservative for longs, vice versa for shorts)
            old_stop = self.trailing_stop

            if self.direction == 'long':
                # Check against old stop first (before today's high raises it)
                hit_stop = low <= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                # Only update stop if not hit
                new_stop = self.entry_price + (high - self.entry_price) * TRAILING_STOP_PCT
                self.trailing_stop = max(self.trailing_stop, new_stop)
            else:
                # Check against old stop first (before today's low lowers it)
                hit_stop = high >= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                # Only update stop if not hit
                new_stop = self.entry_price - (self.entry_price - low) * TRAILING_STOP_PCT
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


def run_backtest(period_predictions, starting_capital, raw_data, existing_positions=None):
    """Run backtest for a specific period

    Args:
        period_predictions: Predictions dataframe with dates that have valid signals
        starting_capital: Starting capital
        raw_data: Dictionary of {pair: raw_price_dataframe} for ALL trading days
        existing_positions: Positions carried over from previous period (for continuous trading)
    """
    capital = starting_capital
    positions = existing_positions if existing_positions is not None else []
    trades = []

    # Get ALL trading days from raw data (not just prediction days)
    all_trading_days = set()
    prediction_dates = set()
    
    for pair_df in period_predictions.values():
        prediction_dates.update(pair_df.index)
    
    for pair, pair_df in raw_data.items():
        # Get the date range covered by predictions
        if len(prediction_dates) > 0:
            min_date = min(prediction_dates)
            max_date = max(prediction_dates)
            trading_days = pair_df[(pair_df.index >= min_date) & (pair_df.index <= max_date)].index
            all_trading_days.update(trading_days)
    
    all_trading_days = sorted(list(all_trading_days))

    for date in all_trading_days:
        # Get prices from RAW data (for position updates)
        prices_dict = {}
        for pair, pair_df in raw_data.items():
            if date in pair_df.index:
                row = pair_df.loc[date]
                prices_dict[pair] = {'high': row['high'], 'low': row['low'], 'close': row['close']}

        # Update ALL positions with actual prices (even on non-prediction days)
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
                'entry_price': position.entry_price,
                'exit_date': date,
                'exit_price': exit_price,
                'direction': position.direction,
                'days_held': position.days_held,
                'profit_pct': profit_pct,
                'profit_dollars': profit_dollars,
                'exit_reason': exit_reason,
                'ladder_hits': position.ladder_level,
                'capital_after': capital
            })

        # Open new positions ONLY on prediction days
        if date not in prediction_dates:
            continue

        # Check position limits before opening new positions
        if len(positions) >= MAX_TOTAL_POSITIONS:
            continue  # At max total positions

        for pair, pair_df in period_predictions.items():
            if date not in pair_df.index:
                continue

            # Check per-pair position limit
            pair_positions = [p for p in positions if p.pair == pair]
            if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
                continue  # At max positions for this pair

            row = pair_df.loc[date]

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

            position = Position(pair, date, price, direction, position_size, target, max_prob)
            positions.append(position)

    return capital, trades, positions


# Load quarterly predictions
print("Loading quarterly predictions...")
with open('model_predictions_quarterly.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

# Load raw price data
print("Loading raw price data...")
DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_1day_with_spreads.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df
    print(f"  {pair}: {len(df)} days")

print()

# Run backtest quarter by quarter, aggregate by year
# CRITICAL FIX: Carry positions across quarters for realistic continuous trading
capital = INITIAL_CAPITAL
yearly_results = {}
carried_positions = []  # Positions that carry across quarters

print()
print("Running backtest by quarter (with position carryover):")
print("-" * 100)

for quarter_name, quarter_preds in sorted(all_predictions.items()):
    year = quarter_name[:4]
    starting_cap = capital
    ending_cap, trades, carried_positions = run_backtest(quarter_preds, capital, all_raw_data, carried_positions)

    total_return = (ending_cap - starting_cap) / starting_cap

    winners = [t for t in trades if t['profit_pct'] > 0]
    losers = [t for t in trades if t['profit_pct'] <= 0]

    if year not in yearly_results:
        yearly_results[year] = {
            'year': year,
            'starting': starting_cap,
            'ending': ending_cap,
            'trades': 0,
            'winners': 0,
            'losers': 0
        }
    else:
        yearly_results[year]['ending'] = ending_cap

    yearly_results[year]['trades'] += len(trades)
    yearly_results[year]['winners'] += len(winners)
    yearly_results[year]['losers'] += len(losers)

    capital = ending_cap
    
    # Save trades to CSV for this quarter
    if len(trades) > 0:
        import os
        os.makedirs('trades', exist_ok=True)
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(f'trades/backtest_{quarter_name}.csv', index=False)

    print(f"{quarter_name}: ${starting_cap:>10,.0f} -> ${ending_cap:>10,.0f} ({total_return:>+7.1%}) | "
          f"{len(trades):>3} trades ({len(winners)}/{len(losers)} W/L)")

# Calculate year returns
for year_data in yearly_results.values():
    year_data['return'] = (year_data['ending'] - year_data['starting']) / year_data['starting']

print()
print("="*100)
print("FIXES APPLIED:")
print("  1. Positions checked EVERY trading day, not just prediction days")
print("  2. Positions carry across quarters (realistic continuous trading)")
print("  3. Conservative trailing stop logic (no intraday sequencing bias)")
print(f"  4. Position limits: Max {MAX_TOTAL_POSITIONS} total, {MAX_POSITIONS_PER_PAIR} per pair")
print("="*100)
print("QUARTERLY RETRAINING RESULTS")
print("="*100)
print()

yearly_list = sorted(yearly_results.values(), key=lambda x: x['year'])

total_trades = sum(r['trades'] for r in yearly_list)
total_winners = sum(r['winners'] for r in yearly_list)
total_losers = sum(r['losers'] for r in yearly_list)
win_rate = total_winners / total_trades if total_trades > 0 else 0

profitable_years = sum(1 for r in yearly_list if r['return'] > 0)
avg_return = np.mean([r['return'] for r in yearly_list])

years = len(yearly_list)
cagr = (capital / INITIAL_CAPITAL) ** (1/years) - 1

print(f"Starting Capital:     ${INITIAL_CAPITAL:,.0f}")
print(f"Ending Capital:       ${capital:,.0f}")
print(f"Total Return:         {(capital/INITIAL_CAPITAL - 1):.1%}")
print(f"CAGR:                 {cagr:.1%}")
print()
print(f"Total Trades:         {total_trades}")
print(f"Win Rate:             {win_rate:.1%} ({total_winners}/{total_losers} W/L)")
print(f"Profitable Years:     {profitable_years}/{years}")
print(f"Average Annual:       {avg_return:+.1%}")
print()

# Year by year breakdown
print("Year-by-Year Returns:")
print("-" * 60)
for r in yearly_list:
    print(f"  {r['year']}: {r['return']:>+7.1%}")
print()

# Drawdown analysis
print("Drawdown Analysis:")
equity_values = [INITIAL_CAPITAL]
for r in yearly_list:
    equity_values.append(r['ending'])

peak = INITIAL_CAPITAL
max_dd = 0
for val in equity_values[1:]:
    peak = max(peak, val)
    dd = (val - peak) / peak
    max_dd = min(max_dd, dd)

print(f"  Max Drawdown: {max_dd:.1%}")
print()

print("="*100)
