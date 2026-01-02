"""
ANALYZE LOSING TRADES
=====================
Identify patterns in losers to reduce drawdown:
- Confidence levels
- Pairs
- Time periods
- Market conditions
- Consecutive patterns
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("ANALYZING LOSING TRADES - FIND PATTERNS TO REDUCE DRAWDOWN")
print("="*100)
print()

# Load data
print("Loading data...")
with open('model_predictions_quarterly.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_1day_with_spreads.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

print("Data loaded")
print()

# Fixed parameters (optimized)
INITIAL_CAPITAL = 500
BASE_RISK_PER_TRADE = 0.007
MIN_CONFIDENCE = 0.65
MAX_TOTAL_POSITIONS = 90
MAX_POSITIONS_PER_PAIR = 12
EMERGENCY_STOP_DAYS = 15
EMERGENCY_STOP_LOSS_PCT = -0.04
TRAILING_STOP_TRIGGER = 0.005
TRAILING_STOP_PCT = 0.60
LADDER_LEVELS = [0.008, 0.015]
LADDER_SCALE_PCT = 0.40


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
            old_stop = self.trailing_stop

            if self.direction == 'long':
                hit_stop = low <= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price + (high - self.entry_price) * TRAILING_STOP_PCT
                self.trailing_stop = max(self.trailing_stop, new_stop)
            else:
                hit_stop = high >= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
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


# Run backtest and collect trade data
print("Running backtest to collect trade data...")
capital = INITIAL_CAPITAL
carried_positions = []
trades = []

for quarter_name, quarter_preds in sorted(all_predictions.items()):
    # Get trading days
    all_trading_days = set()
    prediction_dates = set()

    for pair_df in quarter_preds.values():
        prediction_dates.update(pair_df.index)

    for pair, pair_df in all_raw_data.items():
        if len(prediction_dates) > 0:
            min_date = min(prediction_dates)
            max_date = max(prediction_dates)
            trading_days = pair_df[(pair_df.index >= min_date) & (pair_df.index <= max_date)].index
            all_trading_days.update(trading_days)

    all_trading_days = sorted(list(all_trading_days))

    for date in all_trading_days:
        # Get prices
        prices_dict = {}
        for pair, pair_df in all_raw_data.items():
            if date in pair_df.index:
                row = pair_df.loc[date]
                prices_dict[pair] = {'high': row['high'], 'low': row['low'], 'close': row['close']}

        # Update positions
        positions_to_close = []
        for position in carried_positions:
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
            carried_positions.remove(position)

            trades.append({
                'pair': position.pair,
                'entry_date': position.entry_date,
                'exit_date': date,
                'direction': position.direction,
                'confidence': position.confidence,
                'days_held': position.days_held,
                'profit_pct': profit_pct,
                'exit_reason': exit_reason
            })

        # Open new positions
        if date not in prediction_dates:
            continue

        if len(carried_positions) >= MAX_TOTAL_POSITIONS:
            continue

        for pair, pair_df in quarter_preds.items():
            if date not in pair_df.index:
                continue

            pair_positions = [p for p in carried_positions if p.pair == pair]
            if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
                continue

            row = pair_df.loc[date]

            breakout_high_prob = row['breakout_high_prob']
            breakout_low_prob = row['breakout_low_prob']
            max_prob = max(breakout_high_prob, breakout_low_prob)

            if max_prob <= MIN_CONFIDENCE:
                continue

            assumed_risk_pct = 0.02
            risk_amount = capital * BASE_RISK_PER_TRADE
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
            carried_positions.append(position)

trades_df = pd.DataFrame(trades)
trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

print(f"Total trades: {len(trades_df):,}")
print()

# Separate winners and losers
winners = trades_df[trades_df['profit_pct'] > 0]
losers = trades_df[trades_df['profit_pct'] <= 0]

print(f"Winners: {len(winners):,} ({len(winners)/len(trades_df):.1%})")
print(f"Losers: {len(losers):,} ({len(losers)/len(trades_df):.1%})")
print()

# ANALYSIS 1: Confidence distribution
print("="*100)
print("LOSERS BY CONFIDENCE LEVEL")
print("="*100)
print()

conf_bins = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
conf_labels = ['65-70%', '70-75%', '75-80%', '80-85%', '85-90%', '90-95%', '95-100%']

trades_df['conf_bin'] = pd.cut(trades_df['confidence'], bins=conf_bins, labels=conf_labels)

print("Confidence | Total | Winners | Losers | Loss Rate | Avg Loss | Avg Win")
print("-" * 90)
for label in conf_labels:
    subset = trades_df[trades_df['conf_bin'] == label]
    if len(subset) > 0:
        subset_winners = subset[subset['profit_pct'] > 0]
        subset_losers = subset[subset['profit_pct'] <= 0]
        loss_rate = len(subset_losers) / len(subset)
        avg_loss = subset_losers['profit_pct'].mean() if len(subset_losers) > 0 else 0
        avg_win = subset_winners['profit_pct'].mean() if len(subset_winners) > 0 else 0

        print(f"{label:10s} | {len(subset):>5,} | {len(subset_winners):>7,} | {len(subset_losers):>6,} | {loss_rate:>8.1%} | {avg_loss:>+7.2%} | {avg_win:>+6.2%}")

print()

# ANALYSIS 2: Pairs
print("="*100)
print("LOSERS BY PAIR")
print("="*100)
print()

print("Pair    | Total | Winners | Losers | Loss Rate | Avg Loss")
print("-" * 70)
for pair in sorted(trades_df['pair'].unique()):
    subset = trades_df[trades_df['pair'] == pair]
    subset_winners = subset[subset['profit_pct'] > 0]
    subset_losers = subset[subset['profit_pct'] <= 0]
    loss_rate = len(subset_losers) / len(subset)
    avg_loss = subset_losers['profit_pct'].mean() if len(subset_losers) > 0 else 0

    print(f"{pair:7s} | {len(subset):>5,} | {len(subset_winners):>7,} | {len(subset_losers):>6,} | {loss_rate:>8.1%} | {avg_loss:>+7.2%}")

print()

# ANALYSIS 3: Time periods (years)
print("="*100)
print("LOSERS BY YEAR")
print("="*100)
print()

trades_df['year'] = trades_df['entry_date'].dt.year

print("Year | Total | Winners | Losers | Loss Rate | Avg Loss | Avg Win")
print("-" * 80)
for year in sorted(trades_df['year'].unique()):
    subset = trades_df[trades_df['year'] == year]
    subset_winners = subset[subset['profit_pct'] > 0]
    subset_losers = subset[subset['profit_pct'] <= 0]
    loss_rate = len(subset_losers) / len(subset)
    avg_loss = subset_losers['profit_pct'].mean() if len(subset_losers) > 0 else 0
    avg_win = subset_winners['profit_pct'].mean() if len(subset_winners) > 0 else 0

    print(f"{year} | {len(subset):>5,} | {len(subset_winners):>7,} | {len(subset_losers):>6,} | {loss_rate:>8.1%} | {avg_loss:>+7.2%} | {avg_win:>+6.2%}")

print()

# ANALYSIS 4: Exit reasons
print("="*100)
print("LOSERS BY EXIT REASON")
print("="*100)
print()

print("Exit Reason        | Total | Loss Rate | Avg Loss")
print("-" * 60)
for reason in trades_df['exit_reason'].unique():
    subset = trades_df[trades_df['exit_reason'] == reason]
    subset_losers = subset[subset['profit_pct'] <= 0]
    loss_rate = len(subset_losers) / len(subset) if len(subset) > 0 else 0
    avg_loss = subset_losers['profit_pct'].mean() if len(subset_losers) > 0 else 0

    print(f"{reason:18s} | {len(subset):>5,} | {loss_rate:>8.1%} | {avg_loss:>+7.2%}")

print()

# ANALYSIS 5: Direction
print("="*100)
print("LOSERS BY DIRECTION")
print("="*100)
print()

print("Direction | Total | Winners | Losers | Loss Rate | Avg Loss")
print("-" * 70)
for direction in ['long', 'short']:
    subset = trades_df[trades_df['direction'] == direction]
    subset_winners = subset[subset['profit_pct'] > 0]
    subset_losers = subset[subset['profit_pct'] <= 0]
    loss_rate = len(subset_losers) / len(subset)
    avg_loss = subset_losers['profit_pct'].mean() if len(subset_losers) > 0 else 0

    print(f"{direction:9s} | {len(subset):>5,} | {len(subset_winners):>7,} | {len(subset_losers):>6,} | {loss_rate:>8.1%} | {avg_loss:>+7.2%}")

print()

# ANALYSIS 6: Consecutive losers (streaks)
print("="*100)
print("LOSING STREAKS ANALYSIS")
print("="*100)
print()

# Sort by date
trades_sorted = trades_df.sort_values('exit_date')
trades_sorted['is_loser'] = trades_sorted['profit_pct'] <= 0

# Find streaks
streak = 0
max_streak = 0
streak_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
in_drawdown = False
dd_start_equity = 0
dd_start_idx = 0
max_dd = 0
max_dd_period = None

equity = 500
equity_curve = [500]

for idx, row in trades_sorted.iterrows():
    profit_dollars = row['profit_pct'] * (row.get('position_size', 1000))  # Approximate

    if row['is_loser']:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        if streak > 0:
            streak_key = min(streak, 5)
            streak_count[streak_key] = streak_count.get(streak_key, 0) + 1
        streak = 0

    equity += profit_dollars
    equity_curve.append(equity)

# Count streaks
print("Losing Streak Length | Count")
print("-" * 40)
for length in sorted(streak_count.keys()):
    count = streak_count[length]
    if count > 0:
        print(f"{length:19d} | {count:>5,}")

print(f"{'5+ (longest streaks)':19s} | {streak_count.get(5, 0):>5,}")
print(f"\nLongest losing streak: {max_streak} trades")
print()

# ANALYSIS 7: Worst losing periods
print("="*100)
print("WORST LOSING PERIODS (Drawdown Contributors)")
print("="*100)
print()

# Calculate rolling drawdown
trades_sorted['cumulative_profit'] = trades_sorted['profit_pct'].cumsum()
trades_sorted['peak_profit'] = trades_sorted['cumulative_profit'].expanding().max()
trades_sorted['drawdown'] = trades_sorted['cumulative_profit'] - trades_sorted['peak_profit']

# Find worst drawdown periods
worst_dd_idx = trades_sorted['drawdown'].idxmin()
worst_dd_value = trades_sorted.loc[worst_dd_idx, 'drawdown']
worst_dd_date = trades_sorted.loc[worst_dd_idx, 'exit_date']

# Find trades during worst drawdown
dd_trades = trades_sorted[
    (trades_sorted['exit_date'] >= worst_dd_date - pd.Timedelta(days=30)) &
    (trades_sorted['exit_date'] <= worst_dd_date + pd.Timedelta(days=30))
]

print(f"Worst drawdown: {worst_dd_value:.2%} on {worst_dd_date.date()}")
print(f"\nTrades around worst drawdown (±30 days):")
print(f"  Total: {len(dd_trades)}")
print(f"  Losers: {(dd_trades['profit_pct'] <= 0).sum()} ({(dd_trades['profit_pct'] <= 0).mean():.1%})")
print(f"  Avg loss: {dd_trades[dd_trades['profit_pct'] <= 0]['profit_pct'].mean():.2%}")
print()

print("Patterns during worst drawdown:")
print("-" * 60)
print(f"  Most common pairs: {dd_trades['pair'].value_counts().head(3).to_dict()}")
print(f"  Avg confidence: {dd_trades['confidence'].mean():.2%}")
print(f"  Most common exit reason: {dd_trades['exit_reason'].mode()[0]}")
print()

# ANALYSIS 8: Key insights
print("="*100)
print("KEY INSIGHTS - ACTIONABLE PATTERNS")
print("="*100)
print()

# Find worst confidence bucket
worst_conf = None
worst_conf_loss_rate = 0
for label in conf_labels:
    subset = trades_df[trades_df['conf_bin'] == label]
    if len(subset) > 100:  # Need meaningful sample size
        loss_rate = (subset['profit_pct'] <= 0).mean()
        if loss_rate > worst_conf_loss_rate:
            worst_conf_loss_rate = loss_rate
            worst_conf = label

# Find worst pair
worst_pair = None
worst_pair_loss_rate = 0
for pair in trades_df['pair'].unique():
    subset = trades_df[trades_df['pair'] == pair]
    loss_rate = (subset['profit_pct'] <= 0).mean()
    if loss_rate > worst_pair_loss_rate:
        worst_pair_loss_rate = loss_rate
        worst_pair = pair

# Find worst year
worst_year = None
worst_year_return = float('inf')
for year in trades_df['year'].unique():
    subset = trades_df[trades_df['year'] == year]
    total_return = subset['profit_pct'].sum()
    if total_return < worst_year_return:
        worst_year_return = total_return
        worst_year = year

print(f"1. WORST CONFIDENCE BUCKET: {worst_conf} with {worst_conf_loss_rate:.1%} loss rate")
print(f"   Action: Consider reducing exposure or avoiding {worst_conf} confidence trades")
print()

print(f"2. WORST PAIR: {worst_pair} with {worst_pair_loss_rate:.1%} loss rate")
print(f"   Action: Consider reducing position size on {worst_pair} or skipping it")
print()

print(f"3. WORST YEAR: {worst_year} with {worst_year_return:.2%} total return")
subset_worst_year = trades_df[trades_df['year'] == worst_year]
print(f"   Loss rate that year: {(subset_worst_year['profit_pct'] <= 0).mean():.1%}")
print(f"   Avg loser: {subset_worst_year[subset_worst_year['profit_pct'] <= 0]['profit_pct'].mean():.2%}")
print(f"   Action: Investigate what made {worst_year} different (market regime?)")
print()

print(f"4. EMERGENCY STOPS: {(trades_df['exit_reason'] == 'emergency_stop').sum():,} trades")
emergency_trades = trades_df[trades_df['exit_reason'] == 'emergency_stop']
print(f"   Average loss: {emergency_trades['profit_pct'].mean():.2%}")
print(f"   Percentage of all losses: {len(emergency_trades) / len(losers):.1%}")
print(f"   Action: These are the main drawdown contributors")
print()

# Find if low confidence correlates with emergency stops
low_conf_emergency = emergency_trades[emergency_trades['confidence'] < 0.75]
print(f"5. LOW CONFIDENCE EMERGENCY STOPS: {len(low_conf_emergency)} of {len(emergency_trades)} emergency stops")
print(f"   {len(low_conf_emergency) / len(emergency_trades):.1%} of emergency stops were low confidence (<75%)")
print(f"   Action: Reduce or skip trades with confidence <70%?")
print()

print("="*100)
print("DRAWDOWN REDUCTION STRATEGIES TO TEST")
print("="*100)
print()

print("Based on this analysis, test these filters:")
print("  1. Skip confidence <70% (if loss rate significantly higher)")
print("  2. Reduce size on worst-performing pair by 50%")
print(f"  3. Skip trades during historically bad periods (e.g., similar to {worst_year})")
print("  4. Reduce size after 2 consecutive losers (risk management)")
print("  5. Tighter stops for low confidence (<70%) trades")
print()
print("="*100)
