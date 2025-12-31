"""
Find which rows have different target_breakout_high values
"""
import pandas as pd
from production_simulation import calculate_features, create_targets
from mock_broker_api import MockBrokerAPI

pair = 'USDCAD'
train_start = pd.Timestamp('2010-01-01')
train_end = pd.Timestamp('2015-12-21')

print("="*100)
print(f"FINDING TARGET DIFFERENCES FOR {pair}")
print("="*100)
print()

# BACKTEST
df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

df_with_features = calculate_features(df.copy())
df_with_targets = create_targets(df_with_features)
train_df_backtest = df_with_targets[(df_with_targets.index >= train_start) &
                                     (df_with_targets.index <= train_end)].copy()
train_df_backtest = train_df_backtest.dropna()

# PRODUCTION
api = MockBrokerAPI(data_dir='data')
training_data_prod = api.get_history(pair, count=999999, end_date=train_end)
training_data_prod = calculate_features(training_data_prod)
training_data_prod = create_targets(training_data_prod)
training_data_prod = training_data_prod[(training_data_prod.index >= train_start) &
                                         (training_data_prod.index <= train_end)]
training_data_prod = training_data_prod.dropna()

print(f"Backtest HIGH targets: {train_df_backtest['target_breakout_high'].sum()}")
print(f"Production HIGH targets: {training_data_prod['target_breakout_high'].sum()}")
print()

# Find common dates
common_dates = train_df_backtest.index.intersection(training_data_prod.index)
print(f"Common dates: {len(common_dates)} out of {len(train_df_backtest)}")
print()

if len(common_dates) == len(train_df_backtest):
    # Compare target values
    back_targets = train_df_backtest.loc[common_dates, 'target_breakout_high']
    prod_targets = training_data_prod.loc[common_dates, 'target_breakout_high']

    differences = back_targets != prod_targets
    num_diffs = differences.sum()

    print(f"Rows with different HIGH targets: {num_diffs}")
    print()

    if num_diffs > 0:
        diff_dates = common_dates[differences]
        print("First 10 differences:")
        print("-"*100)
        for i, date in enumerate(diff_dates[:10]):
            back_val = back_targets.loc[date]
            prod_val = prod_targets.loc[date]
            back_row = train_df_backtest.loc[date]
            prod_row = training_data_prod.loc[date]

            print(f"{i+1}. {date.date()}:")
            print(f"   Backtest: target={back_val}, high={back_row['high']:.5f}, high_20d={back_row['high_20d']:.5f}")
            print(f"   Production: target={prod_val}, high={prod_row['high']:.5f}, high_20d={prod_row['high_20d']:.5f}")

            # Check if the underlying data matches
            if abs(back_row['high'] - prod_row['high']) < 0.0001:
                print(f"   -> OHLC data matches, but targets differ!")
            else:
                print(f"   -> OHLC data DIFFERS!")
            print()
else:
    print("Dates don't match between backtest and production!")
    print(f"Backtest dates: {len(train_df_backtest)}")
    print(f"Production dates: {len(training_data_prod)}")
    print(f"Common: {len(common_dates)}")

print()
