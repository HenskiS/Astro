"""
Debug why Jan 8 features might be NaN
"""
import pandas as pd
from production_simulation import calculate_features
from mock_broker_api import MockBrokerAPI

pair = 'USDCAD'
test_date = pd.Timestamp('2016-01-08')

print("="*100)
print(f"DEBUGGING JAN 8 FEATURES FOR {pair}")
print("="*100)
print()

# Use the API (with the fix)
api = MockBrokerAPI(data_dir='data')
history = api.get_history(pair, count=999999, end_date=test_date)

print(f"Requested: {test_date}")
print(f"Last date in history: {history.index[-1]}")
print(f"History length: {len(history)}")
print()

# Calculate features
features_before_drop = calculate_features(history)
print(f"After calculate_features, last date: {features_before_drop.index[-1]}")
print(f"Length before dropna: {len(features_before_drop)}")

# Check last row for NaN
last_row = features_before_drop.iloc[-1]
nan_features = last_row[last_row.isna()].index.tolist()
print(f"NaN features in last row: {len(nan_features)}")
if nan_features:
    print(f"  {nan_features[:10]}")  # Show first 10
print()

# Drop NaN
features_after_drop = features_before_drop.dropna()
print(f"After dropna, last date: {features_after_drop.index[-1]}")
print(f"Length after dropna: {len(features_after_drop)}")
print()

if features_after_drop.index[-1].date() != test_date.date():
    print(f"[ISSUE] Last date {features_after_drop.index[-1].date()} != requested {test_date.date()}")
    print(f"Jan 8 was dropped by dropna!")
else:
    print(f"[OK] Jan 8 is present after dropna")
print()
