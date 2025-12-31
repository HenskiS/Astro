"""
Test if the API is returning the correct end_date
"""
import pandas as pd
from mock_broker_api import MockBrokerAPI

pair = 'USDCAD'
test_date = pd.Timestamp('2016-01-08')

print("="*100)
print(f"TESTING API DATE HANDLING FOR {pair}")
print("="*100)
print()

# Test get_history with end_date
api = MockBrokerAPI(data_dir='data')
history = api.get_history(pair, count=999999, end_date=test_date)

print(f"Requested end_date: {test_date}")
print(f"Last date in history: {history.index[-1]}")
print(f"Last 5 dates:")
for date in history.index[-5:]:
    print(f"  {date}")
print()

# Check if test_date is in the history
if test_date in history.index:
    print(f"[OK] {test_date} IS in history")
else:
    print(f"[ISSUE] {test_date} NOT in history!")

    # Check normalized dates
    normalized_dates = history.index.normalize()
    test_date_normalized = test_date.normalize()

    if test_date_normalized in normalized_dates:
        print(f"[INFO] But {test_date_normalized} IS in normalized dates")
        matching = history[history.index.normalize() == test_date_normalized]
        print(f"  Matching rows: {len(matching)}")
        if len(matching) > 0:
            print(f"  Matching date: {matching.index[0]}")
    else:
        print(f"[ERROR] {test_date_normalized} NOT even in normalized dates")

print()

# Load raw CSV to see what dates are available
df_raw = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw.set_index('date')

print(f"Dates around {test_date.date()} in raw CSV:")
nearby = df_raw[(df_raw.index.normalize() >= test_date.normalize() - pd.Timedelta(days=3)) &
                (df_raw.index.normalize() <= test_date.normalize() + pd.Timedelta(days=3))]
for date in nearby.index:
    print(f"  {date}")
print()
