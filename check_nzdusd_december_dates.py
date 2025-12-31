"""
Check if NZDUSD has all dates in late December 2015
"""
import pandas as pd
from mock_broker_api import MockBrokerAPI

pair = 'NZDUSD'
train_end = pd.Timestamp('2015-12-21')

print("="*100)
print(f"CHECKING {pair} DATES IN LATE DECEMBER 2015")
print("="*100)
print()

# Load full data
df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# Check dates around Dec 20-31
start_check = pd.Timestamp('2015-12-15')
end_check = pd.Timestamp('2016-01-05')

relevant_dates = df[(df.index >= start_check) & (df.index <= end_check)]

print(f"Dates from {start_check.date()} to {end_check.date()}:")
print("-"*100)
for date in relevant_dates.index:
    day_name = date.day_name()
    print(f"  {date.date()} ({day_name})")
print()

# Check what production API returns
api = MockBrokerAPI(data_dir='data')
data_end_date = train_end + pd.Timedelta(days=10)  # Dec 31
history = api.get_history(pair, count=999999, end_date=data_end_date)

print(f"API get_history(end_date={data_end_date.date()}) returns data through:")
print(f"  {history.index[-1]}")
print()

# Check dates Dec 20-31 in API result
dec_dates = history[(history.index >= pd.Timestamp('2015-12-20')) &
                    (history.index <= pd.Timestamp('2015-12-31'))]
print(f"Dates from Dec 20-31 in API result:")
for date in dec_dates.index:
    print(f"  {date.date()}")
print()

# Check if all dates needed for target calculation are present
target_date = pd.Timestamp('2015-12-20')
required_dates = pd.date_range(target_date, target_date + pd.Timedelta(days=10), freq='D')

print(f"To calculate target for Dec 20, we need Dec 20-30:")
missing = []
for req_date in required_dates:
    exists = any(d.normalize() == req_date.normalize() for d in dec_dates.index)
    status = "✓" if exists else "✗ MISSING"
    print(f"  {req_date.date()}: {status}")
    if not exists:
        missing.append(req_date)

if missing:
    print()
    print(f"[ISSUE] Missing {len(missing)} dates needed for target calculation!")
    for d in missing:
        print(f"  {d.date()} ({d.day_name()})")
print()
