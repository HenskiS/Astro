"""
FETCH 5-MINUTE DATA FROM OANDA
===============================
Fetches 5-minute candle data for ultra-ultra high-frequency breakout trading.

Since Oanda limits to 5000 candles per request (17 days of 5m data),
we'll fetch multiple batches to get ~3-4 months of data.

USAGE:
  python fetch_5m_data.py              # Fetch data (skip if files exist)
  python fetch_5m_data.py --force      # Re-fetch all data
  python fetch_5m_data.py --analyze    # Fetch and analyze spreads by hour
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parse arguments
parser = argparse.ArgumentParser(description='Fetch 5-minute data from Oanda')
parser.add_argument('--force', action='store_true', help='Re-fetch data even if files exist')
parser.add_argument('--analyze', action='store_true', help='Analyze spreads by hour after fetching')
args = parser.parse_args()

# Oanda API configuration
OANDA_API_KEY = os.getenv('OANDA_API_KEY', '')
OANDA_ACCOUNT_TYPE = os.getenv('OANDA_ACCOUNT_TYPE', 'practice')

if OANDA_ACCOUNT_TYPE == 'practice':
    OANDA_API_URL = 'https://api-fxpractice.oanda.com'
else:
    OANDA_API_URL = 'https://api-fxtrade.oanda.com'

# Pairs to fetch (all 8 pairs)
PAIRS = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CAD', 'USD_CHF', 'NZD_USD', 'EUR_JPY']

# Fetch parameters
COUNT_PER_REQUEST = 5000
NUM_REQUESTS = 6  # 6 * 5000 * 5m = 30,000 candles = ~104 days

print("="*80)
print("FETCHING 5-MINUTE DATA FROM OANDA")
print("="*80)
print()

if not OANDA_API_KEY:
    print("ERROR: OANDA_API_KEY not set!")
    exit(1)

print(f"API URL: {OANDA_API_URL}")
print(f"Account Type: {OANDA_ACCOUNT_TYPE}")
print(f"Fetching {COUNT_PER_REQUEST * NUM_REQUESTS:,} 5-minute candles per pair (~{COUNT_PER_REQUEST * NUM_REQUESTS * 5 / 1440:.0f} days)")
print()

# Create data directory
os.makedirs('data_5m', exist_ok=True)

all_data_frames = {}

for pair in PAIRS:
    pair_name = pair.replace('_', '')
    output_file = f'data_5m/{pair_name}_5m.csv'

    # Skip if exists
    if os.path.exists(output_file) and not args.force:
        print(f"{pair}: File exists, skipping (use --force to re-fetch)")
        if args.analyze:
            df = pd.read_csv(output_file)
            df['date'] = pd.to_datetime(df['date'])
            all_data_frames[pair_name] = df
        continue

    print(f"Fetching {pair} ({NUM_REQUESTS} batches)...")

    url = f"{OANDA_API_URL}/v3/instruments/{pair}/candles"
    headers = {
        'Authorization': f'Bearer {OANDA_API_KEY}',
        'Content-Type': 'application/json'
    }

    all_rows = []
    to_time = None

    for batch_num in range(NUM_REQUESTS):
        params = {
            'granularity': 'M5',  # 5-minute candles
            'count': COUNT_PER_REQUEST,
            'price': 'MBA'
        }

        if to_time:
            params['to'] = to_time

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            candles = data['candles']

            batch_rows = []
            for candle in candles:
                if not candle['complete']:
                    continue

                timestamp = pd.to_datetime(candle['time'])

                batch_rows.append({
                    'date': timestamp,
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume']),
                    'bid_open': float(candle['bid']['o']),
                    'bid_high': float(candle['bid']['h']),
                    'bid_low': float(candle['bid']['l']),
                    'bid_close': float(candle['bid']['c']),
                    'ask_open': float(candle['ask']['o']),
                    'ask_high': float(candle['ask']['h']),
                    'ask_low': float(candle['ask']['l']),
                    'ask_close': float(candle['ask']['c']),
                })

            all_rows.extend(batch_rows)

            if len(batch_rows) > 0:
                earliest = min(row['date'] for row in batch_rows)
                to_time = earliest.isoformat()

            print(f"  Batch {batch_num + 1}/{NUM_REQUESTS}: {len(batch_rows)} candles")

        except requests.exceptions.HTTPError as e:
            print(f"  ERROR: HTTP {e.response.status_code}")
            print(f"  {e.response.text}")
            break
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            break

        time.sleep(0.5)

    if len(all_rows) > 0:
        df = pd.DataFrame(all_rows)

        # Calculate spread
        df['spread'] = df['ask_close'] - df['bid_close']
        df['spread_pct'] = (df['spread'] / df['close']) * 100

        # Sort and remove duplicates
        df = df.sort_values('date')
        df = df.drop_duplicates(subset=['date'], keep='first')

        df.to_csv(output_file, index=False)

        print(f"  Total: {len(df)} candles")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Days of data: {(df['date'].max() - df['date'].min()).days}")
        print(f"  Avg spread: {df['spread_pct'].mean():.4f}%")
        print()

        if args.analyze:
            all_data_frames[pair_name] = df
    else:
        print(f"  No data fetched!")
        print()

print("="*80)
print("DONE!")
print("="*80)
print()

# Analyze spreads
if args.analyze and len(all_data_frames) > 0:
    print("="*80)
    print("SPREAD ANALYSIS BY HOUR (5M DATA)")
    print("="*80)
    print()

    for pair_name, df in all_data_frames.items():
        print(f"{pair_name}:")

        df['hour'] = pd.to_datetime(df['date']).dt.hour

        hourly_spreads = df.groupby('hour')['spread_pct'].agg(['mean', 'median', 'std', 'count'])
        hourly_spreads = hourly_spreads.sort_values('mean', ascending=False)

        print(f"  Overall avg spread: {df['spread_pct'].mean():.4f}%")
        print(f"  Median spread: {df['spread_pct'].median():.4f}%")
        print()
        print("  Highest spread hours (UTC):")
        for idx, row in hourly_spreads.head(5).iterrows():
            print(f"    Hour {idx:2d}: {row['mean']:.4f}% avg (median {row['median']:.4f}%, {int(row['count'])} samples)")

        print()
        print("  Lowest spread hours (UTC):")
        for idx, row in hourly_spreads.tail(5).iterrows():
            print(f"    Hour {idx:2d}: {row['mean']:.4f}% avg (median {row['median']:.4f}%, {int(row['count'])} samples)")

        print()

    print("="*80)
    print("NOTES FOR 5M TRADING")
    print("="*80)
    print()
    print("Considerations for 5-minute timeframe:")
    print("  - Spread costs are VERY significant - likely 2-3x the profit per trade")
    print("  - Need very tight risk management")
    print("  - Lookback period: ~240 periods = 20 hours (vs 80 periods for 15m)")
    print("  - Scalping-level frequency - potentially hundreds of trades")
    print("  - May not be profitable after spread costs")
    print()
