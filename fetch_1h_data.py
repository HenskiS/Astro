"""
FETCH 1-HOUR DATA FROM OANDA
=============================
Fetches 1-hour candle data for testing the breakout strategy on higher frequency

USAGE:
  python fetch_1h_data.py              # Fetch data (skip if files exist)
  python fetch_1h_data.py --force      # Re-fetch all data
  python fetch_1h_data.py --analyze    # Fetch and analyze spreads by hour
"""
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Fetch 1-hour data from Oanda')
parser.add_argument('--force', action='store_true', help='Re-fetch data even if files exist')
parser.add_argument('--analyze', action='store_true', help='Analyze spreads by hour after fetching')
args = parser.parse_args()

# Oanda API configuration
OANDA_API_KEY = os.getenv('OANDA_API_KEY', '')
OANDA_ACCOUNT_TYPE = os.getenv('OANDA_ACCOUNT_TYPE', 'practice')  # 'practice' or 'live'

if OANDA_ACCOUNT_TYPE == 'practice':
    OANDA_API_URL = 'https://api-fxpractice.oanda.com'
else:
    OANDA_API_URL = 'https://api-fxtrade.oanda.com'

# Pairs to fetch (8 pairs to match daily strategy)
PAIRS = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CAD', 'USD_CHF', 'NZD_USD', 'EUR_JPY']  # Oanda format uses underscores

# Number of candles to fetch (max 5000 per request)
COUNT = 5000

print("="*80)
print("FETCHING 1-HOUR DATA FROM OANDA")
print("="*80)
print()

if not OANDA_API_KEY:
    print("ERROR: OANDA_API_KEY environment variable not set!")
    print()
    print("Please set your Oanda API key:")
    print("  Windows: set OANDA_API_KEY=your_key_here")
    print("  Linux/Mac: export OANDA_API_KEY=your_key_here")
    print()
    print("You can get an API key from: https://www.oanda.com/account/tpa/personal_token")
    exit(1)

print(f"API URL: {OANDA_API_URL}")
print(f"Account Type: {OANDA_ACCOUNT_TYPE}")
print(f"Fetching {COUNT} 1-hour candles per pair")
print()

# Create data directory
os.makedirs('data_1h', exist_ok=True)

# Track all fetched data for analysis
all_data_frames = {}

for pair in PAIRS:
    # Convert pair name to standard format (EUR_USD -> EURUSD)
    pair_name = pair.replace('_', '')
    output_file = f'data_1h/{pair_name}_1h.csv'

    # Skip if file exists and --force not used
    if os.path.exists(output_file) and not args.force:
        print(f"{pair}: File already exists, skipping (use --force to re-fetch)")
        if args.analyze:
            df = pd.read_csv(output_file)
            df['date'] = pd.to_datetime(df['date'])
            all_data_frames[pair_name] = df
        continue

    print(f"Fetching {pair}...")

    # Oanda API endpoint
    url = f"{OANDA_API_URL}/v3/instruments/{pair}/candles"

    headers = {
        'Authorization': f'Bearer {OANDA_API_KEY}',
        'Content-Type': 'application/json'
    }

    params = {
        'granularity': 'H1',  # 1-hour candles
        'count': COUNT,
        'price': 'MBA'  # Mid, Bid, Ask prices
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        candles = data['candles']

        # Parse candles into DataFrame
        rows = []
        for candle in candles:
            if not candle['complete']:
                continue  # Skip incomplete candles

            timestamp = pd.to_datetime(candle['time'])

            rows.append({
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

        df = pd.DataFrame(rows)

        # Calculate spread
        df['spread'] = df['ask_close'] - df['bid_close']
        df['spread_pct'] = (df['spread'] / df['close']) * 100

        # Sort by date
        df = df.sort_values('date')

        # Save to CSV
        df.to_csv(output_file, index=False)

        print(f"  Saved {len(df)} candles to {output_file}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Avg spread: {df['spread_pct'].mean():.4f}%")
        print()

        # Store for analysis
        if args.analyze:
            all_data_frames[pair_name] = df

    except requests.exceptions.HTTPError as e:
        print(f"  ERROR: HTTP {e.response.status_code}")
        print(f"  {e.response.text}")
        print()
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        print()

    # Rate limiting
    time.sleep(0.5)

print("="*80)
print("DONE!")
print("="*80)
print()

# Analyze spreads by hour if requested
if args.analyze and len(all_data_frames) > 0:
    print("="*80)
    print("SPREAD ANALYSIS BY HOUR")
    print("="*80)
    print()

    for pair_name, df in all_data_frames.items():
        print(f"{pair_name}:")

        # Extract hour from date
        df['hour'] = pd.to_datetime(df['date']).dt.hour

        # Calculate average spread by hour
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

        # Identify high spread hours (> 75th percentile)
        high_spread_threshold = df['spread_pct'].quantile(0.75)
        high_spread_hours = hourly_spreads[hourly_spreads['mean'] > high_spread_threshold].index.tolist()

        if len(high_spread_hours) > 0:
            print(f"  High spread hours (>75th percentile = {high_spread_threshold:.4f}%): {high_spread_hours}")
            print()

    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()
    print("Consider avoiding trades during:")
    print("  - Weekend hours (Friday 21:00+ UTC, Sunday)")
    print("  - Low liquidity periods (identified above as high spread hours)")
    print("  - Hours where spread > 2x median spread")
    print()
