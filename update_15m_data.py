"""
UPDATE 15-MINUTE DATA FROM OANDA
=================================
Incrementally updates 15-minute candle data by fetching only new candles
since the last timestamp in existing CSV files.

USAGE:
  python update_15m_data.py
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Oanda API configuration
OANDA_API_KEY = os.getenv('OANDA_API_KEY', '')
OANDA_ACCOUNT_TYPE = os.getenv('OANDA_ACCOUNT_TYPE', 'practice')

if OANDA_ACCOUNT_TYPE == 'practice':
    OANDA_API_URL = 'https://api-fxpractice.oanda.com'
else:
    OANDA_API_URL = 'https://api-fxtrade.oanda.com'

# Pairs to update
PAIRS = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CAD', 'USD_CHF', 'NZD_USD', 'EUR_JPY']

print("="*80)
print("UPDATING 15-MINUTE DATA FROM OANDA")
print("="*80)
print()

if not OANDA_API_KEY:
    print("ERROR: OANDA_API_KEY not set!")
    exit(1)

print(f"API URL: {OANDA_API_URL}")
print(f"Account Type: {OANDA_ACCOUNT_TYPE}")
print()

# Create data directory if it doesn't exist
os.makedirs('data_15m', exist_ok=True)

for pair in PAIRS:
    pair_name = pair.replace('_', '')
    output_file = f'data_15m/{pair_name}_15m.csv'

    # Check if file exists and get latest date
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        df_existing['date'] = pd.to_datetime(df_existing['date'])
        latest_date = df_existing['date'].max()

        print(f"{pair}:")
        print(f"  Current data ends: {latest_date}")
        print(f"  Fetching new candles since {latest_date}...")
    else:
        print(f"{pair}: No existing data found, skipping (run fetch_15m_data.py first)")
        print()
        continue

    # Fetch new candles
    url = f"{OANDA_API_URL}/v3/instruments/{pair}/candles"
    headers = {
        'Authorization': f'Bearer {OANDA_API_KEY}',
        'Content-Type': 'application/json'
    }

    # Fetch from latest date to now (Oanda will return up to 5000 candles)
    # Format: RFC3339 with timezone (e.g., "2024-01-01T00:00:00Z")
    from_time = latest_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    params = {
        'granularity': 'M15',
        'from': from_time,  # Start from last known date
        'price': 'MBA'  # Mid, Bid, Ask
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        candles = data['candles']

        new_rows = []
        for candle in candles:
            if not candle['complete']:
                continue

            timestamp = pd.to_datetime(candle['time'])

            # Skip if we already have this timestamp
            if timestamp <= latest_date:
                continue

            new_rows.append({
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

        if len(new_rows) > 0:
            # Create DataFrame for new data
            df_new = pd.DataFrame(new_rows)
            df_new['spread'] = df_new['ask_close'] - df_new['bid_close']
            df_new['spread_pct'] = (df_new['spread'] / df_new['close']) * 100

            # Append to existing data
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.sort_values('date')
            df_combined = df_combined.drop_duplicates(subset=['date'], keep='last')

            # Save updated data
            df_combined.to_csv(output_file, index=False)

            print(f"  Added {len(new_rows)} new candles")
            print(f"  New data range: {df_new['date'].min()} to {df_new['date'].max()}")
            print(f"  Total candles: {len(df_combined)}")
        else:
            print(f"  No new candles (data is up to date)")

    except requests.exceptions.HTTPError as e:
        print(f"  ERROR: HTTP {e.response.status_code}")
        print(f"  {e.response.text}")
    except Exception as e:
        print(f"  ERROR: {str(e)}")

    print()

    # Rate limiting
    time.sleep(0.5)

print("="*80)
print("UPDATE COMPLETE!")
print("="*80)
