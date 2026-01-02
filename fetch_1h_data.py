"""
FETCH 1-HOUR DATA FROM OANDA
=============================
Fetches 1-hour candle data for testing the breakout strategy on higher frequency
"""
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Oanda API configuration
OANDA_API_KEY = os.getenv('OANDA_API_KEY', '')
OANDA_ACCOUNT_TYPE = os.getenv('OANDA_ACCOUNT_TYPE', 'practice')  # 'practice' or 'live'

if OANDA_ACCOUNT_TYPE == 'practice':
    OANDA_API_URL = 'https://api-fxpractice.oanda.com'
else:
    OANDA_API_URL = 'https://api-fxtrade.oanda.com'

# Pairs to fetch
PAIRS = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD']  # Oanda format uses underscores

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

for pair in PAIRS:
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

        # Convert pair name back to standard format (EUR_USD -> EURUSD)
        pair_name = pair.replace('_', '')

        # Save to CSV
        output_file = f'data_1h/{pair_name}_1h.csv'
        df.to_csv(output_file, index=False)

        print(f"  Saved {len(df)} candles to {output_file}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Avg spread: {df['spread_pct'].mean():.4f}%")
        print()

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
