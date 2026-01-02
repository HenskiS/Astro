"""
Fetch daily data with spreads for NEW pairs (expansion for multi-pair diversification).
Tier 1 pairs: USDCAD, USDCHF, NZDUSD, EURGBP, EURJPY, AUDJPY
"""
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class OandaDailySpreadFetcher:
    """Fetch OANDA daily data with bid/ask prices for spread analysis"""

    def __init__(self):
        self.api_key = os.getenv('OANDA_API_KEY')
        self.base_url = "https://api-fxtrade.oanda.com/v3"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        if not self.api_key:
            raise ValueError("OANDA_API_KEY not found in .env")

    def get_instrument_name(self, pair):
        """Convert EURUSD -> EUR_USD"""
        if len(pair) == 6:
            return f"{pair[:3]}_{pair[3:]}"
        raise ValueError(f"Unknown pair format: {pair}")

    def fetch_daily_with_spreads(self, pair, count=5000):
        """
        Fetch daily historical data with bid/ask prices.

        Returns DataFrame with:
        - date, open, high, low, close, volume (mid prices)
        - bid_close, ask_close
        - spread (in price units)
        - spread_pips (spread in pips)
        - spread_pct (spread as % of price)
        """
        instrument = self.get_instrument_name(pair)

        url = f"{self.base_url}/instruments/{instrument}/candles"
        params = {
            'count': min(count, 5000),
            'granularity': 'D',
            'price': 'MBA',  # Mid, Bid, Ask prices
            'alignmentTimezone': 'America/New_York',
            'dailyAlignment': 9  # 9 AM EST alignment
        }

        print(f"Fetching {pair} daily data with bid/ask prices...")
        print(f"Count: {params['count']}, Alignment: 9 AM EST")

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            if 'candles' not in data:
                print(f"Error: {data}")
                return pd.DataFrame()

            candles = data['candles']

            if not candles:
                print(f"No candles returned for {pair}")
                return pd.DataFrame()

            # Parse candles with bid/ask data
            rows = []
            for candle in candles:
                if not candle['complete']:
                    continue

                # Mid prices (for OHLC)
                mid_o = float(candle['mid']['o'])
                mid_h = float(candle['mid']['h'])
                mid_l = float(candle['mid']['l'])
                mid_c = float(candle['mid']['c'])

                # Bid/Ask close prices
                bid_c = float(candle['bid']['c'])
                ask_c = float(candle['ask']['c'])

                # Calculate spread
                spread = ask_c - bid_c

                # Convert spread to pips
                if 'JPY' in pair:
                    spread_pips = spread * 100  # JPY pairs: 2 decimals
                else:
                    spread_pips = spread * 10000  # Standard pairs: 4 decimals

                rows.append({
                    'date': candle['time'],
                    'open': mid_o,
                    'high': mid_h,
                    'low': mid_l,
                    'close': mid_c,
                    'volume': int(candle['volume']),
                    'bid_close': bid_c,
                    'ask_close': ask_c,
                    'spread': spread,
                    'spread_pips': spread_pips,
                    'spread_pct': spread / mid_c
                })

            df = pd.DataFrame(rows)

            if df.empty:
                print(f"No complete candles found")
                return df

            # Convert timestamp to datetime
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            print(f"Successfully fetched {len(df)} candles")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"\nSpread statistics (pips):")
            print(f"  Min: {df['spread_pips'].min():.2f}")
            print(f"  Mean: {df['spread_pips'].mean():.2f}")
            print(f"  Median: {df['spread_pips'].median():.2f}")
            print(f"  Max: {df['spread_pips'].max():.2f}")
            print(f"\nSpread statistics (% of price):")
            print(f"  Min: {df['spread_pct'].min()*100:.4f}%")
            print(f"  Mean: {df['spread_pct'].mean()*100:.4f}%")
            print(f"  Median: {df['spread_pct'].median()*100:.4f}%")
            print(f"  Max: {df['spread_pct'].max()*100:.4f}%")

            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Tier 1 new pairs for multi-pair expansion
    NEW_PAIRS = ['USDCAD', 'USDCHF', 'NZDUSD', 'EURGBP', 'EURJPY', 'AUDJPY']

    print("=" * 80)
    print("OANDA Daily Data Fetcher - NEW PAIRS WITH SPREADS")
    print("=" * 80)
    print(f"Pairs: {', '.join(NEW_PAIRS)}")
    print("=" * 80)

    fetcher = OandaDailySpreadFetcher()
    results = {}

    for pair in NEW_PAIRS:
        print(f"\n{'=' * 80}")
        print(f"{pair}")
        print("=" * 80)

        df = fetcher.fetch_daily_with_spreads(pair, count=5000)

        if not df.empty:
            results[pair] = {
                'candles': len(df),
                'spread_mean': df['spread_pips'].mean(),
                'spread_pct_mean': df['spread_pct'].mean() * 100
            }

            # Save with spreads
            os.makedirs('data', exist_ok=True)
            spreads_file = f'data/{pair}_1day_with_spreads.csv'
            df.to_csv(spreads_file, index=False)
            print(f"\nSaved to {spreads_file}")

            # Also save standard format (for compatibility with existing backtest)
            standard_file = f'data/{pair}_1day_oanda.csv'
            df_standard = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
            df_standard.to_csv(standard_file, index=False)
            print(f"Saved to {standard_file}")
        else:
            results[pair] = None
            print(f"FAILED to fetch {pair}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Pair':<10} {'Candles':>10} {'Spread (pips)':>15} {'Spread (%)':>12}")
    print("-" * 50)
    for pair in NEW_PAIRS:
        if results.get(pair):
            r = results[pair]
            print(f"{pair:<10} {r['candles']:>10} {r['spread_mean']:>15.2f} {r['spread_pct_mean']:>11.4f}%")
        else:
            print(f"{pair:<10} {'FAILED':>10}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
