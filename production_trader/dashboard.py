"""
Trading Dashboard - Real-time Monitoring
========================================
Streamlit dashboard for monitoring live trading activity.

USAGE:
    streamlit run production_trader/dashboard.py

Features:
- Live account metrics (balance, P/L, positions)
- Real-time candlestick charts from OANDA
- Open positions table
- Recent trades history
- Auto-refresh every 15 seconds
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from production_trader.execution.oanda_broker import OandaBroker

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .positive {
        color: #00cc00;
        font-weight: bold;
    }
    .negative {
        color: #ff0000;
        font-weight: bold;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
STATE_FILE = 'production_trader/state/trading_state.json'


@st.cache_resource
def get_broker(account_type='live'):
    """Initialize OANDA broker (cached)"""
    try:
        # Load credentials based on account type
        if account_type == 'practice':
            api_key = os.getenv('OANDA_PRACTICE_API_KEY')
            account_id = os.getenv('OANDA_PRACTICE_ACCOUNT_ID')
        else:  # live
            api_key = os.getenv('OANDA_API_KEY')
            account_id = os.getenv('OANDA_ACCOUNT_ID')

        if not api_key or not account_id:
            st.error(f"OANDA {account_type} credentials not found in environment variables!")
            st.info(f"Please set: OANDA{'_PRACTICE' if account_type == 'practice' else ''}_API_KEY and OANDA{'_PRACTICE' if account_type == 'practice' else ''}_ACCOUNT_ID")
            return None

        broker = OandaBroker(api_key, account_id, account_type)

        if broker.check_connection():
            return broker
        else:
            st.error(f"Failed to connect to OANDA {account_type} account")
            return None

    except Exception as e:
        st.error(f"Error initializing broker: {e}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_candles(pair: str, count: int = 200, account_type: str = 'live'):
    """Fetch historical candles from OANDA (cached)"""
    broker = get_broker(account_type)
    if broker:
        try:
            return broker.get_historical_candles(pair, timeframe='M15', count=count)
        except Exception as e:
            st.error(f"Error fetching candles for {pair}: {e}")
            return None
    return None


def load_state():
    """Load trading state from JSON file"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        else:
            return {
                'capital': 500.0,
                'peak_capital': 500.0,
                'daily_pnl': 0.0,
                'total_trades': 0,
                'positions': {},
                'last_update': None
            }
    except Exception as e:
        st.error(f"Error loading state: {e}")
        return None


def create_candlestick_chart(df: pd.DataFrame, pair: str, positions=None, trades=None):
    """Create candlestick chart with optional position/trade markers"""

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#00cc00',
        decreasing_line_color='#ff0000'
    ))

    # Add a thin line connecting close prices to show continuity during gaps
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        mode='lines',
        line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
        name='Close (connected)',
        showlegend=False,
        hoverinfo='skip'
    ))

    # Add position markers if provided
    if positions:
        for pos in positions:
            if pos['pair'] == pair:
                # Entry marker
                fig.add_trace(go.Scatter(
                    x=[pos['entry_date']],
                    y=[pos['entry_price']],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='green' if pos['direction'] == 'long' else 'red',
                        symbol='triangle-up' if pos['direction'] == 'long' else 'triangle-down',
                        line=dict(width=2, color='white')
                    ),
                    name=f"Entry ({pos['direction'].upper()})",
                    showlegend=True,
                    hovertemplate=f"Entry: {pos['entry_price']:.5f}<br>{pos['direction'].upper()}<extra></extra>"
                ))

    # Add trade markers if provided
    if trades:
        for trade in trades:
            if trade['pair'] == pair:
                # Entry marker
                color = 'green' if trade['direction'] == 'long' else 'red'
                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(trade['entry_time'])],
                    y=[trade['entry_price']],
                    mode='markers',
                    marker=dict(size=8, color=color, symbol='circle'),
                    name=f"Trade Entry",
                    showlegend=False,
                    hovertemplate=f"Entry: {trade['entry_price']:.5f}<br>{trade['direction'].upper()}<extra></extra>"
                ))

                # Exit marker
                exit_color = 'green' if trade['pl'] > 0 else 'red'
                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(trade['exit_time'])],
                    y=[trade['exit_price']],
                    mode='markers',
                    marker=dict(size=8, color=exit_color, symbol='x'),
                    name=f"Trade Exit",
                    showlegend=False,
                    hovertemplate=f"Exit: {trade['exit_price']:.5f}<br>P/L: {trade['pl']:.2f}<extra></extra>"
                ))

    # Layout
    fig.update_layout(
        title=f"{pair} - 15 Minute Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        height=600,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        # Remove gaps in x-axis for non-trading hours (weekends)
        xaxis=dict(
            rangebreaks=[
                # Hide weekends (Saturday & Sunday)
                dict(bounds=["sat", "mon"]),
            ]
        )
    )

    return fig


def format_duration(minutes: int) -> str:
    """Format duration in minutes to human-readable string"""
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    if hours < 24:
        return f"{hours}h {mins}m"
    days = hours // 24
    hours = hours % 24
    return f"{days}d {hours}h"


def main():
    """Main dashboard"""

    # Title
    st.title("📊 Trading Dashboard")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Account type selector
        account_type = st.radio(
            "Account Type",
            ["live", "practice"],
            index=0,  # Default to live
            help="Select which OANDA account to monitor"
        )

        # Show current account type with color
        if account_type == "live":
            st.warning("⚠️ **LIVE ACCOUNT** - Real money!")
        else:
            st.info("📝 Practice account (paper trading)")

        st.markdown("---")

        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (15s)", value=False)

        if auto_refresh:
            st.info("Dashboard will refresh every 15 seconds")

        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.cache_resource.clear()  # Clear broker cache too
            st.rerun()

        st.markdown("---")
        st.markdown("### 📋 Navigation")
        page = st.radio("Select Page", ["Live Monitor", "Trade History", "Performance"])

    # Load state
    state = load_state()
    if not state:
        st.error("Failed to load trading state")
        return

    # Get account summary from OANDA
    broker = get_broker(account_type)
    account = None
    if broker:
        try:
            account = broker.get_account_summary()
        except Exception as e:
            st.warning(f"Could not fetch account summary: {e}")

    # ==================== PAGE: LIVE MONITOR ====================
    if page == "Live Monitor":

        # Top metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            balance = account.balance if account else state['capital']
            st.metric(
                "💰 Balance",
                f"${balance:,.2f}",
                delta=f"{state.get('daily_pnl', 0):+.2f} today"
            )

        with col2:
            open_count = account.open_trade_count if account else len(state.get('positions', {}))
            st.metric(
                "📈 Open Positions",
                open_count
            )

        with col3:
            total_trades = state.get('total_trades', 0)
            st.metric(
                "📊 Total Trades",
                total_trades
            )

        with col4:
            peak = state.get('peak_capital', balance)
            drawdown = ((balance - peak) / peak * 100) if peak > 0 else 0
            st.metric(
                "📉 Drawdown",
                f"{drawdown:.2f}%",
                delta=None
            )

        st.markdown("---")

        # Chart section
        st.subheader("📈 Chart View")

        # Pair selector
        selected_pair = st.selectbox("Select Pair", PAIRS, index=0)

        # Fetch and display candles
        with st.spinner(f"Loading {selected_pair} chart..."):
            candles = get_candles(selected_pair, count=200, account_type=account_type)

            if candles is not None and not candles.empty:
                # Convert positions to list format
                positions = []
                for trade_id, pos in state.get('positions', {}).items():
                    positions.append({
                        'pair': pos['pair'],
                        'direction': pos['direction'],
                        'entry_price': pos['entry_price'],
                        'entry_date': pd.to_datetime(pos['entry_date']),
                        'size': pos['size']
                    })

                # Create chart
                fig = create_candlestick_chart(candles, selected_pair, positions=positions)
                st.plotly_chart(fig, width='stretch')

                # Chart info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"{candles['close'].iloc[-1]:.5f}")
                with col2:
                    change = ((candles['close'].iloc[-1] - candles['close'].iloc[-50]) / candles['close'].iloc[-50] * 100) if len(candles) >= 50 else 0
                    st.metric("50-Bar Change", f"{change:+.2f}%")
                with col3:
                    st.metric("Spread", f"{candles['spread_pct'].iloc[-1]*100:.3f}%")
            else:
                st.error(f"Failed to load candles for {selected_pair}")

        st.markdown("---")

        # Open Positions
        st.subheader("📊 Open Positions")

        if state.get('positions'):
            positions_data = []
            for trade_id, pos in state['positions'].items():
                # Calculate duration
                entry_time = datetime.fromisoformat(pos['entry_date'])
                duration_mins = int((datetime.now() - entry_time).total_seconds() / 60)

                # Get current price
                current_price = "..."
                unrealized_pl = "..."
                if broker:
                    try:
                        prices = broker.get_current_prices([pos['pair']])
                        if pos['pair'] in prices:
                            price_data = prices[pos['pair']]
                            current_price = price_data.bid_close if pos['direction'] == 'long' else price_data.ask_close

                            # Calculate unrealized P/L
                            if pos['direction'] == 'long':
                                pl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                            else:
                                pl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * 100

                            unrealized_pl = f"{pl_pct:+.2f}%"
                    except:
                        pass

                positions_data.append({
                    'Pair': pos['pair'],
                    'Direction': pos['direction'].upper(),
                    'Entry': f"{pos['entry_price']:.5f}",
                    'Current': f"{current_price:.5f}" if isinstance(current_price, float) else current_price,
                    'Size': pos['size'],
                    'Confidence': f"{pos['confidence']:.1%}",
                    'Unrealized P/L': unrealized_pl,
                    'Duration': format_duration(duration_mins)
                })

            df_positions = pd.DataFrame(positions_data)
            st.dataframe(df_positions, width='stretch', hide_index=True)
        else:
            st.info("No open positions")

    # ==================== PAGE: TRADE HISTORY ====================
    elif page == "Trade History":
        st.subheader("📜 Recent Trades")

        # Load trades from CSV if exists
        trades_file = 'production_trader/trades_history.csv'
        if os.path.exists(trades_file):
            df_trades = pd.read_csv(trades_file)

            # Quick date filters
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                filter_pair = st.multiselect("Filter by Pair", PAIRS, default=PAIRS)
            with col2:
                date_filter = st.radio(
                    "Time Period",
                    ["Today", "Last 7 Days", "Last 30 Days", "All Time"],
                    horizontal=True,
                    index=1  # Default to "Last 7 Days"
                )
            with col3:
                st.write("")  # Spacer

            # Convert date filter to days
            if date_filter == "Today":
                filter_days = 0  # Special case for today
            elif date_filter == "Last 7 Days":
                filter_days = 7
            elif date_filter == "Last 30 Days":
                filter_days = 30
            else:  # All Time
                filter_days = 99999

            # Apply filters
            df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])

            # Handle "Today" filter specially (same calendar day)
            if filter_days == 0:
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                df_filtered = df_trades[
                    (df_trades['pair'].isin(filter_pair)) &
                    (df_trades['exit_time'] >= today_start)
                ].copy()
            else:
                cutoff_date = datetime.now() - timedelta(days=filter_days)
                df_filtered = df_trades[
                    (df_trades['pair'].isin(filter_pair)) &
                    (df_trades['exit_time'] >= cutoff_date)
                ].copy()

            # Calculate metrics
            if not df_filtered.empty:
                # Show date range
                earliest = df_filtered['exit_time'].min()
                latest = df_filtered['exit_time'].max()
                st.caption(f"Showing {len(df_filtered)} trades from {earliest.strftime('%m/%d %H:%M')} to {latest.strftime('%m/%d %H:%M')}")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Trades", len(df_filtered))
                with col2:
                    win_rate = (df_filtered['pl'] > 0).mean() * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                with col3:
                    total_pl = df_filtered['pl'].sum()
                    st.metric("Total P/L", f"${total_pl:+.2f}", delta=f"{(total_pl / 500 * 100):.2f}% of capital" if total_pl != 0 else None)
                with col4:
                    avg_pl = df_filtered['pl'].mean()
                    st.metric("Avg P/L", f"${avg_pl:+.2f}")

                st.markdown("---")

                # Format for display
                df_display = df_filtered.copy()
                df_display['Entry Time'] = df_display['entry_time'].apply(lambda x: pd.to_datetime(x).strftime('%m/%d %H:%M'))
                df_display['Exit Time'] = df_display['exit_time'].apply(lambda x: x.strftime('%m/%d %H:%M'))

                # Add win/loss indicator
                df_display['Result'] = df_display['pl'].apply(lambda x: '✅' if x > 0 else ('⚠️' if x == 0 else '❌'))

                df_display['P/L $'] = df_display['pl'].apply(lambda x: f"${x:+.2f}")
                df_display['P/L %'] = df_display['pl_pct'].apply(lambda x: f"{x*100:+.2f}%")

                # Add duration if available (calculate from entry/exit times)
                if 'bars_held' in df_display.columns:
                    df_display['Duration'] = df_display['bars_held'].apply(lambda x: f"{int(x)}p")

                # Build display columns
                display_cols = ['Result', 'pair', 'direction', 'Entry Time', 'Exit Time', 'P/L $', 'P/L %']
                if 'Duration' in df_display.columns:
                    display_cols.append('Duration')
                if 'exit_reason' in df_display.columns:
                    display_cols.append('exit_reason')

                # Display table
                st.dataframe(
                    df_display[display_cols].sort_values('Exit Time', ascending=False),
                    width='stretch',
                    hide_index=True
                )
            else:
                st.info("No trades found for selected filters")
        else:
            st.info("No trade history available yet")

    # ==================== PAGE: PERFORMANCE ====================
    elif page == "Performance":
        st.subheader("📊 Performance Analysis")
        st.info("Performance analytics coming soon!")

    # Footer
    st.markdown("---")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    # Auto-refresh (at the end so page loads first)
    if auto_refresh:
        import time
        time.sleep(15)
        st.rerun()


if __name__ == "__main__":
    main()
