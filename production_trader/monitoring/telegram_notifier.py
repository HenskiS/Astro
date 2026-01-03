"""
Telegram Notifier
=================
Sends trading alerts and daily summaries via Telegram.

Alert Types:
- Position opened/closed
- Emergency stops triggered
- Max drawdown warnings
- API connection issues
- Daily summary at 00:00 UTC
"""
import logging
from datetime import datetime
from typing import Dict, Optional
import asyncio

logger = logging.getLogger(__name__)

# Try to import telegram library
try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")


class TelegramNotifier:
    """
    Sends trading notifications via Telegram.

    Provides real-time alerts and daily summaries for
    monitoring the production trading system.
    """

    def __init__(self, config):
        """
        Initialize Telegram notifier.

        Args:
            config: Telegram configuration
        """
        self.enabled = config.enabled and TELEGRAM_AVAILABLE
        self.bot_token = config.bot_token if self.enabled else None
        self.chat_id = config.chat_id if self.enabled else None
        self.bot = None

        if self.enabled:
            if not self.bot_token or not self.chat_id:
                logger.warning("Telegram credentials not configured - disabling notifications")
                self.enabled = False
            else:
                try:
                    self.bot = Bot(token=self.bot_token)
                    logger.info("TelegramNotifier initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Telegram bot: {e}")
                    self.enabled = False
        else:
            logger.info("TelegramNotifier disabled (telegram library not available)")

    def _send_message(self, message: str):
        """Send a message via Telegram"""
        if not self.enabled:
            logger.debug(f"Telegram disabled - would send: {message}")
            return

        try:
            # Run async send in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
            )
            loop.close()
            logger.debug("Telegram message sent")

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    def notify_position_opened(self, position: Dict):
        """
        Notify about new position opened.

        Args:
            position: Position dictionary
        """
        message = (
            f"✅ *Position Opened*\n\n"
            f"Pair: `{position['pair']}`\n"
            f"Direction: *{position['direction'].upper()}*\n"
            f"Size: `{position['size']}`\n"
            f"Entry: `{position['entry_price']:.5f}`\n"
            f"Confidence: `{position['confidence']:.1%}`\n"
            f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC`"
        )
        self._send_message(message)

    def notify_position_closed(
        self,
        pair: str,
        direction: str,
        profit_pct: float,
        profit_dollars: float,
        reason: str,
        periods_held: int,
        capital_after: float
    ):
        """
        Notify about position closed.

        Args:
            pair: Currency pair
            direction: Position direction
            profit_pct: Profit percentage
            profit_dollars: Profit in dollars
            reason: Exit reason
            periods_held: How long position was held
            capital_after: Capital after closing
        """
        # Emoji based on profit
        if profit_dollars > 0:
            emoji = "🎯" if reason == 'target' else "💰"
        else:
            emoji = "📉"

        # Format hold time
        hours = periods_held / 4  # Assuming 15m periods
        hold_time = f"{hours:.1f}h" if hours < 24 else f"{hours/24:.1f}d"

        message = (
            f"{emoji} *Position Closed*\n\n"
            f"Pair: `{pair}`\n"
            f"Direction: *{direction.upper()}*\n"
            f"P/L: `{profit_pct:+.2%}` (${profit_dollars:+.2f})\n"
            f"Reason: `{reason}`\n"
            f"Hold Time: `{hold_time}`\n"
            f"Capital: `${capital_after:.2f}`\n"
            f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC`"
        )
        self._send_message(message)

    def notify_emergency_stop(self, pair: str, drawdown: float):
        """
        Notify about emergency stop triggered.

        Args:
            pair: Currency pair
            drawdown: Current drawdown
        """
        message = (
            f"🚨 *EMERGENCY STOP*\n\n"
            f"Pair: `{pair}`\n"
            f"Drawdown: `{drawdown:.1%}`\n"
            f"Action: Position closed\n"
            f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC`"
        )
        self._send_message(message)

    def notify_max_drawdown(self, current_capital: float, peak_capital: float):
        """
        Notify about max drawdown exceeded.

        Args:
            current_capital: Current capital
            peak_capital: Peak capital
        """
        drawdown = (current_capital - peak_capital) / peak_capital

        message = (
            f"🚨 *MAX DRAWDOWN EXCEEDED*\n\n"
            f"Current: `${current_capital:.2f}`\n"
            f"Peak: `${peak_capital:.2f}`\n"
            f"Drawdown: `{drawdown:.1%}`\n"
            f"Action: *CLOSING ALL POSITIONS*\n"
            f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC`"
        )
        self._send_message(message)

    def notify_daily_loss_limit(self, daily_pnl: float, capital: float):
        """
        Notify about daily loss limit exceeded.

        Args:
            daily_pnl: Daily P&L
            capital: Current capital
        """
        loss_pct = daily_pnl / capital

        message = (
            f"⚠️ *Daily Loss Limit Exceeded*\n\n"
            f"Daily P/L: `${daily_pnl:+.2f}` ({loss_pct:+.1%})\n"
            f"Capital: `${capital:.2f}`\n"
            f"Action: *No new positions until tomorrow*\n"
            f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC`"
        )
        self._send_message(message)

    def notify_api_error(self, error_message: str):
        """
        Notify about API connection error.

        Args:
            error_message: Error description
        """
        message = (
            f"❌ *API Connection Error*\n\n"
            f"Error: `{error_message}`\n"
            f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC`"
        )
        self._send_message(message)

    def notify_system_start(self, config_info: Dict):
        """
        Notify that system has started.

        Args:
            config_info: System configuration info
        """
        message = (
            f"🚀 *Production Trader Started*\n\n"
            f"Account: `{config_info.get('account_type', 'unknown')}`\n"
            f"Capital: `${config_info.get('capital', 0):.2f}`\n"
            f"Pairs: `{config_info.get('pairs_count', 0)}`\n"
            f"Max Positions: `{config_info.get('max_positions', 0)}`\n"
            f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC`"
        )
        self._send_message(message)

    def notify_system_stop(self, final_capital: float, total_trades: int):
        """
        Notify that system has stopped.

        Args:
            final_capital: Final capital
            total_trades: Total trades executed
        """
        message = (
            f"🛑 *Production Trader Stopped*\n\n"
            f"Final Capital: `${final_capital:.2f}`\n"
            f"Total Trades: `{total_trades}`\n"
            f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC`"
        )
        self._send_message(message)

    def send_daily_summary(
        self,
        capital: float,
        daily_pnl: float,
        trades_today: int,
        open_positions: int,
        win_rate: Optional[float] = None
    ):
        """
        Send daily summary at 00:00 UTC.

        Args:
            capital: Current capital
            daily_pnl: Daily P&L
            trades_today: Number of trades today
            open_positions: Number of open positions
            win_rate: Win rate percentage (optional)
        """
        daily_return = daily_pnl / capital if capital > 0 else 0

        # Emoji based on daily performance
        if daily_return > 0.01:
            emoji = "🚀"
        elif daily_return > 0:
            emoji = "📈"
        elif daily_return > -0.01:
            emoji = "📊"
        else:
            emoji = "📉"

        message = (
            f"{emoji} *Daily Summary*\n"
            f"_{datetime.now().strftime('%Y-%m-%d')}_\n\n"
            f"Capital: `${capital:.2f}`\n"
            f"Daily P/L: `${daily_pnl:+.2f}` ({daily_return:+.2%})\n"
            f"Trades Today: `{trades_today}`\n"
            f"Open Positions: `{open_positions}`\n"
        )

        if win_rate is not None:
            message += f"Win Rate: `{win_rate:.1%}`\n"

        message += f"\nTime: `{datetime.now().strftime('%H:%M:%S')} UTC`"

        self._send_message(message)


# Test code
if __name__ == '__main__':
    print("Testing TelegramNotifier...")

    # Mock config
    class MockConfig:
        enabled = False  # Set to True to test with real credentials
        bot_token = None
        chat_id = None

    notifier = TelegramNotifier(MockConfig())

    # Test notifications
    print("\n1. Testing position opened notification...")
    notifier.notify_position_opened({
        'pair': 'EURUSD',
        'direction': 'long',
        'size': 1000,
        'entry_price': 1.10000,
        'confidence': 0.85
    })

    print("\n2. Testing position closed notification...")
    notifier.notify_position_closed(
        pair='EURUSD',
        direction='long',
        profit_pct=0.015,
        profit_dollars=15.50,
        reason='target',
        periods_held=24,
        capital_after=515.50
    )

    print("\n3. Testing daily summary...")
    notifier.send_daily_summary(
        capital=515.50,
        daily_pnl=15.50,
        trades_today=10,
        open_positions=5,
        win_rate=0.875
    )

    print("\n✓ TelegramNotifier tests complete")
    print("(Messages not sent - Telegram disabled in test mode)")
