"""
Time Simulator
==============
Controls time advancement during simulation.
"""
from datetime import datetime, timedelta
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class TimeSimulator:
    """
    Controls time advancement during simulation.

    Advances time bar-by-bar (15-minute intervals) and triggers
    events at appropriate times (signal checks, position updates, etc.)
    """

    def __init__(self, start_time: datetime, end_time: datetime):
        """
        Initialize time simulator.

        Args:
            start_time: Simulation start time
            end_time: Simulation end time
        """
        # Ensure times are timezone-aware (UTC)
        import pytz
        if start_time.tzinfo is None:
            start_time = pytz.UTC.localize(start_time)
        if end_time.tzinfo is None:
            end_time = pytz.UTC.localize(end_time)

        self.start_time = start_time
        self.end_time = end_time
        self.current_time = start_time
        self.bar_interval = timedelta(minutes=15)

        logger.info(f"TimeSimulator: {start_time} to {end_time}")

    def advance_to_next_bar(self) -> bool:
        """
        Advance time to the next 15-minute bar.

        Returns:
            True if advanced successfully, False if reached end_time
        """
        next_time = self.current_time + self.bar_interval

        if next_time > self.end_time:
            logger.info("Reached end of simulation period")
            return False

        self.current_time = next_time
        return True

    def is_15m_signal_time(self) -> bool:
        """
        Check if current time is a 15-minute signal check time.

        Signals are checked at :00, :15, :30, :45 of each hour.
        """
        return self.current_time.minute in [0, 15, 30, 45]

    def is_position_update_time(self) -> bool:
        """
        Check if current time is a position update time.

        In production, positions are updated every minute.
        In simulation, we update every bar (15 minutes).
        """
        return True  # Every bar

    def is_emergency_check_time(self) -> bool:
        """
        Check if current time is an emergency check time.

        Production checks every 5 minutes, we'll check every bar.
        """
        return self.current_time.minute % 15 == 0

    def is_new_day(self, last_day: Optional[datetime]) -> bool:
        """
        Check if we've crossed into a new trading day.

        Args:
            last_day: Previous day's date

        Returns:
            True if it's a new day
        """
        if last_day is None:
            return True

        return self.current_time.date() != last_day.date()

    def get_progress_pct(self) -> float:
        """Get simulation progress percentage"""
        total_duration = (self.end_time - self.start_time).total_seconds()
        elapsed = (self.current_time - self.start_time).total_seconds()
        return (elapsed / total_duration) * 100 if total_duration > 0 else 0

    def get_remaining_bars(self) -> int:
        """Get number of bars remaining in simulation"""
        remaining = self.end_time - self.current_time
        return int(remaining.total_seconds() / (self.bar_interval.total_seconds()))

    def __repr__(self) -> str:
        return f"TimeSimulator(current={self.current_time}, progress={self.get_progress_pct():.1f}%)"
