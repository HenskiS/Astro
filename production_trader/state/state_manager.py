"""
State Management
================
Handles state persistence using JSON for crash recovery.
"""
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def convert_to_python_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        Object with all numpy types converted to Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_types(item) for item in obj)
    else:
        return obj


class StateManager:
    """
    Manages trading state with JSON persistence.

    Handles:
    - Current capital
    - Open positions
    - Last check times
    - Daily P&L tracking
    """

    def __init__(self, json_file: str):
        """
        Initialize state manager.

        Args:
            json_file: Path to JSON state file
        """
        self.json_file = Path(json_file)
        self.state = self._load_or_create_state()

    def _load_or_create_state(self) -> Dict[str, Any]:
        """
        Load existing state or create new one.

        Returns:
            State dictionary
        """
        if self.json_file.exists():
            try:
                with open(self.json_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded state from {self.json_file}")
                return state
            except Exception as e:
                logger.error(f"Error loading state: {e}")
                return self._create_initial_state()
        else:
            return self._create_initial_state()

    def _create_initial_state(self) -> Dict[str, Any]:
        """
        Create initial state.

        Returns:
            New state dictionary
        """
        return {
            'capital': 500.0,
            'peak_capital': 500.0,
            'positions': {},
            'last_15m_check': None,
            'last_position_update': None,
            'last_emergency_check': None,
            'daily_pnl': 0.0,
            'daily_pnl_date': datetime.now().date().isoformat(),
            'trades_today': 0,
            'total_trades': 0,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

    def save_state(self) -> None:
        """
        Save current state to JSON file.
        """
        try:
            # Ensure directory exists
            self.json_file.parent.mkdir(parents=True, exist_ok=True)

            # Update timestamp
            self.state['updated_at'] = datetime.now().isoformat()

            # Convert numpy types to Python types for JSON serialization
            state_to_save = convert_to_python_types(self.state)

            # Write to file
            with open(self.json_file, 'w') as f:
                json.dump(state_to_save, f, indent=2)

            logger.debug("State saved successfully")

        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def get_capital(self) -> float:
        """Get current capital"""
        return self.state['capital']

    def set_capital(self, capital: float) -> None:
        """Set current capital"""
        self.state['capital'] = capital
        self.state['peak_capital'] = max(self.state['peak_capital'], capital)

    def get_peak_capital(self) -> float:
        """Get peak capital"""
        return self.state['peak_capital']

    def get_positions(self) -> Dict:
        """Get all positions"""
        return self.state['positions']

    def add_position(self, position_id: str, position_data: Dict) -> None:
        """Add a position"""
        self.state['positions'][position_id] = position_data

    def remove_position(self, position_id: str) -> None:
        """Remove a position"""
        if position_id in self.state['positions']:
            del self.state['positions'][position_id]

    def update_daily_pnl(self, pnl: float) -> None:
        """
        Update daily P&L, reset if new day.

        Args:
            pnl: Profit/loss to add
        """
        today = datetime.now().date().isoformat()

        if today != self.state['daily_pnl_date']:
            # New day, reset
            self.state['daily_pnl'] = 0.0
            self.state['daily_pnl_date'] = today
            self.state['trades_today'] = 0

        self.state['daily_pnl'] += pnl
        self.state['trades_today'] += 1

    def get_daily_pnl(self) -> float:
        """Get today's P&L"""
        today = datetime.now().date().isoformat()
        if today != self.state['daily_pnl_date']:
            return 0.0
        return self.state['daily_pnl']

    def update_last_check(self, check_type: str) -> None:
        """
        Update last check time.

        Args:
            check_type: '15m_check', 'position_update', or 'emergency_check'
        """
        key = f'last_{check_type}'
        if key in self.state:
            self.state[key] = datetime.now().isoformat()
