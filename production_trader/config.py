"""
Configuration Management
========================
Loads configuration from config.yaml with environment variable substitution.
"""
import os
import yaml
import re
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


@dataclass
class OandaConfig:
    """OANDA API configuration"""
    account_type: str
    account_id: str
    api_key: str


@dataclass
class CapitalConfig:
    """Capital and risk management configuration"""
    initial: float
    max_drawdown: float
    daily_loss_limit: float


@dataclass
class Strategy15mConfig:
    """15-minute strategy configuration"""
    enabled: bool
    pairs: list
    min_confidence: float
    lookback_periods: int
    avoid_hours: list
    position_size_pct: float  # 10% of capital per trade
    max_positions_total: int
    max_positions_per_pair: int
    immediate_stop_loss_pct: float  # -5% emergency stop
    emergency_stop_periods: int
    emergency_stop_loss_pct: float  # -4% for loser detection in 24-bar check
    trailing_stop_trigger: str  # 'on_target' to activate when breakout target hit
    trailing_stop_pct: float
    slippage_pct: float


@dataclass
class TelegramConfig:
    """Telegram notification configuration"""
    enabled: bool
    bot_token: str
    chat_id: str
    daily_summary_time: str


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str
    file: str
    max_size_mb: int
    backup_count: int


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    telegram: TelegramConfig
    logging: LoggingConfig


@dataclass
class StateConfig:
    """State persistence configuration"""
    json_file: str
    db_file: str
    save_interval_minutes: int


@dataclass
class ModelsConfig:
    """Model configuration"""
    directory: str
    feature_data_dir: str


@dataclass
class SystemConfig:
    """System configuration"""
    check_interval_seconds: int
    position_update_interval_minutes: int
    emergency_check_interval_minutes: int


@dataclass
class Config:
    """Main configuration object"""
    oanda: OandaConfig
    capital: CapitalConfig
    strategy_15m: Strategy15mConfig
    monitoring: MonitoringConfig
    state: StateConfig
    models: ModelsConfig
    system: SystemConfig


def substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in format ${VAR_NAME}.

    Args:
        value: Value to process (can be str, dict, list, etc.)

    Returns:
        Value with environment variables substituted
    """
    if isinstance(value, str):
        # Find all ${VAR_NAME} patterns
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)

        for var_name in matches:
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(f"Environment variable {var_name} not found")
            value = value.replace(f'${{{var_name}}}', env_value)

        return value

    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]

    else:
        return value


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file with environment variable substitution.

    Args:
        config_path: Path to config.yaml file. If None, uses default location.

    Returns:
        Config object with all settings
    """
    if config_path is None:
        # Default to config.yaml in same directory as this file
        config_path = Path(__file__).parent / 'config.yaml'

    # Load YAML
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # Substitute environment variables
    config_dict = substitute_env_vars(raw_config)

    # Build config objects - handle oanda credentials based on account_type
    account_type = config_dict['oanda']['account_type']
    oanda_creds = config_dict['oanda'][account_type]  # Get practice or live credentials
    oanda = OandaConfig(
        account_type=account_type,
        account_id=oanda_creds['account_id'],
        api_key=oanda_creds['api_key']
    )
    capital = CapitalConfig(**config_dict['capital'])
    strategy_15m = Strategy15mConfig(**config_dict['strategy_15m'])

    telegram = TelegramConfig(**config_dict['monitoring']['telegram'])
    logging_cfg = LoggingConfig(**config_dict['monitoring']['logging'])
    monitoring = MonitoringConfig(telegram=telegram, logging=logging_cfg)

    state = StateConfig(**config_dict['state'])
    models = ModelsConfig(**config_dict['models'])
    system = SystemConfig(**config_dict['system'])

    return Config(
        oanda=oanda,
        capital=capital,
        strategy_15m=strategy_15m,
        monitoring=monitoring,
        state=state,
        models=models,
        system=system
    )


def validate_config(config: Config) -> None:
    """
    Validate configuration values.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate OANDA settings
    if config.oanda.account_type not in ['practice', 'live']:
        raise ValueError(f"Invalid account_type: {config.oanda.account_type}")

    # Validate capital settings
    if config.capital.initial <= 0:
        raise ValueError("Initial capital must be positive")

    if not (0 < config.capital.max_drawdown < 0.5):
        raise ValueError("Max drawdown should be between 0% and 50%")

    # Validate strategy settings
    if not (0.5 <= config.strategy_15m.min_confidence <= 1.0):
        raise ValueError("Min confidence should be between 0.5 and 1.0")

    if not (0 < config.strategy_15m.position_size_pct <= 1.00):
        raise ValueError("Position size percent should be between 0% and 100%")

    if config.strategy_15m.max_positions_total <= 0:
        raise ValueError("Max positions must be positive")

    # Validate pairs
    valid_pairs = {'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY'}
    for pair in config.strategy_15m.pairs:
        if pair not in valid_pairs:
            raise ValueError(f"Invalid pair: {pair}")

    print("✓ Configuration validated successfully")


if __name__ == '__main__':
    # Test configuration loading
    print("Testing configuration loading...")
    config = load_config()
    validate_config(config)

    print("\nConfiguration loaded:")
    print(f"  Account type: {config.oanda.account_type}")
    print(f"  Initial capital: ${config.capital.initial}")
    print(f"  Strategy enabled: {config.strategy_15m.enabled}")
    print(f"  Pairs: {', '.join(config.strategy_15m.pairs)}")
    print(f"  Telegram enabled: {config.monitoring.telegram.enabled}")
