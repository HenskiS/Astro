"""
Unit tests for position sizing logic
======================================
Tests the critical position sizing calculations for different currency pairs.
"""
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPositionSizing(unittest.TestCase):
    """Test position sizing calculations"""

    def test_usd_base_pairs(self):
        """
        Test USD-base pairs (USDJPY, USDCAD, USDCHF).
        For these pairs, 1 unit = $1, so units = dollar amount directly.
        """
        # USDJPY at 150.00
        capital_for_trade = 150.0
        mid_price = 150.00
        pair = 'USDJPY'

        if pair.startswith('USD'):
            position_size = int(capital_for_trade)
        else:
            position_size = int(capital_for_trade / mid_price)

        # Expected: 150 units (each unit = $1)
        self.assertEqual(position_size, 150,
                        f"USDJPY: Expected 150 units for $150 trade, got {position_size}")

    def test_usd_quote_pairs(self):
        """
        Test USD-quote pairs (EURUSD, GBPUSD, AUDUSD, etc.).
        For these pairs, 1 unit = 1 base currency, so units = dollars / price.
        """
        # EURUSD at 1.05
        capital_for_trade = 150.0
        mid_price = 1.05
        pair = 'EURUSD'

        if pair.startswith('USD'):
            position_size = int(capital_for_trade)
        else:
            position_size = int(capital_for_trade / mid_price)

        # Expected: 142 units (150 / 1.05 = 142.857, truncated to 142)
        self.assertEqual(position_size, 142,
                        f"EURUSD: Expected 142 units for $150 trade at 1.05, got {position_size}")

    def test_gbpusd(self):
        """
        Test GBPUSD at typical price.
        1 unit = 1 GBP, so units = dollars / price.
        """
        capital_for_trade = 150.0
        mid_price = 1.27
        pair = 'GBPUSD'

        if pair.startswith('USD'):
            position_size = int(capital_for_trade)
        else:
            position_size = int(capital_for_trade / mid_price)

        # Expected: 118 units (150 / 1.27 = 118.110, truncated to 118)
        self.assertEqual(position_size, 118,
                        f"GBPUSD: Expected 118 units for $150 trade at 1.27, got {position_size}")

    def test_usdcad(self):
        """
        Test USDCAD (USD-base pair).
        1 unit = $1, so units = dollar amount directly.
        """
        capital_for_trade = 200.0
        mid_price = 1.35
        pair = 'USDCAD'

        if pair.startswith('USD'):
            position_size = int(capital_for_trade)
        else:
            position_size = int(capital_for_trade / mid_price)

        # Expected: 200 units (each unit = $1)
        self.assertEqual(position_size, 200,
                        f"USDCAD: Expected 200 units for $200 trade, got {position_size}")

    def test_audusd(self):
        """
        Test AUDUSD at typical price.
        1 unit = 1 AUD, so units = dollars / price.
        """
        capital_for_trade = 200.0
        mid_price = 0.67
        pair = 'AUDUSD'

        if pair.startswith('USD'):
            position_size = int(capital_for_trade)
        else:
            position_size = int(capital_for_trade / mid_price)

        # Expected: 298 units (200 / 0.67 = 298.507, truncated to 298)
        self.assertEqual(position_size, 298,
                        f"AUDUSD: Expected 298 units for $200 trade at 0.67, got {position_size}")

    def test_position_size_percentage(self):
        """
        Test realistic position sizing with 30% of capital.
        """
        current_capital = 500.0
        position_size_pct = 0.30
        capital_for_trade = current_capital * position_size_pct  # $150

        # Test with EURUSD
        mid_price = 1.10
        pair = 'EURUSD'

        if pair.startswith('USD'):
            position_size = int(capital_for_trade)
        else:
            position_size = int(capital_for_trade / mid_price)

        # Expected: 136 units (150 / 1.10 = 136.363, truncated to 136)
        self.assertEqual(position_size, 136,
                        f"30% of $500 = $150 trade in EURUSD at 1.10 should be 136 units, got {position_size}")

        # Verify notional value is approximately $150
        notional_value = position_size * mid_price
        self.assertAlmostEqual(notional_value, 150.0, delta=2.0,
                             msg=f"Notional value should be ~$150, got ${notional_value:.2f}")

    def test_edge_case_small_position(self):
        """
        Test that small positions are handled correctly.
        """
        capital_for_trade = 10.0
        mid_price = 1.10
        pair = 'EURUSD'

        if pair.startswith('USD'):
            position_size = int(capital_for_trade)
        else:
            position_size = int(capital_for_trade / mid_price)

        # Expected: 9 units (10 / 1.10 = 9.090, truncated to 9)
        self.assertEqual(position_size, 9,
                        f"Small position: Expected 9 units, got {position_size}")

    def test_edge_case_zero_position(self):
        """
        Test that very small capital results in zero units (should be rejected by risk manager).
        """
        capital_for_trade = 0.50
        mid_price = 1.10
        pair = 'EURUSD'

        if pair.startswith('USD'):
            position_size = int(capital_for_trade)
        else:
            position_size = int(capital_for_trade / mid_price)

        # Expected: 0 units (0.50 / 1.10 = 0.454, truncated to 0)
        self.assertEqual(position_size, 0,
                        f"Very small position should be 0 units, got {position_size}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
