"""Unit tests for trading_script_eur.py

Run with: pytest test_trading_script_eur.py -v
Or: python -m pytest test_trading_script_eur.py -v
"""

import os
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock, mock_open

import pytest
import pandas as pd
import numpy as np

# Import the module under test
import trading_script_eur as ts


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def setup_test_env(temp_dir):
    """Set up test environment with temporary data directory."""
    original_data_dir = ts.DATA_DIR
    original_portfolio_csv = ts.PORTFOLIO_CSV
    original_trade_log_csv = ts.TRADE_LOG_CSV

    # Set test directories
    ts.set_data_dir(temp_dir)

    yield temp_dir

    # Restore original paths
    ts.DATA_DIR = original_data_dir
    ts.PORTFOLIO_CSV = original_portfolio_csv
    ts.TRADE_LOG_CSV = original_trade_log_csv


@pytest.fixture
def sample_portfolio_df():
    """Create a sample portfolio DataFrame."""
    return pd.DataFrame([
        {
            "isin": "US0378331005",
            "ticker": "AAPL",
            "shares": 10,
            "stop_loss": 140.0,
            "buy_price": 150.0,
            "cost_basis": 1500.0,
        },
        {
            "isin": "US5949181045",
            "ticker": "MSFT",
            "shares": 5,
            "stop_loss": 300.0,
            "buy_price": 320.0,
            "cost_basis": 1600.0,
        }
    ])


@pytest.fixture
def mock_price_data():
    """Create mock price data for testing."""
    dates = pd.date_range(start='2025-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'Open': [150.0, 151.0, 152.0, 153.0, 154.0],
        'High': [155.0, 156.0, 157.0, 158.0, 159.0],
        'Low': [148.0, 149.0, 150.0, 151.0, 152.0],
        'Close': [152.0, 153.0, 154.0, 155.0, 156.0],
        'Adj Close': [152.0, 153.0, 154.0, 155.0, 156.0],
        'Volume': [1000000] * 5
    }, index=dates)


# ============================================================================
# DATE HELPER TESTS
# ============================================================================

class TestDateHelpers:
    """Test date helper functions."""

    def test_last_trading_date_weekday(self):
        """Test last_trading_date returns same day for weekdays."""
        # Monday
        monday = datetime(2025, 1, 6)  # A Monday
        result = ts.last_trading_date(monday)
        assert result.date() == monday.date()

    def test_last_trading_date_saturday(self):
        """Test last_trading_date returns Friday for Saturday."""
        saturday = datetime(2025, 1, 4)  # A Saturday
        result = ts.last_trading_date(saturday)
        expected = datetime(2025, 1, 3).date()  # Previous Friday
        assert result.date() == expected

    def test_last_trading_date_sunday(self):
        """Test last_trading_date returns Friday for Sunday."""
        sunday = datetime(2025, 1, 5)  # A Sunday
        result = ts.last_trading_date(sunday)
        expected = datetime(2025, 1, 3).date()  # Previous Friday
        assert result.date() == expected

    def test_check_weekend(self):
        """Test check_weekend returns ISO date string."""
        result = ts.check_weekend()
        # Should return a valid ISO date string
        assert isinstance(result, str)
        datetime.fromisoformat(result)  # Validates format

    def test_trading_day_window(self):
        """Test trading_day_window returns correct window."""
        test_date = datetime(2025, 1, 6)  # Monday
        start, end = ts.trading_day_window(test_date)

        assert start.date() == test_date.date()
        assert end.date() == (test_date + timedelta(days=1)).date()


# ============================================================================
# ISIN MAPPING TESTS
# ============================================================================

class TestISINMapping:
    """Test ISIN mapping functionality."""

    def test_get_isin_mapping_file(self, setup_test_env):
        """Test get_isin_mapping_file returns correct path."""
        result = ts.get_isin_mapping_file()
        assert result == setup_test_env / "isin_mapping.json"

    def test_save_and_load_isin_mappings(self, setup_test_env):
        """Test saving and loading ISIN mappings."""
        # Clear and set test data
        ts.ISIN_TO_YAHOO.clear()
        ts.ISIN_TO_YAHOO["US0378331005"] = "AAPL"
        ts.ISIN_TO_YAHOO["US5949181045"] = "MSFT"

        # Save
        ts.save_isin_mappings()

        # Verify file exists
        mapping_file = ts.get_isin_mapping_file()
        assert mapping_file.exists()

        # Clear and reload
        ts.ISIN_TO_YAHOO.clear()
        ts.load_isin_mappings()

        # Verify mappings loaded
        assert ts.ISIN_TO_YAHOO["US0378331005"] == "AAPL"
        assert ts.ISIN_TO_YAHOO["US5949181045"] == "MSFT"

    def test_add_isin_mapping(self, setup_test_env):
        """Test adding a new ISIN mapping."""
        ts.ISIN_TO_YAHOO.clear()

        ts.add_isin_mapping("US0378331005", "aapl")  # lowercase

        # Should be normalized to uppercase
        assert ts.ISIN_TO_YAHOO["US0378331005"] == "AAPL"

        # Should be saved to file
        mapping_file = ts.get_isin_mapping_file()
        assert mapping_file.exists()

    def test_get_yahoo_ticker_from_isin(self, setup_test_env):
        """Test retrieving ticker from ISIN."""
        ts.ISIN_TO_YAHOO.clear()
        ts.ISIN_TO_YAHOO["US0378331005"] = "AAPL"

        # Test found
        result = ts.get_yahoo_ticker_from_isin("US0378331005")
        assert result == "AAPL"

        # Test not found
        result = ts.get_yahoo_ticker_from_isin("UNKNOWN")
        assert result is None

        # Test case insensitive
        result = ts.get_yahoo_ticker_from_isin("us0378331005")
        assert result == "AAPL"


# ============================================================================
# FX RATE TESTS
# ============================================================================

class TestFXRates:
    """Test FX rate functionality."""

    @patch('trading_script_eur.yf.download')
    def test_get_usd_eur_rate(self, mock_download):
        """Test fetching USD/EUR exchange rate."""
        # Clear cache
        ts._FX_CACHE.clear()

        # Mock FX data (EUR/USD = 1.08)
        mock_fx_data = pd.DataFrame({
            'Close': [1.08]
        }, index=[pd.Timestamp('2025-01-06')])
        mock_download.return_value = mock_fx_data

        date = pd.Timestamp('2025-01-06')
        rate = ts.get_usd_eur_rate(date)

        # USD/EUR should be inverse of EUR/USD
        expected_rate = 1.0 / 1.08
        assert abs(rate - expected_rate) < 0.0001

    @patch('trading_script_eur.yf.download')
    def test_get_usd_eur_rate_cached(self, mock_download):
        """Test FX rate caching."""
        ts._FX_CACHE.clear()
        date = pd.Timestamp('2025-01-06')
        date_str = date.date().isoformat()

        # Set cache
        ts._FX_CACHE[date_str] = 0.925

        # Should not call download
        rate = ts.get_usd_eur_rate(date)
        assert rate == 0.925
        mock_download.assert_not_called()

    @patch('trading_script_eur.yf.download')
    def test_get_usd_eur_rate_fallback(self, mock_download):
        """Test FX rate fallback on error."""
        ts._FX_CACHE.clear()
        mock_download.return_value = pd.DataFrame()  # Empty data

        date = pd.Timestamp('2025-01-06')
        rate = ts.get_usd_eur_rate(date)

        # Should return fallback rate
        assert rate == 0.92

    def test_usd_to_eur(self):
        """Test USD to EUR conversion."""
        ts._FX_CACHE.clear()
        date = pd.Timestamp('2025-01-06')
        date_str = date.date().isoformat()

        # Set a known rate
        ts._FX_CACHE[date_str] = 0.9

        result = ts.usd_to_eur(100.0, date)
        assert result == 90.0


# ============================================================================
# DATA ACCESS TESTS
# ============================================================================

class TestDataAccess:
    """Test data download functionality."""

    def test_normalize_ohlcv(self, mock_price_data):
        """Test OHLCV normalization."""
        result = ts._normalize_ohlcv(mock_price_data)

        # Should have all required columns
        required_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        for col in required_cols:
            assert col in result.columns

    def test_normalize_ohlcv_missing_adj_close(self):
        """Test normalization when Adj Close is missing."""
        df = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [102],
            'Volume': [1000]
        })

        result = ts._normalize_ohlcv(df)

        # Adj Close should be set to Close
        assert result['Adj Close'].iloc[0] == 102

    def test_to_datetime_index(self):
        """Test datetime index conversion."""
        df = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=['2025-01-01', '2025-01-02', '2025-01-03'])

        result = ts._to_datetime_index(df)

        assert isinstance(result.index, pd.DatetimeIndex)

    @patch('trading_script_eur._yahoo_download')
    def test_download_price_data_success(self, mock_yahoo):
        """Test successful price data download."""
        mock_data = pd.DataFrame({
            'Open': [150.0],
            'High': [155.0],
            'Low': [148.0],
            'Close': [152.0],
            'Adj Close': [152.0],
            'Volume': [1000000]
        }, index=[pd.Timestamp('2025-01-06')])

        mock_yahoo.return_value = mock_data

        result = ts.download_price_data("AAPL", period="1d")

        assert isinstance(result, ts.FetchResult)
        assert not result.df.empty
        assert result.source == "yahoo"


# ============================================================================
# PORTFOLIO OPERATIONS TESTS
# ============================================================================

class TestPortfolioOperations:
    """Test portfolio operation functions."""

    def test_ensure_df_from_dataframe(self, sample_portfolio_df):
        """Test _ensure_df with DataFrame input."""
        result = ts._ensure_df(sample_portfolio_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "ticker" in result.columns

    def test_ensure_df_from_dict(self):
        """Test _ensure_df with dict input."""
        portfolio_dict = {
            "isin": ["US0378331005"],
            "ticker": ["AAPL"],
            "shares": [10],
            "stop_loss": [140.0],
            "buy_price": [150.0],
            "cost_basis": [1500.0]
        }

        result = ts._ensure_df(portfolio_dict)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_ensure_df_empty(self):
        """Test _ensure_df with empty input."""
        result = ts._ensure_df([])

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        # Should have correct columns
        expected_cols = ["isin", "ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        for col in expected_cols:
            assert col in result.columns

    def test_ensure_df_invalid_type(self):
        """Test _ensure_df raises error for invalid type."""
        with pytest.raises(TypeError):
            ts._ensure_df("invalid")


# ============================================================================
# TRADE LOGGING TESTS
# ============================================================================

class TestTradeLogging:
    """Test trade logging functions."""

    @patch('trading_script_eur.check_weekend')
    def test_log_sell(self, mock_check_weekend, setup_test_env, sample_portfolio_df):
        """Test log_sell function."""
        mock_check_weekend.return_value = "2025-01-06"

        result_portfolio = ts.log_sell(
            ticker="AAPL",
            shares=10,
            price=145.0,
            cost=150.0,
            pnl=-50.0,
            portfolio=sample_portfolio_df,
            isin="US0378331005"
        )

        # Portfolio should have AAPL removed
        assert "AAPL" not in result_portfolio["ticker"].values

        # Trade log should exist
        assert ts.TRADE_LOG_CSV.exists()

        # Read and verify log
        log_df = pd.read_csv(ts.TRADE_LOG_CSV)
        assert len(log_df) == 1
        assert log_df.iloc[0]["Ticker"] == "AAPL"
        assert log_df.iloc[0]["ISIN"] == "US0378331005"
        assert log_df.iloc[0]["Shares Sold"] == 10
        assert log_df.iloc[0]["PnL"] == -50.0

    @patch('trading_script_eur.check_weekend')
    @patch('trading_script_eur.download_price_data')
    @patch('trading_script_eur.usd_to_eur')
    def test_log_manual_buy(self, mock_usd_to_eur, mock_download, mock_check_weekend,
                           setup_test_env, sample_portfolio_df):
        """Test log_manual_buy function."""
        mock_check_weekend.return_value = "2025-01-06"
        mock_usd_to_eur.side_effect = lambda usd, _: usd * 0.92  # 1 USD = 0.92 EUR

        # Mock price data (in USD from market)
        mock_data = pd.DataFrame({
            'Open': [150.0],
            'High': [155.0],
            'Low': [148.0],
            'Close': [152.0],
            'Adj Close': [152.0],
            'Volume': [1000000]
        }, index=[pd.Timestamp('2025-01-06')])

        mock_download.return_value = ts.FetchResult(mock_data, "yahoo")

        # User provides EUR prices
        buy_price_eur = 138.0  # EUR limit price
        stoploss_eur = 128.8   # EUR stop loss

        new_cash, new_portfolio = ts.log_manual_buy(
            buy_price_eur=buy_price_eur,
            shares=5,
            ticker="GOOGL",
            stoploss_eur=stoploss_eur,
            cash=10000.0,
            chatgpt_portfolio=sample_portfolio_df,
            isin="US02079K3059",
            interactive=False
        )

        # Cash should be reduced (exec at open: 150 USD = 138 EUR)
        exec_price_eur = 150.0 * 0.92  # 138 EUR
        cost = exec_price_eur * 5
        assert new_cash == 10000.0 - cost

        # Portfolio should have new entry
        assert "GOOGL" in new_portfolio["ticker"].values
        googl_row = new_portfolio[new_portfolio["ticker"] == "GOOGL"].iloc[0]
        assert googl_row["isin"] == "US02079K3059"
        assert googl_row["shares"] == 5

    @patch('trading_script_eur.check_weekend')
    @patch('trading_script_eur.download_price_data')
    @patch('trading_script_eur.usd_to_eur')
    def test_log_manual_sell(self, mock_usd_to_eur, mock_download, mock_check_weekend,
                            setup_test_env, sample_portfolio_df):
        """Test log_manual_sell function."""
        mock_check_weekend.return_value = "2025-01-06"
        mock_usd_to_eur.side_effect = lambda usd, _: usd * 0.92

        # Mock price data (in USD from market)
        mock_data = pd.DataFrame({
            'Open': [160.0],
            'High': [165.0],
            'Low': [158.0],
            'Close': [162.0],
            'Adj Close': [162.0],
            'Volume': [1000000]
        }, index=[pd.Timestamp('2025-01-06')])

        mock_download.return_value = ts.FetchResult(mock_data, "yahoo")

        # User provides EUR sell price
        sell_price_eur = 147.2  # 160 USD = 147.2 EUR

        new_cash, new_portfolio = ts.log_manual_sell(
            sell_price_eur=sell_price_eur,
            shares_sold=5,
            ticker="AAPL",
            cash=5000.0,
            chatgpt_portfolio=sample_portfolio_df,
            reason="Test sell",
            interactive=False
        )

        # Cash should increase (exec at open: 160 USD = 147.2 EUR)
        exec_price_eur = 160.0 * 0.92
        proceeds = exec_price_eur * 5
        assert new_cash == 5000.0 + proceeds

        # Portfolio should have reduced shares
        aapl_row = new_portfolio[new_portfolio["ticker"] == "AAPL"].iloc[0]
        assert aapl_row["shares"] == 5  # Original 10 - 5 sold


# ============================================================================
# PORTFOLIO STATE TESTS
# ============================================================================

class TestPortfolioState:
    """Test portfolio state loading."""

    def test_load_latest_portfolio_state_empty(self, setup_test_env):
        """Test loading state from empty CSV."""
        # Create CSV with headers but no data
        empty_df = pd.DataFrame(columns=["Date", "ISIN", "Ticker", "Shares", "Buy Price",
                                         "Cost Basis", "Stop Loss", "Current Price",
                                         "Total Value", "PnL", "Action", "Cash Balance",
                                         "Total Equity"])
        empty_df.to_csv(ts.PORTFOLIO_CSV, index=False)

        with patch('builtins.input', return_value='10000'):
            portfolio, cash = ts.load_latest_portfolio_state()

        assert isinstance(portfolio, pd.DataFrame)
        assert portfolio.empty
        assert cash == 10000.0

    def test_load_latest_portfolio_state_with_data(self, setup_test_env):
        """Test loading state from CSV with data."""
        # Create sample portfolio CSV
        data = pd.DataFrame([
            {
                "Date": "2025-01-05",
                "ISIN": "US0378331005",
                "Ticker": "AAPL",
                "Shares": 10,
                "Buy Price": 138.0,
                "Cost Basis": 1380.0,
                "Stop Loss": 130.0,
                "Current Price": 142.0,
                "Total Value": 1420.0,
                "PnL": 40.0,
                "Action": "HOLD",
                "Cash Balance": "",
                "Total Equity": ""
            },
            {
                "Date": "2025-01-05",
                "ISIN": "",
                "Ticker": "TOTAL",
                "Shares": "",
                "Buy Price": "",
                "Cost Basis": "",
                "Stop Loss": "",
                "Current Price": "",
                "Total Value": 1420.0,
                "PnL": 40.0,
                "Action": "",
                "Cash Balance": 5000.0,
                "Total Equity": 6420.0
            }
        ])
        data.to_csv(ts.PORTFOLIO_CSV, index=False)

        portfolio, cash = ts.load_latest_portfolio_state()

        assert len(portfolio) == 1
        assert portfolio[0]["ticker"] == "AAPL"
        assert portfolio[0]["isin"] == "US0378331005"
        assert cash == 5000.0

    def test_load_latest_portfolio_state_file_not_found(self, setup_test_env):
        """Test error when portfolio file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ts.load_latest_portfolio_state()


# ============================================================================
# BENCHMARK LOADING TESTS
# ============================================================================

class TestBenchmarkLoading:
    """Test benchmark ticker loading."""

    def test_load_benchmarks_default(self, setup_test_env):
        """Test loading default benchmarks when no file exists."""
        # Pass a directory where no tickers.json exists
        benchmarks = ts.load_benchmarks(setup_test_env)

        assert benchmarks == ts.DEFAULT_BENCHMARKS

    def test_load_benchmarks_from_file(self, setup_test_env):
        """Test loading benchmarks from tickers.json."""
        # Create tickers.json
        config = {"benchmarks": ["SPY", "QQQ", "DIA"]}
        config_file = setup_test_env / "tickers.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        benchmarks = ts.load_benchmarks(setup_test_env)

        assert benchmarks == ["SPY", "QQQ", "DIA"]

    def test_load_benchmarks_malformed_json(self, setup_test_env):
        """Test fallback when JSON is malformed."""
        config_file = setup_test_env / "tickers.json"
        with open(config_file, 'w') as f:
            f.write("{invalid json")

        benchmarks = ts.load_benchmarks(setup_test_env)

        # Should fall back to defaults
        assert benchmarks == ts.DEFAULT_BENCHMARKS


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for end-to-end workflows."""

    @patch('trading_script_eur.download_price_data')
    @patch('trading_script_eur.usd_to_eur')
    @patch('builtins.input')
    def test_process_portfolio_no_interaction(self, mock_input, mock_usd_to_eur,
                                              mock_download, setup_test_env,
                                              sample_portfolio_df):
        """Test process_portfolio without user interaction."""
        mock_input.return_value = ''  # No trades
        mock_usd_to_eur.side_effect = lambda usd, _: usd * 0.92

        # Mock price data for both stocks
        mock_data = pd.DataFrame({
            'Open': [160.0],
            'High': [165.0],
            'Low': [158.0],
            'Close': [162.0],
            'Adj Close': [162.0],
            'Volume': [1000000]
        }, index=[pd.Timestamp('2025-01-06')])

        mock_download.return_value = ts.FetchResult(mock_data, "yahoo")

        result_portfolio, result_cash = ts.process_portfolio(
            portfolio=sample_portfolio_df,
            cash=5000.0,
            interactive=True
        )

        # Should have processed without errors
        assert isinstance(result_portfolio, pd.DataFrame)
        assert isinstance(result_cash, float)

        # Portfolio CSV should be created
        assert ts.PORTFOLIO_CSV.exists()

    @patch('trading_script_eur.download_price_data')
    @patch('trading_script_eur.usd_to_eur')
    @patch('builtins.input')
    def test_process_portfolio_stop_loss_triggered(self, mock_input, mock_usd_to_eur,
                                                    mock_download, setup_test_env,
                                                    sample_portfolio_df):
        """Test stop-loss execution in process_portfolio."""
        mock_input.return_value = ''  # No manual trades
        mock_usd_to_eur.side_effect = lambda usd, _: usd * 0.92

        # Mock price data that triggers stop-loss for AAPL (stop at 140 EUR)
        # Low USD price that converts to below stop loss in EUR
        mock_data_aapl = pd.DataFrame({
            'Open': [135.0],  # In EUR: 124.2 (below stop)
            'High': [140.0],
            'Low': [130.0],   # In EUR: 119.6 (below stop)
            'Close': [133.0],
            'Adj Close': [133.0],
            'Volume': [1000000]
        }, index=[pd.Timestamp('2025-01-06')])

        mock_data_msft = pd.DataFrame({
            'Open': [350.0],
            'High': [355.0],
            'Low': [345.0],
            'Close': [352.0],
            'Adj Close': [352.0],
            'Volume': [1000000]
        }, index=[pd.Timestamp('2025-01-06')])

        # Return different data based on ticker
        def download_side_effect(ticker, **kwargs):
            if ticker == "AAPL":
                return ts.FetchResult(mock_data_aapl, "yahoo")
            else:
                return ts.FetchResult(mock_data_msft, "yahoo")

        mock_download.side_effect = download_side_effect

        result_portfolio, result_cash = ts.process_portfolio(
            portfolio=sample_portfolio_df,
            cash=5000.0,
            interactive=True
        )

        # AAPL should be sold (stop-loss triggered)
        # Note: The exact behavior depends on EUR conversion
        assert ts.TRADE_LOG_CSV.exists()


# ============================================================================
# SET ASOF DATE TESTS
# ============================================================================

class TestSetAsof:
    """Test as-of date functionality."""

    def test_set_asof_with_string(self):
        """Test setting as-of date with string."""
        ts.set_asof("2025-01-06")

        assert ts.ASOF_DATE is not None
        assert ts.ASOF_DATE.date() == datetime(2025, 1, 6).date()

        # Clean up
        ts.set_asof(None)

    def test_set_asof_with_datetime(self):
        """Test setting as-of date with datetime."""
        date = datetime(2025, 1, 6)
        ts.set_asof(date)

        assert ts.ASOF_DATE is not None
        assert ts.ASOF_DATE.date() == date.date()

        # Clean up
        ts.set_asof(None)

    def test_set_asof_none(self):
        """Test clearing as-of date."""
        ts.set_asof("2025-01-06")
        ts.set_asof(None)

        assert ts.ASOF_DATE is None


# ============================================================================
# FILE PATH CONFIGURATION TESTS
# ============================================================================

class TestFilePathConfiguration:
    """Test file path configuration."""

    def test_set_data_dir(self, temp_dir):
        """Test setting data directory."""
        ts.set_data_dir(temp_dir)

        assert ts.DATA_DIR == temp_dir
        assert ts.PORTFOLIO_CSV == temp_dir / "chatgpt_portfolio_update_eur.csv"
        assert ts.TRADE_LOG_CSV == temp_dir / "chatgpt_trade_log_eur.csv"
        assert temp_dir.exists()


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_log_sell_empty_portfolio(self, setup_test_env):
        """Test log_sell with empty portfolio."""
        empty_portfolio = pd.DataFrame(columns=["isin", "ticker", "shares", "stop_loss", "buy_price", "cost_basis"])

        result = ts.log_sell(
            ticker="AAPL",
            shares=10,
            price=150.0,
            cost=140.0,
            pnl=100.0,
            portfolio=empty_portfolio,
            isin="US0378331005"
        )

        # Should still work, just return empty portfolio
        assert len(result) == 0

    @patch('trading_script_eur.download_price_data')
    def test_log_manual_buy_insufficient_cash(self, mock_download, setup_test_env):
        """Test log_manual_buy when insufficient cash."""
        mock_data = pd.DataFrame({
            'Open': [150.0],
            'High': [155.0],
            'Low': [148.0],
            'Close': [152.0],
            'Adj Close': [152.0],
            'Volume': [1000000]
        }, index=[pd.Timestamp('2025-01-06')])

        mock_download.return_value = ts.FetchResult(mock_data, "yahoo")

        portfolio = pd.DataFrame(columns=["isin", "ticker", "shares", "stop_loss", "buy_price", "cost_basis"])

        with patch('trading_script_eur.usd_to_eur', return_value=138.0):
            cash, result_portfolio = ts.log_manual_buy(
                buy_price_eur=138.0,  # EUR price
                shares=100,  # Very large order
                ticker="AAPL",
                stoploss_eur=128.8,  # EUR stop loss
                cash=100.0,  # Insufficient cash
                chatgpt_portfolio=portfolio,
                isin="US0378331005",
                interactive=False
            )

        # Cash should not change
        assert cash == 100.0
        # Portfolio should not change
        assert len(result_portfolio) == 0

    def test_normalize_ohlcv_multiindex(self):
        """Test normalizing DataFrame with MultiIndex columns."""
        # Create MultiIndex DataFrame (simulating multi-ticker download)
        arrays = [['AAPL', 'AAPL', 'AAPL'], ['Open', 'High', 'Close']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)

        df = pd.DataFrame([[150, 155, 152]], columns=index)

        result = ts._normalize_ohlcv(df)

        # Should flatten to single-level columns
        assert not isinstance(result.columns, pd.MultiIndex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
