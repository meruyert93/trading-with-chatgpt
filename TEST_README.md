# Trading Script EUR - Test Suite

Comprehensive unit tests for `trading_script_eur.py`.

## Installation

Install test dependencies:

```bash
pip install -r test_requirements.txt
```

## Running Tests

### Run all tests
```bash
pytest test_trading_script_eur.py -v
```

### Run with coverage report
```bash
pytest test_trading_script_eur.py -v --cov=trading_script_eur --cov-report=html
```

### Run specific test class
```bash
pytest test_trading_script_eur.py::TestISINMapping -v
```

### Run specific test
```bash
pytest test_trading_script_eur.py::TestISINMapping::test_save_and_load_isin_mappings -v
```

## Test Coverage

The test suite covers:

### 1. **Date Helper Functions** (`TestDateHelpers`)
- Last trading date calculation (weekdays, weekends)
- Weekend handling (Saturday → Friday, Sunday → Friday)
- Trading day window generation

### 2. **ISIN Mapping** (`TestISINMapping`)
- Saving and loading ISIN mappings from JSON
- Adding new ISIN mappings
- Retrieving tickers from ISIN codes
- Case-insensitive lookups
- Normalization (uppercase)

### 3. **FX Rate Functionality** (`TestFXRates`)
- USD/EUR exchange rate fetching
- Rate caching mechanism
- Fallback rates on error
- USD to EUR conversion

### 4. **Data Access Layer** (`TestDataAccess`)
- OHLCV data normalization
- DateTime index conversion
- Missing column handling (Adj Close)
- Price data download with fallbacks

### 5. **Portfolio Operations** (`TestPortfolioOperations`)
- DataFrame creation from various input types
- Empty portfolio initialization
- Column structure validation
- Type error handling

### 6. **Trade Logging** (`TestTradeLogging`)
- Automated sell logging (stop-loss)
- Manual buy logging (MOO and limit orders)
- Manual sell logging
- ISIN tracking in trade logs
- Cash and portfolio updates

### 7. **Portfolio State Management** (`TestPortfolioState`)
- Loading from empty CSV
- Loading from populated CSV
- ISIN column handling
- File not found error handling

### 8. **Benchmark Loading** (`TestBenchmarkLoading`)
- Default benchmarks when no config exists
- Loading from tickers.json
- Malformed JSON fallback

### 9. **Integration Tests** (`TestIntegration`)
- End-to-end portfolio processing
- Stop-loss trigger execution
- Multi-stock portfolio handling
- Interactive mode simulation

### 10. **Configuration** (`TestSetAsof`, `TestFilePathConfiguration`)
- As-of date setting (string, datetime, None)
- Data directory configuration
- File path updates

### 11. **Edge Cases** (`TestEdgeCases`)
- Empty portfolio handling
- Insufficient cash scenarios
- MultiIndex DataFrame normalization
- Error recovery

## Test Structure

Each test class is organized by functionality:

```
test_trading_script_eur.py
├── Fixtures (setup/teardown)
│   ├── temp_dir
│   ├── setup_test_env
│   ├── sample_portfolio_df
│   └── mock_price_data
├── TestDateHelpers
├── TestISINMapping
├── TestFXRates
├── TestDataAccess
├── TestPortfolioOperations
├── TestTradeLogging
├── TestPortfolioState
├── TestBenchmarkLoading
├── TestIntegration
├── TestSetAsof
├── TestFilePathConfiguration
└── TestEdgeCases
```

## Mocking Strategy

The tests use mocking to:
- Avoid actual API calls to Yahoo Finance
- Control date/time for reproducible tests
- Simulate user input in interactive mode
- Test error conditions

Key mocked functions:
- `yf.download` - Market data
- `download_price_data` - Price data fetching
- `usd_to_eur` - FX conversion
- `builtins.input` - User input
- `check_weekend` - Date normalization

## Fixtures

### `temp_dir`
Creates a temporary directory for test files, automatically cleaned up after tests.

### `setup_test_env`
Configures the test environment with temporary data directory and restores original paths after tests.

### `sample_portfolio_df`
Provides a sample portfolio DataFrame with AAPL and MSFT positions for testing.

### `mock_price_data`
Returns mock OHLCV data for a 5-day period.

## Best Practices

1. **Isolation**: Each test is independent and doesn't affect others
2. **Cleanup**: Temporary files are automatically removed
3. **Reproducibility**: Fixed dates and prices ensure consistent results
4. **Coverage**: Tests cover happy paths, edge cases, and error conditions
5. **Documentation**: Each test has a clear docstring explaining what it tests

## Continuous Integration

To run tests in CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r test_requirements.txt
    pytest test_trading_script_eur.py -v --cov=trading_script_eur
```

## Troubleshooting

### Import Errors
If you get import errors, ensure you're running from the project directory:
```bash
cd /path/to/ChatGPT-Micro-Cap-Experiment
pytest test_trading_script_eur.py -v
```

### Mock Issues
If mocks aren't working, check the patch target matches the actual import path in the module.

### Temporary File Cleanup
If tests fail and leave temporary files, they should be cleaned up automatically. If not:
```bash
# Manual cleanup
rm -rf /tmp/pytest-*
```

## Adding New Tests

When adding features to `trading_script_eur.py`:

1. Add corresponding test class or method
2. Use appropriate fixtures for setup
3. Mock external dependencies
4. Test both success and failure cases
5. Verify test coverage remains high

Example:
```python
class TestNewFeature:
    """Test new feature description."""

    def test_new_feature_success(self, setup_test_env):
        """Test successful execution of new feature."""
        # Arrange
        # Act
        # Assert
        pass

    def test_new_feature_error_handling(self, setup_test_env):
        """Test error handling in new feature."""
        # Arrange
        # Act
        # Assert
        pass
```

## Coverage Goals

Target: **>90% code coverage**

Check coverage:
```bash
pytest test_trading_script_eur.py --cov=trading_script_eur --cov-report=term-missing
```

View detailed HTML report:
```bash
pytest test_trading_script_eur.py --cov=trading_script_eur --cov-report=html
open htmlcov/index.html  # macOS
```
