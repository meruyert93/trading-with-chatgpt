from venv import logger
import pandas as pd
import yfinance as yf

from trading_script_eur import _FX_CACHE, FX_PAIR, last_trading_date


def get_usd_eur_rate(date: pd.Timestamp | None = None) -> float:
    """
    Fetch USD/EUR exchange rate for a given date.
    Returns the rate to convert USD to EUR (multiply USD by this rate to get EUR).

    For EUR/USD quote (e.g., 1.08), USD to EUR conversion is 1/rate.
    Example: EUR/USD = 1.08 means 1 EUR = 1.08 USD, so 1 USD = 1/1.08 EUR ≈ 0.926 EUR
    """
    if date is None:
        date = last_trading_date()

    date_str = date.date().isoformat()

    # Check cache
    if date_str in _FX_CACHE:
        logger.debug("Using cached FX rate for %s: %.4f", date_str, _FX_CACHE[date_str])
        return _FX_CACHE[date_str]

    try:
        # Fetch EUR/USD rate from Yahoo Finance
        logger.info("Fetching USD/EUR exchange rate for %s", date_str)
        start = date - pd.Timedelta(days=1)  # Look back a day to ensure we get data
        end = date + pd.Timedelta(days=1)

        fx_data = yf.download(FX_PAIR, start=start, end=end, progress=False)

        if fx_data.empty:
            logger.warning("No FX data available for %s, using default rate 0.92", date_str)
            return 0.92  # Fallback rate if no data

        # Get the close price for the date (EUR/USD rate)
        
        print("fx_data:", fx_data)
        eur_usd_rate = float(fx_data['Close'].iloc[-1])
        logger.info("Fetched EUR/USD rate for %s: %.4f", date_str, eur_usd_rate)

        # Convert to USD/EUR rate (inverse)
        usd_eur_rate = 1.0 / eur_usd_rate

        _FX_CACHE[date_str] = usd_eur_rate
        logger.info("EUR/USD rate: %.4f, USD/EUR rate: %.4f", eur_usd_rate, usd_eur_rate)

        return usd_eur_rate

    except Exception as e:
        logger.error("Failed to fetch FX rate: %s. Using default rate 0.92", e)
        return 0.92  # Approximate fallback rate
      
def main():
    # Example usage
    rate = get_usd_eur_rate()
    print(f"Current USD/EUR rate: {rate:.4f}")
  
if __name__ == "__main__":
    main()