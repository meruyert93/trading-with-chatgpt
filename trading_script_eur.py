"""EUR-adapted trading script for ChatGPT micro-cap portfolio (Trade Republic Germany).

This is a modified version of trading_script.py with the following EUR-specific features:
- Automatic USD/EUR exchange rate fetching
- All US stock prices converted to EUR
- Display values in EUR (€) instead of USD ($)
- Properly tracks costs and P&L in EUR accounting for FX conversion
- Compatible with Trade Republic Germany

Usage:
    python trading_script_eur.py --data-dir "Start Your Own"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Union
import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import json
import logging

from decimal import Decimal, InvalidOperation
import re
import sys

# Optional pandas-datareader import for Stooq access
try:
    import pandas_datareader.data as pdr
    _HAS_PDR = True
except Exception:
    _HAS_PDR = False

# -------- CURRENCY CONFIGURATION --------
CURRENCY = "EUR"
CURRENCY_SYMBOL = "€"
FX_PAIR = "EURUSD=X"  # Yahoo Finance ticker for EUR/USD rate

# Cache for FX rate (date -> rate)
_FX_CACHE: Dict[str, float] = {}

# -------- AS-OF override --------
ASOF_DATE: pd.Timestamp | None = None

def set_asof(date: str | datetime | pd.Timestamp | None) -> None:
    """Set a global 'as of' date so the script treats that day as 'today'. Use 'YYYY-MM-DD' format."""
    global ASOF_DATE
    if date is None:
        print("No prior date passed. Using today's date...")
        ASOF_DATE = None
        return
    ASOF_DATE = pd.Timestamp(date).normalize()
    pure_date = ASOF_DATE.date()

    print(f"Setting date as {pure_date}.")

# Allow env var override:  ASOF_DATE=YYYY-MM-DD python trading_script.py
_env_asof = os.environ.get("ASOF_DATE")
if _env_asof:
    set_asof(_env_asof)

def _effective_now() -> datetime:
    return (ASOF_DATE.to_pydatetime() if ASOF_DATE is not None else datetime.now())

# ------------------------------
# Globals / file locations
# ------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR  # Save files alongside this script by default
PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update_eur.csv"
TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log_eur.csv"
DEFAULT_BENCHMARKS = ["IWO", "XBI", "SPY", "IWM"]

# Set up logger for this module
logger = logging.getLogger(__name__)

# Log initial global state configuration (only when run as main script)
def _log_initial_state():
    """Log the initial global file path configuration."""
    logger.info("=== Trading Script Initial Configuration (EUR Version) ===")
    logger.info("Currency: %s (%s)", CURRENCY, CURRENCY_SYMBOL)
    logger.info("Script directory: %s", SCRIPT_DIR)
    logger.info("Data directory: %s", DATA_DIR)
    logger.info("Portfolio CSV: %s", PORTFOLIO_CSV)
    logger.info("Trade log CSV: %s", TRADE_LOG_CSV)
    logger.info("Default benchmarks: %s", DEFAULT_BENCHMARKS)
    logger.info("==============================================")

# ------------------------------
# FX Rate Fetching
# ------------------------------

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
        # Fetch EUR/USD rate from Yahoo Finance for the exact date
        logger.info("Fetching USD/EUR exchange rate for %s", date_str)
        print("Fetching USD/EUR exchange rate for", date_str)

        fx_data = yf.download(FX_PAIR, start=date_str, end=date_str, progress=False)

        if fx_data.empty:
            logger.warning("No FX data available for %s", date_str)
            print(f"\n⚠️  No FX data available for {date_str}")
            print("Please provide the EUR/USD exchange rate manually.")
            print("(You can find current rates at: https://www.google.com/search?q=eur+to+usd)")

            while True:
                try:
                    eur_usd_input = input(f"Enter EUR/USD rate for {date_str} (e.g., 1.08): ").strip()
                    eur_usd_rate = float(eur_usd_input)
                    if eur_usd_rate <= 0:
                        print("❌ Rate must be positive. Please try again.")
                        continue
                    usd_eur_rate = 1.0 / eur_usd_rate
                    _FX_CACHE[date_str] = usd_eur_rate
                    print(f"✓ Using EUR/USD rate: {eur_usd_rate:.4f} (USD/EUR: {usd_eur_rate:.4f})")
                    logger.info("Manual EUR/USD rate: %.4f, USD/EUR rate: %.4f", eur_usd_rate, usd_eur_rate)
                    return usd_eur_rate
                except ValueError:
                    print("❌ Invalid input. Please enter a numeric value (e.g., 1.08)")

        # Get the open price for the date (EUR/USD rate at market open)

        print("fx_data:", fx_data)
        eur_usd_rate = float(fx_data['Open'].iloc[-1])
        logger.info("Fetched EUR/USD rate for %s: %.4f", date_str, eur_usd_rate)
        print(f"Fetched EUR/USD rate for {date_str}: {eur_usd_rate:.4f}")

        # Convert to USD/EUR rate (inverse)
        usd_eur_rate = 1.0 / eur_usd_rate

        _FX_CACHE[date_str] = usd_eur_rate
        logger.info("EUR/USD rate: %.4f, USD/EUR rate: %.4f", eur_usd_rate, usd_eur_rate)
        print(f"EUR/USD rate: {eur_usd_rate:.4f}, USD/EUR rate: {usd_eur_rate:.4f}")

        return usd_eur_rate

    except Exception as e:
        logger.error("Failed to fetch FX rate: %s", e)
        print(f"\n⚠️  Error fetching FX rate: {e}")
        print("Please provide the EUR/USD exchange rate manually.")
        print("(You can find current rates at: https://www.google.com/search?q=eur+to+usd)")

        while True:
            try:
                eur_usd_input = input(f"Enter EUR/USD rate for {date_str} (e.g., 1.08): ").strip()
                eur_usd_rate = float(eur_usd_input)
                if eur_usd_rate <= 0:
                    print("❌ Rate must be positive. Please try again.")
                    continue
                usd_eur_rate = 1.0 / eur_usd_rate
                _FX_CACHE[date_str] = usd_eur_rate
                print(f"✓ Using EUR/USD rate: {eur_usd_rate:.4f} (USD/EUR: {usd_eur_rate:.4f})")
                logger.info("Manual EUR/USD rate: %.4f, USD/EUR rate: %.4f", eur_usd_rate, usd_eur_rate)
                return usd_eur_rate
            except ValueError:
                print("❌ Invalid input. Please enter a numeric value (e.g., 0.92)")

def usd_to_eur(usd_amount: float, date: pd.Timestamp | None = None) -> float:
    """Convert USD amount to EUR using the exchange rate for the given date."""
    rate = get_usd_eur_rate(date)
    return usd_amount * rate

# ------------------------------
# Equity parsing helper (CLI override)
# ------------------------------
def _normalize_number_string(s: str) -> str:
    """Remove commas/underscores/spaces and optional leading $ or €; preserve scientific notation."""
    s = str(s).strip()
    if s.startswith("$") or s.startswith("€"):
        s = s[1:]
    # remove commas, underscores, spaces
    s = re.sub(r"[,_\s]", "", s)
    return s

def parse_starting_equity(s: Union[str, float, Decimal]) -> Optional[Decimal]:
    """Return Decimal if s represents a positive number, otherwise None."""
    if isinstance(s, (float, Decimal)):
        try:
            d = Decimal(str(s))
        except Exception:
            return None
    else:
        try:
            norm = _normalize_number_string(str(s))
            if norm == "":
                return None
            d = Decimal(norm)
        except (InvalidOperation, ValueError):
            return None
    if d <= 0:
        return None
    return d


# ------------------------------
# ISIN to Yahoo Finance ticker mapping
# ------------------------------

ISIN_TO_YAHOO: Dict[str, str] = {}

def get_isin_mapping_file() -> Path:
    """Get the path to the ISIN mapping JSON file."""
    return DATA_DIR / "isin_mapping.json"

def load_isin_mappings() -> None:
    """Load ISIN→ticker mappings from JSON file on startup."""
    global ISIN_TO_YAHOO
    mapping_file = get_isin_mapping_file()

    if mapping_file.exists():
        logger.info("Reading JSON file: %s", mapping_file)
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                ISIN_TO_YAHOO.update(loaded)
            logger.info("Successfully loaded %d ISIN mappings from %s", len(loaded), mapping_file)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse ISIN mapping file %s: %s", mapping_file, e)
        except Exception as e:
            logger.warning("Failed to load ISIN mappings from %s: %s", mapping_file, e)
    else:
        logger.info("No existing ISIN mapping file found at %s", mapping_file)

def save_isin_mappings() -> None:
    """Save current ISIN→ticker mappings to JSON file."""
    mapping_file = get_isin_mapping_file()
    logger.info("Writing JSON file: %s", mapping_file)
    try:
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(ISIN_TO_YAHOO, f, indent=2, ensure_ascii=False)
        logger.info("Successfully saved %d ISIN mappings to %s", len(ISIN_TO_YAHOO), mapping_file)
    except Exception as e:
        logger.error("Failed to save ISIN mappings to %s: %s", mapping_file, e)

def add_isin_mapping(isin: str, ticker: str) -> None:
    """Add a new ISIN→ticker mapping and save immediately."""
    isin = isin.strip().upper()
    ticker = ticker.strip().upper()

    ISIN_TO_YAHOO[isin] = ticker
    save_isin_mappings()

    logger.info("Added ISIN mapping: %s → %s", isin, ticker)
    print(f"✓ Saved mapping: {isin} → {ticker}")

def get_yahoo_ticker_from_isin(isin: str) -> str | None:
    """Convert ISIN to Yahoo Finance ticker. Returns None if not found."""
    return ISIN_TO_YAHOO.get(isin.strip().upper())

# ------------------------------
# Configuration helpers — benchmark tickers (tickers.json)
# ------------------------------

def _read_json_file(path: Path) -> Optional[Dict]:
    """Read and parse JSON from `path`. Return dict on success, None if not found or invalid.

    - FileNotFoundError -> return None
    - JSON decode error -> log a warning and return None
    - Other IO errors -> log a warning and return None
    """
    try:
        logger.info("Reading JSON file: %s", path)
        with path.open("r", encoding="utf-8") as fh:
            result = json.load(fh)
            logger.info("Successfully read JSON file: %s", path)
            return result
    except FileNotFoundError:
        logger.info("JSON file not found: %s", path)
        return None
    except json.JSONDecodeError as exc:
        logger.warning("tickers.json present but malformed: %s -> %s. Falling back to defaults.", path, exc)
        return None
    except Exception as exc:
        logger.warning("Unable to read tickers.json (%s): %s. Falling back to defaults.", path, exc)
        return None

def load_benchmarks(script_dir: Path | None = None) -> List[str]:
    """Return a list of benchmark tickers.

    Looks for a `tickers.json` file in either:
      - script_dir (if provided) OR the module SCRIPT_DIR, and then
      - script_dir.parent (project root candidate).

    Expected schema:
      {"benchmarks": ["IWO", "XBI", "SPY", "IWM"]}

    Behavior:
    - If file missing or malformed -> return DEFAULT_BENCHMARKS copy.
    - If 'benchmarks' key missing or not a list -> log warning and return defaults.
    - Normalizes tickers (strip, upper) and preserves order while removing duplicates.
    """
    base = Path(script_dir) if script_dir else SCRIPT_DIR
    candidates = [base, base.parent]

    cfg = None
    cfg_path = None
    for c in candidates:
        p = (c / "tickers.json").resolve()
        data = _read_json_file(p)
        if data is not None:
            cfg = data
            cfg_path = p
            break

    if not cfg:
        return DEFAULT_BENCHMARKS.copy()

    benchmarks = cfg.get("benchmarks")
    if not isinstance(benchmarks, list):
        logger.warning("tickers.json at %s missing 'benchmarks' array. Falling back to defaults.", cfg_path)
        return DEFAULT_BENCHMARKS.copy()

    seen = set()
    result: list[str] = []
    for t in benchmarks:
        if not isinstance(t, str):
            continue
        up = t.strip().upper()
        if not up:
            continue
        if up not in seen:
            seen.add(up)
            result.append(up)

    return result if result else DEFAULT_BENCHMARKS.copy()


# ------------------------------
# Date helpers
# ------------------------------

def last_trading_date(today: datetime | None = None) -> pd.Timestamp:
    """Return last trading date (Mon–Fri), mapping Sat/Sun -> Fri."""
    dt = pd.Timestamp(today or _effective_now())
    if dt.weekday() == 5:  # Sat -> Fri
        friday_date = (dt - pd.Timedelta(days=1)).normalize()
        logger.info("Script running on Saturday - using Friday's data (%s) instead of today's date", friday_date.date())
        return friday_date
    if dt.weekday() == 6:  # Sun -> Fri
        friday_date = (dt - pd.Timedelta(days=2)).normalize()
        logger.info("Script running on Sunday - using Friday's data (%s) instead of today's date", friday_date.date())
        return friday_date
    return dt.normalize()

def check_weekend() -> str:
    """Backwards-compatible wrapper returning ISO date string for last trading day."""
    return last_trading_date().date().isoformat()

def trading_day_window(target: datetime | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) window for the last trading day (Fri on weekends)."""
    d = last_trading_date(target)
    return d, (d + pd.Timedelta(days=1))


# ------------------------------
# Data access layer
# ------------------------------

# Known Stooq symbol remaps for common indices
STOOQ_MAP = {
    "^GSPC": "^SPX",  # S&P 500
    "^DJI": "^DJI",   # Dow Jones
    "^IXIC": "^IXIC", # Nasdaq Composite
    # "^RUT": not on Stooq; keep Yahoo
}

# Symbols we should *not* attempt on Stooq
STOOQ_BLOCKLIST = {"^RUT"}


# ------------------------------
# Data access layer (UPDATED)
# ------------------------------

@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str  # "yahoo" | "stooq-pdr" | "stooq-csv" | "yahoo:<proxy>-proxy" | "empty"

def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten multiIndex frame so we can lazily lookup values by index.
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # If the second level is the same ticker for all cols, drop it
            if len(set(df.columns.get_level_values(1))) == 1:
                df = df.copy()
                df.columns = df.columns.get_level_values(0)
            else:
                # multiple tickers: flatten with join
                df = df.copy()
                df.columns = ["_".join(map(str, t)).strip("_") for t in df.columns.to_flat_index()]
        except Exception:
            df = df.copy()
            df.columns = ["_".join(map(str, t)).strip("_") for t in df.columns.to_flat_index()]

    # Ensure all expected columns exist
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return df[cols]

def _yahoo_download(ticker: str, **kwargs: Any) -> pd.DataFrame:
    """Call yfinance.download with a real UA and silence all chatter."""
    import io, logging
    from contextlib import redirect_stderr, redirect_stdout

    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                df = cast(pd.DataFrame, yf.download(ticker, **kwargs))
        except Exception:
            return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _stooq_csv_download(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch OHLCV from Stooq CSV endpoint (daily). Good for US tickers and many ETFs."""
    import requests, io
    if ticker in STOOQ_BLOCKLIST:
        return pd.DataFrame()
    t = STOOQ_MAP.get(ticker, ticker)

    # Stooq daily CSV: lowercase; equities/ETFs use .us, indices keep ^ prefix
    if not t.startswith("^"):
        sym = t.lower()
        if not sym.endswith(".us"):
            sym = f"{sym}.us"
    else:
        sym = t.lower()

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or not r.text.strip():
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # Filter to [start, end) (Stooq end is exclusive)
        df = df.loc[(df.index >= start.normalize()) & (df.index < end.normalize())]

        # Normalize to Yahoo-like schema
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    except Exception:
        return pd.DataFrame()

def _stooq_download(
    ticker: str,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
) -> pd.DataFrame:
    """Fetch OHLCV from Stooq via pandas-datareader; returns empty DF on failure."""
    if not _HAS_PDR or ticker in STOOQ_BLOCKLIST:
        return pd.DataFrame()

    t = STOOQ_MAP.get(ticker, ticker)
    if not t.startswith("^"):
        t = t.lower()

    try:
        # Ensure pdr is imported locally if not available globally
        if not _HAS_PDR:
            return pd.DataFrame()
        import pandas_datareader.data as pdr_local
        df = cast(pd.DataFrame, pdr_local.DataReader(t, "stooq", start=start, end=end))
        df.sort_index(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def _weekend_safe_range(period: str | None, start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Compute a concrete [start, end) window.
    - If explicit start/end provided: use them (add +1 day to end to make it exclusive).
    - If period is '1d': use the last trading day's [Fri, Sat) window on weekends.
    - If period like '2d'/'5d': build a window ending at the last trading day.
    """
    if start or end:
        end_ts = pd.Timestamp(end) if end else last_trading_date() + pd.Timedelta(days=1)
        start_ts = pd.Timestamp(start) if start else (end_ts - pd.Timedelta(days=5))
        return start_ts.normalize(), pd.Timestamp(end_ts).normalize()

    # No explicit dates; derive from period
    if isinstance(period, str) and period.endswith("d"):
        days = int(period[:-1])
    else:
        days = 1

    # Anchor to last trading day (Fri on Sun/Sat)
    end_trading = last_trading_date()
    start_ts = (end_trading - pd.Timedelta(days=days)).normalize()
    end_ts = (end_trading + pd.Timedelta(days=1)).normalize()
    return start_ts, end_ts

def download_price_data(ticker: str, **kwargs: Any) -> FetchResult:
    """
    Robust OHLCV fetch with multi-stage fallbacks:

    Order:
      1) Yahoo Finance via yfinance
      2) Stooq via pandas-datareader
      3) Stooq direct CSV
      4) Index proxies (e.g., ^GSPC->SPY, ^RUT->IWM) via Yahoo
    Returns a DataFrame with columns [Open, High, Low, Close, Adj Close, Volume].
    """
    # Pull out range args, compute a weekend-safe window
    period = kwargs.pop("period", None)
    start = kwargs.pop("start", None)
    end = kwargs.pop("end", None)
    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)

    s, e = _weekend_safe_range(period, start, end)

    # ---------- 1) Yahoo (date-bounded) ----------
    df_y = _yahoo_download(ticker, start=s, end=e, **kwargs)
    if isinstance(df_y, pd.DataFrame) and not df_y.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_y)), "yahoo")

    # ---------- 2) Stooq via pandas-datareader ----------
    df_s = _stooq_download(ticker, start=s, end=e)
    if isinstance(df_s, pd.DataFrame) and not df_s.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_s)), "stooq-pdr")

    # ---------- 3) Stooq direct CSV ----------
    df_csv = _stooq_csv_download(ticker, s, e)
    if isinstance(df_csv, pd.DataFrame) and not df_csv.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_csv)), "stooq-csv")

    # ---------- 4) Proxy indices if applicable ----------
    proxy_map = {"^GSPC": "SPY", "^RUT": "IWM"}
    proxy = proxy_map.get(ticker)
    if proxy:
        df_proxy = _yahoo_download(proxy, start=s, end=e, **kwargs)
        if isinstance(df_proxy, pd.DataFrame) and not df_proxy.empty:
            return FetchResult(_normalize_ohlcv(_to_datetime_index(df_proxy)), f"yahoo:{proxy}-proxy")

    # ---------- Nothing worked ----------
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    return FetchResult(empty, "empty")



# ------------------------------
# File path configuration
# ------------------------------

def set_data_dir(data_dir: Path) -> None:
    global DATA_DIR, PORTFOLIO_CSV, TRADE_LOG_CSV
    logger.info("Setting data directory: %s", data_dir)
    DATA_DIR = Path(data_dir)
    logger.debug("Creating data directory if it doesn't exist: %s", DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update_eur.csv"
    TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log_eur.csv"
    logger.info("Data directory configured - Portfolio CSV: %s, Trade Log CSV: %s", PORTFOLIO_CSV, TRADE_LOG_CSV)


# ------------------------------
# Portfolio operations
# ------------------------------

def _ensure_df(portfolio: pd.DataFrame | dict[str, list[object]] | list[dict[str, object]]) -> pd.DataFrame:
    if isinstance(portfolio, pd.DataFrame):
        return portfolio.copy()
    if isinstance(portfolio, (dict, list)):
        df = pd.DataFrame(portfolio)
        # Ensure proper columns exist even for empty DataFrames
        if df.empty:
            logger.debug("Creating empty portfolio DataFrame with proper column structure")
            df = pd.DataFrame(columns=["isin", "ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        return df
    raise TypeError("portfolio must be a DataFrame, dict, or list[dict]")

def process_portfolio(
    portfolio: pd.DataFrame | dict[str, list[object]] | list[dict[str, object]],
    cash: float,
    interactive: bool = True,
) -> tuple[pd.DataFrame, float]:
    today_iso = last_trading_date().date().isoformat()
    today_ts = last_trading_date()
    portfolio_df = _ensure_df(portfolio)

    results: list[dict[str, object]] = []
    total_value = 0.0
    total_pnl = 0.0

    # ------- Interactive trade entry (supports MOO) -------
    if interactive:
        while True:
            print(portfolio_df)
            action = input(
                f""" You have {CURRENCY_SYMBOL}{cash:.2f} in cash.
                    Would you like to log a manual trade? Enter 'b' for buy, 's' for sell, or press Enter to continue: """
            ).strip().lower()

            if action == "b":
                # Ask for ISIN first
                isin = input("Enter ISIN (e.g., US0378331005): ").strip().upper()

                # Check if we have a mapping
                ticker = get_yahoo_ticker_from_isin(isin)

                if ticker is None:
                    print(f"ISIN {isin} not found in mapping.")
                    ticker = input("Enter Yahoo Finance ticker (e.g., AAPL): ").strip().upper()

                    # Ask if they want to save this mapping
                    save_mapping = input(f"Save mapping {isin} → {ticker}? (y/n): ").strip().lower()
                    if save_mapping == 'y':
                        add_isin_mapping(isin, ticker)
                else:
                    print(f"Using ticker: {ticker} (from ISIN {isin})")

                order_type = input("Order type? 'm' = market-on-open, 'l' = limit: ").strip().lower()

                try:
                    shares = float(input("Enter number of shares: "))
                    if shares <= 0:
                        raise ValueError
                except ValueError:
                    print("Invalid share amount. Buy cancelled.")
                    continue

                if order_type == "m":
                    try:
                        stop_loss_eur = float(input("Enter stop loss in EUR (or 0 to skip): "))
                        if stop_loss_eur < 0:
                            raise ValueError
                    except ValueError:
                        print("Invalid stop loss. Buy cancelled.")
                        continue

                    s, e = trading_day_window()
                    fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
                    data = fetch.df
                    if data.empty:
                        print(f"MOO buy for {ticker} failed: no market data available (source={fetch.source}).")
                        continue

                    o_usd = float(data["Open"].iloc[-1]) if "Open" in data else float(data["Close"].iloc[-1])
                    exec_price_usd = round(o_usd, 2)

                    # Convert to EUR
                    exec_price_eur = usd_to_eur(exec_price_usd, today_ts)

                    notional = exec_price_eur * shares
                    if notional > cash:
                        print(f"MOO buy for {ticker} failed: cost {CURRENCY_SYMBOL}{notional:.2f} exceeds cash {CURRENCY_SYMBOL}{cash:.2f}.")
                        continue

                    log = {
                        "Date": today_iso,
                        "ISIN": isin,
                        "Ticker": ticker,
                        "Shares Bought": shares,
                        "Buy Price (EUR)": exec_price_eur,
                        "Buy Price (USD)": exec_price_usd,
                        "Cost Basis": notional,
                        "PnL": 0.0,
                        "Reason": "MANUAL BUY MOO - Filled",
                    }
                    # --- Manual BUY MOO logging ---
                    if os.path.exists(TRADE_LOG_CSV):
                        logger.info("Reading CSV file: %s", TRADE_LOG_CSV)
                        df_log = pd.read_csv(TRADE_LOG_CSV)
                        logger.info("Successfully read CSV file: %s", TRADE_LOG_CSV)
                        if df_log.empty:
                            df_log = pd.DataFrame([log])
                        else:
                            df_log = pd.concat([df_log, pd.DataFrame([log])], ignore_index=True)
                    else:
                        df_log = pd.DataFrame([log])
                    logger.info("Writing CSV file: %s", TRADE_LOG_CSV)
                    df_log.to_csv(TRADE_LOG_CSV, index=False)
                    logger.info("Successfully wrote CSV file: %s", TRADE_LOG_CSV)

                    rows = portfolio_df.loc[portfolio_df["ticker"].astype(str).str.upper() == ticker.upper()]
                    if rows.empty:
                        new_trade = {
                            "isin": isin,
                            "ticker": ticker,
                            "shares": float(shares),
                            "stop_loss": float(stop_loss_eur),
                            "buy_price": float(exec_price_eur),
                            "cost_basis": float(notional),
                        }
                        if portfolio_df.empty:
                            portfolio_df = pd.DataFrame([new_trade])
                        else:
                            portfolio_df = pd.concat([portfolio_df, pd.DataFrame([new_trade])], ignore_index=True)
                    else:
                        idx = rows.index[0]
                        cur_shares = float(portfolio_df.at[idx, "shares"])
                        cur_cost = float(portfolio_df.at[idx, "cost_basis"])
                        new_shares = cur_shares + float(shares)
                        new_cost = cur_cost + float(notional)
                        avg_price = new_cost / new_shares if new_shares else 0.0
                        portfolio_df.at[idx, "shares"] = new_shares
                        portfolio_df.at[idx, "cost_basis"] = new_cost
                        portfolio_df.at[idx, "buy_price"] = avg_price
                        portfolio_df.at[idx, "stop_loss"] = float(stop_loss_eur)
                        portfolio_df.at[idx, "isin"] = isin

                    cash -= notional
                    print(f"Manual BUY MOO for {ticker} filled at ${exec_price_usd:.2f} USD ({CURRENCY_SYMBOL}{exec_price_eur:.2f} EUR) ({fetch.source}).")
                    continue

                elif order_type == "l":
                    try:
                        buy_price_eur = float(input("Enter buy LIMIT price in EUR: "))
                        stop_loss_eur = float(input("Enter stop loss in EUR (or 0 to skip): "))
                        if buy_price_eur <= 0 or stop_loss_eur < 0:
                            raise ValueError
                    except ValueError:
                        print("Invalid input. Limit buy cancelled.")
                        continue

                    cash, portfolio_df = log_manual_buy(
                        buy_price_eur, shares, ticker, stop_loss_eur, cash, portfolio_df, isin=isin
                    )
                    continue
                else:
                    print("Unknown order type. Use 'm' or 'l'.")
                    continue

            if action == "s":
                try:
                    ticker = input("Enter ticker symbol: ").strip().upper()
                    sell_order_type = input("Order type? 'm' = market-on-open, 'l' = limit: ").strip().lower()
                    shares = float(input("Enter number of shares to sell: "))
                    if sell_order_type == 'l':
                        sell_price_eur = float(input("Enter sell LIMIT price in EUR: "))
                    elif sell_order_type == 'm':
                        # Get current price in USD and convert to EUR
                        s, e = trading_day_window()
                        fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
                        if not fetch.df.empty:
                            price_usd = float(fetch.df["Open"].iloc[-1]) if "Open" in fetch.df else float(fetch.df["Close"].iloc[-1])
                            sell_price_eur = usd_to_eur(price_usd, today_ts)
                        else:
                            print(f"Cannot get price for {ticker}")
                            continue
                    else:
                        print("Unknown order type. Use 'm' or 'l'.")
                        continue
                    if shares <= 0 or sell_price_eur <= 0:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Manual sell cancelled.")
                    continue

                cash, portfolio_df = log_manual_sell(
                    sell_price_eur, shares, ticker, cash, portfolio_df
                )
                continue

            break  # proceed to pricing

    # ------- Daily pricing + stop-loss execution -------
    s, e = trading_day_window()
    for _, stock in portfolio_df.iterrows():
        ticker = str(stock["ticker"]).upper()
        shares = int(stock["shares"]) if not pd.isna(stock["shares"]) else 0
        cost_eur = float(stock["buy_price"]) if not pd.isna(stock["buy_price"]) else 0.0
        cost_basis = float(stock["cost_basis"]) if not pd.isna(stock["cost_basis"]) else cost_eur * shares
        stop_eur = float(stock["stop_loss"]) if not pd.isna(stock["stop_loss"]) else 0.0

        fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
        data = fetch.df

        # Get ISIN from stock row (works for both dict and Series)
        isin_value = str(stock.get("isin", "")) if hasattr(stock, 'get') else (str(stock["isin"]) if "isin" in stock.index else "")

        if data.empty:
            print(f"No data for {ticker} (source={fetch.source}).")
            row = {
                "Date": today_iso, "ISIN": isin_value, "Ticker": ticker, "Shares": shares,
                "Buy Price": cost_eur, "Cost Basis": cost_basis, "Stop Loss": stop_eur,
                "Current Price": "", "Total Value": "", "PnL": "",
                "Action": "NO DATA", "Cash Balance": "", "Total Equity": "",
            }
            results.append(row)
            continue

        # Get USD prices and convert to EUR
        o_usd = float(data["Open"].iloc[-1]) if "Open" in data else np.nan
        h_usd = float(data["High"].iloc[-1])
        l_usd = float(data["Low"].iloc[-1])
        c_usd = float(data["Close"].iloc[-1])
        if np.isnan(o_usd):
            o_usd = c_usd

        # Convert to EUR
        o_eur = usd_to_eur(o_usd, today_ts)
        h_eur = usd_to_eur(h_usd, today_ts)
        l_eur = usd_to_eur(l_usd, today_ts)
        c_eur = usd_to_eur(c_usd, today_ts)

        if stop_eur and l_eur <= stop_eur:
            exec_price = round(o_eur if o_eur <= stop_eur else stop_eur, 2)
            value = round(exec_price * shares, 2)
            pnl = round((exec_price - cost_eur) * shares, 2)
            action = "SELL - Stop Loss Triggered"
            cash += value
            portfolio_df = log_sell(ticker, shares, exec_price, cost_eur, pnl, portfolio_df, isin=isin_value)
            row = {
                "Date": today_iso, "ISIN": isin_value, "Ticker": ticker, "Shares": shares,
                "Buy Price": cost_eur, "Cost Basis": cost_basis, "Stop Loss": stop_eur,
                "Current Price": exec_price, "Total Value": value, "PnL": pnl,
                "Action": action, "Cash Balance": "", "Total Equity": "",
            }
        else:
            price = round(c_eur, 2)
            value = round(price * shares, 2)
            pnl = round((price - cost_eur) * shares, 2)
            action = "HOLD"
            total_value += value
            total_pnl += pnl
            row = {
                "Date": today_iso, "ISIN": isin_value, "Ticker": ticker, "Shares": shares,
                "Buy Price": cost_eur, "Cost Basis": cost_basis, "Stop Loss": stop_eur,
                "Current Price": price, "Total Value": value, "PnL": pnl,
                "Action": action, "Cash Balance": "", "Total Equity": "",
            }

        results.append(row)

    total_row = {
        "Date": today_iso, "ISIN": "", "Ticker": "TOTAL", "Shares": "", "Buy Price": "",
        "Cost Basis": "", "Stop Loss": "", "Current Price": "",
        "Total Value": round(total_value, 2), "PnL": round(total_pnl, 2),
        "Action": "", "Cash Balance": round(cash, 2),
        "Total Equity": round(total_value + cash, 2),
    }
    results.append(total_row)

    df_out = pd.DataFrame(results)
    if PORTFOLIO_CSV.exists():
        logger.info("Reading CSV file: %s", PORTFOLIO_CSV)
        existing = pd.read_csv(PORTFOLIO_CSV)
        logger.info("Successfully read CSV file: %s", PORTFOLIO_CSV)
        existing = existing[existing["Date"] != str(today_iso)]
        print("Saving results to CSV...")
        df_out = pd.concat([existing, df_out], ignore_index=True)
    logger.info("Writing CSV file: %s", PORTFOLIO_CSV)
    df_out.to_csv(PORTFOLIO_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", PORTFOLIO_CSV)

    return portfolio_df, cash



# ------------------------------
# Trade logging
# ------------------------------

def log_sell(
    ticker: str,
    shares: float,
    price: float,
    cost: float,
    pnl: float,
    portfolio: pd.DataFrame,
    isin: str = "",
) -> pd.DataFrame:
    today = check_weekend()
    log = {
        "Date": today,
        "ISIN": isin,
        "Ticker": ticker,
        "Shares Sold": shares,
        "Sell Price (EUR)": price,
        "Cost Basis": cost,
        "PnL": pnl,
        "Reason": "AUTOMATED SELL - STOPLOSS TRIGGERED",
    }
    print(f"{ticker} stop loss was met. Selling all shares.")
    portfolio = portfolio[portfolio["ticker"] != ticker]

    if TRADE_LOG_CSV.exists():
        logger.info("Reading CSV file: %s", TRADE_LOG_CSV)
        df = pd.read_csv(TRADE_LOG_CSV)
        logger.info("Successfully read CSV file: %s", TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    logger.info("Writing CSV file: %s", TRADE_LOG_CSV)
    df.to_csv(TRADE_LOG_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", TRADE_LOG_CSV)
    return portfolio

def log_manual_buy(
    buy_price_eur: float,
    shares: float,
    ticker: str,
    stoploss_eur: float,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    isin: str | None = None,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    today = check_weekend()
    today_ts = last_trading_date()

    if interactive:
        check = input(
            f"You are placing a BUY LIMIT for {shares} {ticker} at €{buy_price_eur:.2f} EUR.\n"
            f"If this is a mistake, type '1' or, just hit Enter: "
        )
        if check == "1":
            print("Returning...")
            return cash, chatgpt_portfolio

    if not isinstance(chatgpt_portfolio, pd.DataFrame) or chatgpt_portfolio.empty:
        chatgpt_portfolio = pd.DataFrame(
            columns=["isin", "ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        )

    s, e = trading_day_window()
    fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
    data = fetch.df
    if data.empty:
        print(f"Manual buy for {ticker} failed: no market data available (source={fetch.source}).")
        return cash, chatgpt_portfolio

    # Get USD prices from market data
    o_usd = float(data["Open"].iloc[-1]) if "Open" in data else np.nan
    h_usd = float(data["High"].iloc[-1])
    l_usd = float(data["Low"].iloc[-1])
    c_usd = float(data["Close"].iloc[-1])

    if np.isnan(o_usd):
        o_usd = c_usd

    # Convert USD prices to EUR for comparison with user's EUR limit
    o_eur = usd_to_eur(o_usd, today_ts)
    h_eur = usd_to_eur(h_usd, today_ts)
    l_eur = usd_to_eur(l_usd, today_ts)

    # Check if limit order would be filled (in EUR)
    if o_eur <= buy_price_eur:
        exec_price_eur = o_eur
        exec_price_usd = o_usd
    elif l_eur <= buy_price_eur:
        exec_price_eur = buy_price_eur
        # Calculate equivalent USD price
        exec_price_usd = buy_price_eur / usd_to_eur(1.0, today_ts)
    else:
        print(f"Buy limit €{buy_price_eur:.2f} EUR for {ticker} not reached today (range €{l_eur:.2f}-€{h_eur:.2f} EUR). Order not filled.")
        return cash, chatgpt_portfolio

    cost_amt = exec_price_eur * shares
    if cost_amt > cash:
        print(f"Manual buy for {ticker} failed: cost {CURRENCY_SYMBOL}{cost_amt:.2f} exceeds cash balance {CURRENCY_SYMBOL}{cash:.2f}.")
        return cash, chatgpt_portfolio

    log = {
        "Date": today,
        "ISIN": isin or "",
        "Ticker": ticker,
        "Shares Bought": shares,
        "Buy Price (EUR)": exec_price_eur,
        "Buy Price (USD)": exec_price_usd,
        "Cost Basis": cost_amt,
        "PnL": 0.0,
        "Reason": "MANUAL BUY LIMIT - Filled",
    }
    if os.path.exists(TRADE_LOG_CSV):
        logger.info("Reading CSV file: %s", TRADE_LOG_CSV)
        df = pd.read_csv(TRADE_LOG_CSV)
        logger.info("Successfully read CSV file: %s", TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    logger.info("Writing CSV file: %s", TRADE_LOG_CSV)
    df.to_csv(TRADE_LOG_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", TRADE_LOG_CSV)

    rows = chatgpt_portfolio.loc[chatgpt_portfolio["ticker"].str.upper() == ticker.upper()]
    if rows.empty:
        if chatgpt_portfolio.empty:
            chatgpt_portfolio = pd.DataFrame([{
                "isin": isin,
                "ticker": ticker,
                "shares": float(shares),
                "stop_loss": float(stoploss_eur),
                "buy_price": float(exec_price_eur),
                "cost_basis": float(cost_amt),
            }])
        else:
            chatgpt_portfolio = pd.concat(
                [chatgpt_portfolio, pd.DataFrame([{
                    "isin": isin,
                    "ticker": ticker,
                    "shares": float(shares),
                    "stop_loss": float(stoploss_eur),
                    "buy_price": float(exec_price_eur),
                    "cost_basis": float(cost_amt),
                }])],
                ignore_index=True
            )
    else:
        idx = rows.index[0]
        cur_shares = float(chatgpt_portfolio.at[idx, "shares"])
        cur_cost = float(chatgpt_portfolio.at[idx, "cost_basis"])
        new_shares = cur_shares + float(shares)
        new_cost = cur_cost + float(cost_amt)
        chatgpt_portfolio.at[idx, "shares"] = new_shares
        chatgpt_portfolio.at[idx, "cost_basis"] = new_cost
        chatgpt_portfolio.at[idx, "buy_price"] = new_cost / new_shares if new_shares else 0.0
        chatgpt_portfolio.at[idx, "stop_loss"] = float(stoploss_eur)
        chatgpt_portfolio.at[idx, "isin"] = isin

    cash -= cost_amt
    print(f"Manual BUY LIMIT for {ticker} filled at ${exec_price_usd:.2f} USD ({CURRENCY_SYMBOL}{exec_price_eur:.2f} EUR) ({fetch.source}).")
    return cash, chatgpt_portfolio

def log_manual_sell(
    sell_price_eur: float,
    shares_sold: float,
    ticker: str,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    reason: str | None = None,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    today = check_weekend()
    today_ts = last_trading_date()

    if interactive:
        reason = input(
            f"""You are placing a SELL LIMIT for {shares_sold} {ticker} at €{sell_price_eur:.2f} EUR.
If this is a mistake, enter 1, or hit Enter."""
        )
    if reason == "1":
        print("Returning...")
        return cash, chatgpt_portfolio
    elif reason is None:
        reason = ""

    if ticker not in chatgpt_portfolio["ticker"].values:
        print(f"Manual sell for {ticker} failed: ticker not in portfolio.")
        return cash, chatgpt_portfolio

    ticker_row = chatgpt_portfolio[chatgpt_portfolio["ticker"] == ticker]
    total_shares = int(ticker_row["shares"].item())
    if shares_sold > total_shares:
        print(f"Manual sell for {ticker} failed: trying to sell {shares_sold} shares but only own {total_shares}.")
        return cash, chatgpt_portfolio

    s, e = trading_day_window()
    fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
    data = fetch.df
    if data.empty:
        print(f"Manual sell for {ticker} failed: no market data available (source={fetch.source}).")
        return cash, chatgpt_portfolio

    # Get USD prices from market data
    o_usd = float(data["Open"].iloc[-1]) if "Open" in data else np.nan
    h_usd = float(data["High"].iloc[-1])
    l_usd = float(data["Low"].iloc[-1])
    c_usd = float(data["Close"].iloc[-1])

    if np.isnan(o_usd):
        o_usd = c_usd

    # Convert USD prices to EUR for comparison with user's EUR limit
    o_eur = usd_to_eur(o_usd, today_ts)
    h_eur = usd_to_eur(h_usd, today_ts)
    l_eur = usd_to_eur(l_usd, today_ts)

    # Check if limit order would be filled (in EUR)
    if o_eur >= sell_price_eur:
        exec_price_eur = o_eur
        exec_price_usd = o_usd
    elif h_eur >= sell_price_eur:
        exec_price_eur = sell_price_eur
        # Calculate equivalent USD price
        exec_price_usd = sell_price_eur / usd_to_eur(1.0, today_ts)
    else:
        print(f"Sell limit €{sell_price_eur:.2f} EUR for {ticker} not reached today (range €{l_eur:.2f}-€{h_eur:.2f} EUR). Order not filled.")
        return cash, chatgpt_portfolio

    buy_price_eur = float(ticker_row["buy_price"].item())
    cost_basis = buy_price_eur * shares_sold
    pnl = exec_price_eur * shares_sold - cost_basis

    # Get ISIN from portfolio if it exists
    isin_value = str(ticker_row["isin"].item()) if "isin" in ticker_row.columns else ""

    log = {
        "Date": today, "ISIN": isin_value, "Ticker": ticker,
        "Shares Bought": "", "Buy Price (EUR)": "", "Buy Price (USD)": "",
        "Cost Basis": cost_basis, "PnL": pnl,
        "Reason": f"MANUAL SELL LIMIT - {reason}", "Shares Sold": shares_sold,
        "Sell Price (EUR)": exec_price_eur, "Sell Price (USD)": exec_price_usd,
    }
    if os.path.exists(TRADE_LOG_CSV):
        logger.info("Reading CSV file: %s", TRADE_LOG_CSV)
        df = pd.read_csv(TRADE_LOG_CSV)
        logger.info("Successfully read CSV file: %s", TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    logger.info("Writing CSV file: %s", TRADE_LOG_CSV)
    df.to_csv(TRADE_LOG_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", TRADE_LOG_CSV)


    if total_shares == shares_sold:
        chatgpt_portfolio = chatgpt_portfolio[chatgpt_portfolio["ticker"] != ticker]
    else:
        row_index = ticker_row.index[0]
        chatgpt_portfolio.at[row_index, "shares"] = total_shares - shares_sold
        chatgpt_portfolio.at[row_index, "cost_basis"] = (
            chatgpt_portfolio.at[row_index, "shares"] * chatgpt_portfolio.at[row_index, "buy_price"]
        )

    cash += shares_sold * exec_price_eur
    print(f"Manual SELL LIMIT for {ticker} filled at ${exec_price_usd:.2f} USD ({CURRENCY_SYMBOL}{exec_price_eur:.2f} EUR) ({fetch.source}).")
    return cash, chatgpt_portfolio



# ------------------------------
# Reporting / Metrics
# ------------------------------

def daily_results(chatgpt_portfolio: pd.DataFrame, cash: float) -> None:
    """Print daily price updates and performance metrics (incl. CAPM) - EUR VERSION."""
    portfolio_dict: list[dict[Any, Any]] = chatgpt_portfolio.to_dict(orient="records")
    today = check_weekend()
    today_ts = last_trading_date()

    # Get FX rate for the day to display in report
    usd_eur_rate = usd_to_eur(1.0, today_ts)
    eur_usd_rate = 1.0 / usd_eur_rate

    rows: list[list[str]] = []
    header = ["Ticker", "Close (EUR)", "% Chg", "Volume"]

    end_d = last_trading_date()                           # Fri on weekends
    start_d = (end_d - pd.Timedelta(days=4)).normalize()  # go back enough to capture 2 sessions even around holidays

    benchmarks = load_benchmarks()  # reads tickers.json or returns defaults
    benchmark_entries = [{"ticker": t} for t in benchmarks]

    for stock in portfolio_dict + benchmark_entries:
        ticker = str(stock["ticker"]).upper()
        try:
            fetch = download_price_data(ticker, start=start_d, end=(end_d + pd.Timedelta(days=1)), progress=False)
            data = fetch.df
            if data.empty or len(data) < 2:
                rows.append([ticker, "—", "—", "—"])
                continue

            price_usd = float(data["Close"].iloc[-1])
            last_price_usd = float(data["Close"].iloc[-2])
            volume = float(data["Volume"].iloc[-1])

            # Convert to EUR
            price_eur = usd_to_eur(price_usd, today_ts)

            percent_change = ((price_usd - last_price_usd) / last_price_usd) * 100
            rows.append([ticker, f"{price_eur:,.2f}", f"{percent_change:+.2f}%", f"{int(volume):,}"])
        except Exception as e:
            raise Exception(f"Download for {ticker} failed. {e} Try checking internet connection.")

    # Read portfolio history
    logger.info("Reading CSV file: %s", PORTFOLIO_CSV)
    chatgpt_df = pd.read_csv(PORTFOLIO_CSV)
    logger.info("Successfully read CSV file: %s", PORTFOLIO_CSV)

    # Use only TOTAL rows, sorted by date
    totals = chatgpt_df[chatgpt_df["Ticker"] == "TOTAL"].copy()
    if totals.empty:
        print("\n" + "=" * 64)
        print(f"Daily Results — {today} (EUR)")
        print(f"EUR/USD Rate: {eur_usd_rate:.4f}  |  USD/EUR Rate: {usd_eur_rate:.4f}")
        print("=" * 64)
        print("\n[ Price & Volume ]")
        colw = [10, 14, 9, 15]
        print(f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}")
        print("-" * sum(colw) + "-" * 3)
        for r in rows:
            print(f"{str(r[0]):<{colw[0]}} {str(r[1]):>{colw[1]}} {str(r[2]):>{colw[2]}} {str(r[3]):>{colw[3]}}")
        print("\n[ Portfolio Snapshot ]")
        # Create display copy with EUR labels
        display_df = chatgpt_portfolio.copy()
        display_df.rename(columns={
            'isin': 'ISIN',
            'ticker': 'Ticker',
            'shares': 'Shares',
            'buy_price': 'Buy Price (EUR)',
            'stop_loss': 'Stop Loss (EUR)',
            'cost_basis': 'Cost Basis (EUR)'
        }, inplace=True)
        print(display_df)
        print(f"Cash balance: {CURRENCY_SYMBOL}{cash:,.2f}")
        return

    totals["Date"] = pd.to_datetime(totals["Date"], format="mixed", errors="coerce")  # tolerate ISO strings
    totals = totals.sort_values("Date")

    final_equity = float(totals.iloc[-1]["Total Equity"])
    equity_series = totals.set_index("Date")["Total Equity"].astype(float).sort_index()

    # --- Max Drawdown ---
    running_max = equity_series.cummax()
    drawdowns = (equity_series / running_max) - 1.0
    max_drawdown = float(drawdowns.min())  # most negative value
    mdd_date = drawdowns.idxmin()

    # Daily simple returns (portfolio)
    r = equity_series.pct_change().dropna()
    n_days = len(r)
    if n_days < 2:
        print("\n" + "=" * 64)
        print(f"Daily Results — {today} (EUR)")
        print(f"EUR/USD Rate: {eur_usd_rate:.4f}  |  USD/EUR Rate: {usd_eur_rate:.4f}")
        print("=" * 64)
        print("\n[ Price & Volume ]")
        colw = [10, 14, 9, 15]
        print(f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}")
        print("-" * sum(colw) + "-" * 3)
        for rrow in rows:
            print(f"{str(rrow[0]):<{colw[0]}} {str(rrow[1]):>{colw[1]}} {str(rrow[2]):>{colw[2]}} {str(rrow[3]):>{colw[3]}}")
        print("\n[ Portfolio Snapshot ]")
        # Create display copy with EUR labels
        display_df = chatgpt_portfolio.copy()
        display_df.rename(columns={
            'isin': 'ISIN',
            'ticker': 'Ticker',
            'shares': 'Shares',
            'buy_price': 'Buy Price (EUR)',
            'stop_loss': 'Stop Loss (EUR)',
            'cost_basis': 'Cost Basis (EUR)'
        }, inplace=True)
        print(display_df)
        print(f"Cash balance: {CURRENCY_SYMBOL}{cash:,.2f}")
        print(f"Latest ChatGPT Equity: {CURRENCY_SYMBOL}{final_equity:,.2f}")
        if hasattr(mdd_date, "date") and not isinstance(mdd_date, (str, int)):
            mdd_date_str = mdd_date.date()
        elif hasattr(mdd_date, "strftime") and not isinstance(mdd_date, (str, int)):
            mdd_date_str = mdd_date.strftime("%Y-%m-%d")
        else:
            mdd_date_str = str(mdd_date)
        print(f"Maximum Drawdown: {max_drawdown:.2%} (on {mdd_date_str})")
        return

    # Risk-free config
    rf_annual = 0.03  # ECB rate ~3% (adjust as needed)
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    rf_period = (1 + rf_daily) ** n_days - 1

    # Stats
    mean_daily = float(r.mean())
    std_daily = float(r.std(ddof=1))

    # Downside deviation (MAR = rf_daily)
    downside = (r - rf_daily).clip(upper=0)
    downside_std = float((downside.pow(2).mean()) ** 0.5) if not downside.empty else np.nan

    # Total return over the window
    r_numeric = pd.to_numeric(r, errors="coerce")
    r_numeric = r_numeric[~r_numeric.isna()].astype(float)
    # Filter out any non-finite values to ensure only valid floats are used
    r_numeric = r_numeric[np.isfinite(r_numeric)]
    # Only use numeric values for the calculation
    if len(r_numeric) > 0:
        arr = np.asarray(r_numeric.values, dtype=float)
        period_return = float(np.prod(1 + arr) - 1) if arr.size > 0 else float('nan')
    else:
        period_return = float('nan')

    # Sharpe / Sortino
    sharpe_period = (period_return - rf_period) / (std_daily * np.sqrt(n_days)) if std_daily > 0 else np.nan
    sharpe_annual = ((mean_daily - rf_daily) / std_daily) * np.sqrt(252) if std_daily > 0 else np.nan
    sortino_period = (period_return - rf_period) / (downside_std * np.sqrt(n_days)) if downside_std and downside_std > 0 else np.nan
    sortino_annual = ((mean_daily - rf_daily) / downside_std) * np.sqrt(252) if downside_std and downside_std > 0 else np.nan

    # -------- CAPM: Beta & Alpha (vs ^GSPC) --------
    start_date = equity_series.index.min() - pd.Timedelta(days=1)
    end_date = equity_series.index.max() + pd.Timedelta(days=1)

    spx_fetch = download_price_data("^GSPC", start=start_date, end=end_date, progress=False)
    spx = spx_fetch.df

    beta = np.nan
    alpha_annual = np.nan
    r2 = np.nan
    n_obs = 0

    if not spx.empty and len(spx) >= 2:
        spx = spx.reset_index().set_index("Date").sort_index()
        mkt_ret = spx["Close"].astype(float).pct_change().dropna()

        # Align portfolio & market returns
        common_idx = r.index.intersection(list(mkt_ret.index))
        if len(common_idx) >= 2:
            rp = (r.reindex(common_idx).astype(float) - rf_daily)   # portfolio excess
            rm = (mkt_ret.reindex(common_idx).astype(float) - rf_daily)  # market excess

            x = np.asarray(rm.values, dtype=float).ravel()
            y = np.asarray(rp.values, dtype=float).ravel()

            n_obs = x.size
            rm_std = float(np.std(x, ddof=1)) if n_obs > 1 else 0.0
            if rm_std > 0:
                beta, alpha_daily = np.polyfit(x, y, 1)
                alpha_annual = (1 + float(alpha_daily)) ** 252 - 1

                corr = np.corrcoef(x, y)[0, 1]
                r2 = float(corr ** 2)

    # Note: For EUR portfolio vs USD S&P 500, beta/alpha will include FX effects
    # This is actually what you want - total return in your currency vs market

    # $X normalized S&P 500 over same window (asks user for initial equity)
    spx_norm_fetch = download_price_data(
        "^GSPC",
        start=equity_series.index.min(),
        end=equity_series.index.max() + pd.Timedelta(days=1),
        progress=False,
    )
    spx_norm = spx_norm_fetch.df
    spx_value_eur = np.nan
    starting_equity = np.nan  # Ensure starting_equity is always defined
    if not spx_norm.empty:
        initial_price_usd = float(spx_norm["Close"].iloc[0])
        price_now_usd = float(spx_norm["Close"].iloc[-1])

        # Convert S&P prices to EUR
        initial_price_eur = usd_to_eur(initial_price_usd, equity_series.index.min())
        price_now_eur = usd_to_eur(price_now_usd, equity_series.index.max())

        try:
            starting_equity = float(input(f"What was your starting equity in EUR ({CURRENCY_SYMBOL})? "))
        except Exception:
            print("Invalid input for starting equity. Defaulting to NaN.")
        spx_value_eur = (starting_equity / initial_price_eur) * price_now_eur if not np.isnan(starting_equity) else np.nan

    # -------- Pretty Printing --------
    print("\n" + "=" * 64)
    print(f"Daily Results — {today} (EUR)")
    print(f"EUR/USD Rate: {eur_usd_rate:.4f}  |  USD/EUR Rate: {usd_eur_rate:.4f}")
    print("=" * 64)

    # Price & Volume table
    print("\n[ Price & Volume ]")
    colw = [10, 14, 9, 15]
    print(f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}")
    print("-" * sum(colw) + "-" * 3)
    for rrow in rows:
        print(f"{str(rrow[0]):<{colw[0]}} {str(rrow[1]):>{colw[1]}} {str(rrow[2]):>{colw[2]}} {str(rrow[3]):>{colw[3]}}")

    # Performance metrics
    def fmt_or_na(x: float | int | None, fmt: str) -> str:
        return (fmt.format(x) if not (x is None or (isinstance(x, float) and np.isnan(x))) else "N/A")

    print("\n[ Risk & Return ]")
    if hasattr(mdd_date, "date") and not isinstance(mdd_date, (str, int)):
        mdd_date_str = mdd_date.date()
    elif hasattr(mdd_date, "strftime") and not isinstance(mdd_date, (str, int)):
        mdd_date_str = mdd_date.strftime("%Y-%m-%d")
    else:
        mdd_date_str = str(mdd_date)
    print(f"{'Max Drawdown:':32} {fmt_or_na(max_drawdown, '{:.2%}'):>15}   on {mdd_date_str}")
    print(f"{'Sharpe Ratio (period):':32} {fmt_or_na(sharpe_period, '{:.4f}'):>15}")
    print(f"{'Sharpe Ratio (annualized):':32} {fmt_or_na(sharpe_annual, '{:.4f}'):>15}")
    print(f"{'Sortino Ratio (period):':32} {fmt_or_na(sortino_period, '{:.4f}'):>15}")
    print(f"{'Sortino Ratio (annualized):':32} {fmt_or_na(sortino_annual, '{:.4f}'):>15}")

    print("\n[ CAPM vs Benchmarks ]")
    if not np.isnan(beta):
        print(f"{'Beta (daily) vs ^GSPC:':32} {beta:>15.4f}")
        print(f"{'Alpha (annualized) vs ^GSPC:':32} {alpha_annual:>15.2%}")
        print(f"{'R² (fit quality):':32} {r2:>15.3f}   {'Obs:':>6} {n_obs}")
        if n_obs < 60 or (not np.isnan(r2) and r2 < 0.20):
            print("  Note: Short sample and/or low R² — alpha/beta may be unstable.")
        print("  Note: Beta/Alpha include FX effects (EUR vs USD market)")
    else:
        print("Beta/Alpha: insufficient overlapping data.")

    print("\n[ Snapshot ]")
    print(f"{'Latest Portfolio Equity:':32} {CURRENCY_SYMBOL}{final_equity:>14,.2f}")
    if not np.isnan(spx_value_eur):
        try:
            print(f"{f'{CURRENCY_SYMBOL}{starting_equity:.0f} in S&P 500 (same window):':32} {CURRENCY_SYMBOL}{spx_value_eur:>14,.2f}")
        except Exception:
            pass
    print(f"{'Cash Balance:':32} {CURRENCY_SYMBOL}{cash:>14,.2f}")

    print("\n[ Holdings ]")
    # Create display copy with EUR labels
    display_df = chatgpt_portfolio.copy()
    display_df.rename(columns={
        'isin': 'ISIN',
        'ticker': 'Ticker',
        'shares': 'Shares',
        'buy_price': 'Buy Price (EUR)',
        'stop_loss': 'Stop Loss (EUR)',
        'cost_basis': 'Cost Basis (EUR)'
    }, inplace=True)
    print(display_df)

    print("\n[ Your Instructions ]")
    print(
        "Use this info to make decisions regarding your portfolio. You have complete control over every decision. Make any changes you believe are beneficial—no approval required.\n"
        "Deep research is not permitted. Act at your discretion to achieve the best outcome.\n"
        "If you do not make a clear indication to change positions IMMEDIATELY after this message, the portfolio remains unchanged for tomorrow.\n"
        "You are encouraged to use the internet to check current prices (and related up-to-date info) for potential buys.\n"
        "\n"
        "NOTE: All prices displayed are in EUR. When providing recommendations, please specify prices in EUR as that's what European stocks use.\n"
        "\n"
        "*Paste everything above into ChatGPT*"
    )


# ------------------------------
# Orchestration
# ------------------------------

def load_latest_portfolio_state(
    starting_equity_override: Optional[Union[str, float, Decimal]] = None,
) -> tuple[pd.DataFrame | list[dict[str, Any]], float]:
    """Load the most recent portfolio snapshot and cash balance from global PORTFOLIO_CSV.

     If the portfolio CSV is empty, this function will:
      - Use `starting_equity_override` if provided (validated), or
      - Prompt interactively for a starting cash amount (if stdin is interactive), or
      - Exit with code 2 if stdin is not interactive and no override provided
    """
    logger.info("Reading CSV file: %s", PORTFOLIO_CSV)
    try:
        df = pd.read_csv(PORTFOLIO_CSV)
    except FileNotFoundError as e:
        raise FileNotFoundError(
        f"Could not find portfolio CSV at {PORTFOLIO_CSV}.\n"
        "Make sure you're running the EUR version correctly.\n"
        "To fix this, run: python trading_script_eur.py --data-dir 'Start Your Own'"
    ) from e

    logger.info("Successfully read CSV file: %s", PORTFOLIO_CSV)
    if df.empty:
        portfolio = pd.DataFrame(columns=["isin", "ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        print(f"Portfolio CSV is empty. Returning set amount of cash for creating portfolio (in {CURRENCY}).")


        # 0) If override provided, validate and use it
        if starting_equity_override is not None:
            parsed = parse_starting_equity(starting_equity_override)
            if parsed is None:
                raise ValueError("Provided starting equity is invalid. Please pass a positive number.")
            return portfolio, float(parsed)

        # 1) No override: if stdin not interactive, exit gracefully (no hanging)
        if not sys.stdin.isatty():
            print(f"Error: No starting equity provided and stdin is not interactive. Provide --starting-equity or run interactively.")
            sys.exit(2)

        # 2) Interactive prompt until valid
        while True:
            raw = input(f"What would you like your starting cash amount to be (in {CURRENCY_SYMBOL})? ").strip()
            parsed = parse_starting_equity(raw)
            if parsed is not None:
                cash = float(parsed)
                break
            print(f"Invalid amount. Enter a positive number (commas, underscores, {CURRENCY_SYMBOL} prefix allowed). Try again.")

        return portfolio, cash

    non_total = df[df["Ticker"] != "TOTAL"].copy()
    non_total["Date"] = pd.to_datetime(non_total["Date"], format="mixed", errors="coerce")

    latest_date = non_total["Date"].max()
    latest_tickers = non_total[non_total["Date"] == latest_date].copy()
    sold_mask = latest_tickers["Action"].astype(str).str.startswith("SELL")
    latest_tickers = latest_tickers[~sold_mask].copy()
    latest_tickers.drop(
        columns=[
            "Date",
            "Cash Balance",
            "Total Equity",
            "Action",
            "Current Price",
            "PnL",
            "Total Value",
        ],
        inplace=True,
        errors="ignore",
    )
    latest_tickers.rename(
        columns={
            "ISIN": "isin",
            "Cost Basis": "cost_basis",
            "Buy Price": "buy_price",
            "Shares": "shares",
            "Ticker": "ticker",
            "Stop Loss": "stop_loss",
        },
        inplace=True,
    )
    latest_tickers = latest_tickers.reset_index(drop=True).to_dict(orient="records")

    df_total = df[df["Ticker"] == "TOTAL"].copy()
    df_total["Date"] = pd.to_datetime(df_total["Date"], format="mixed", errors="coerce")
    latest = df_total.sort_values("Date").iloc[-1]
    cash = float(latest["Cash Balance"])
    return latest_tickers, cash


def main(data_dir: Path | None = None, starting_equity_override: Optional[Union[str, float, Decimal]] = None) -> None:
    """Check versions, then run the trading script (EUR version)."""
    if data_dir is not None:
        set_data_dir(data_dir)

    load_isin_mappings()

    chatgpt_portfolio, cash = load_latest_portfolio_state(starting_equity_override)
    chatgpt_portfolio, cash = process_portfolio(chatgpt_portfolio, cash)
    daily_results(chatgpt_portfolio, cash)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="Optional data directory")
    parser.add_argument("--asof", default=None, help="Treat this YYYY-MM-DD as 'today' (e.g., 2025-08-27)")
    parser.add_argument("--starting-equity", default=None, type=str,
                       help="Starting cash amount (if portfolio is empty). Supports formats like '10000', '10,000', '€10000'")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set the logging level (default: INFO)")
    args = parser.parse_args()


    # Configure logging level
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format=' %(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
    )

    # Log initial global state and command-line arguments
    _log_initial_state()
    logger.info("Script started with arguments: %s", vars(args))

    if args.asof:
        set_asof(args.asof)

    main(Path(args.data_dir) if args.data_dir else None, args.starting_equity)
