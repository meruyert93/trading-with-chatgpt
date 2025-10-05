# EUR Input Changes - Changelog

## Overview
Updated `trading_script_eur.py` to accept EUR prices from users in the terminal instead of USD. The script now only converts USD market data to EUR for comparison purposes.

## Changes Made

### 1. **Updated `log_manual_buy()` Function**
   - **Parameter Change**: `buy_price_usd` → `buy_price_eur`, `stoploss_usd` → `stoploss_eur`
   - **Behavior**:
     - User provides limit price and stop-loss in EUR
     - Market data is fetched in USD from Yahoo Finance
     - USD prices are converted to EUR for comparison with user's limit
     - If order fills, both EUR and USD execution prices are logged
   - **Line**: 870-996

### 2. **Updated `log_manual_sell()` Function**
   - **Parameter Change**: `sell_price_usd` → `sell_price_eur`
   - **Behavior**:
     - User provides sell limit price in EUR
     - Market data is fetched in USD from Yahoo Finance
     - USD prices are converted to EUR for comparison
     - If order fills, both EUR and USD execution prices are logged
   - **Line**: 998-1104

### 3. **Updated Interactive Buy Input (MOO)**
   - **Change**: Stop-loss input now in EUR instead of USD
   - **Location**: Line 600-606
   - **Prompt**: "Enter stop loss in EUR (or 0 to skip)"

### 4. **Updated Interactive Buy Input (Limit)**
   - **Change**: Buy limit price and stop-loss now in EUR instead of USD
   - **Location**: Line 684-692
   - **Prompts**:
     - "Enter buy LIMIT price in EUR"
     - "Enter stop loss in EUR (or 0 to skip)"

### 5. **Updated Interactive Sell Input (Limit)**
   - **Change**: Sell limit price now in EUR instead of USD
   - **Location**: Line 706-730
   - **Prompt**: "Enter sell LIMIT price in EUR"

### 6. **Updated Interactive Sell Input (MOO)**
   - **Change**: Market price converted from USD to EUR automatically
   - **Location**: Line 708-717

### 7. **Updated Confirmation Prompts**
   - **log_manual_buy**: "You are placing a BUY LIMIT for {shares} {ticker} at €{buy_price_eur:.2f} EUR."
   - **log_manual_sell**: "You are placing a SELL LIMIT for {shares_sold} {ticker} at €{sell_price_eur:.2f} EUR."

### 8. **Updated User Instructions**
   - **Change**: Updated instructions to clarify all prices are in EUR
   - **Location**: Line 1361
   - **New Text**: "NOTE: All prices displayed are in EUR. When providing recommendations, please specify prices in EUR as that's what European stocks use."

## Technical Details

### FX Conversion Logic

**Before (USD Input)**:
```python
# User provided USD, converted to EUR after execution
buy_price_usd = float(input("Enter buy LIMIT price in USD: "))
exec_price_eur = usd_to_eur(exec_price_usd, today_ts)
```

**After (EUR Input)**:
```python
# User provides EUR, market data converted to EUR for comparison
buy_price_eur = float(input("Enter buy LIMIT price in EUR: "))
o_eur = usd_to_eur(o_usd, today_ts)  # Convert market data
if o_eur <= buy_price_eur:
    exec_price_eur = o_eur
    exec_price_usd = o_usd  # Keep original USD price
```

### Price Execution Logic

For **BUY** orders:
1. User specifies EUR limit price (e.g., €138.00)
2. Market opens at USD price (e.g., $150.00)
3. Convert market USD to EUR (e.g., $150 × 0.92 = €138.00)
4. If EUR market price ≤ EUR limit, order fills
5. Log both EUR execution price and equivalent USD price

For **SELL** orders:
1. User specifies EUR limit price (e.g., €147.20)
2. Market opens at USD price (e.g., $160.00)
3. Convert market USD to EUR (e.g., $160 × 0.92 = €147.20)
4. If EUR market price ≥ EUR limit, order fills
5. Log both EUR execution price and equivalent USD price

## Benefits

1. **User-Friendly**: European users think in EUR, not USD
2. **Accurate**: Prices entered match what users see in EUR-based platforms (e.g., Trade Republic)
3. **Transparent**: Both EUR and USD prices are logged for auditing
4. **Consistent**: All portfolio values, P&L, and cash balances remain in EUR

## Testing

All 39 unit tests pass after updates:
- ✅ `test_log_manual_buy` - Updated to use EUR parameters
- ✅ `test_log_manual_sell` - Updated to use EUR parameters
- ✅ `test_log_manual_buy_insufficient_cash` - Updated to use EUR parameters
- ✅ All other tests unchanged and passing

## Migration Notes

**No migration needed** - This is a UI/input change only. Existing CSV files with EUR values remain compatible.

## Example Usage

### Before (USD Input):
```
Enter buy LIMIT price in USD: 150.00
Enter stop loss in USD (or 0 to skip): 140.00
```

### After (EUR Input):
```
Enter buy LIMIT price in EUR: 138.00
Enter stop loss in EUR (or 0 to skip): 128.80
```

Both produce the same result, but the EUR version is more intuitive for European users.

## Files Modified

1. `trading_script_eur.py` - Main script with EUR input changes
2. `test_trading_script_eur.py` - Unit tests updated for EUR parameters

## Compatibility

- ✅ Compatible with existing portfolio CSVs
- ✅ Compatible with existing trade log CSVs
- ✅ Compatible with ISIN mapping functionality
- ✅ All existing features (MOO, limit orders, stop-loss) work as before
