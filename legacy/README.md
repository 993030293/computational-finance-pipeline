# Code and Documentation

This repository contains the code for **Project 2: Financial Data Acquisition and Basic Processing**.

## Environment
- OS: Windows 11
- Python: 3.12.3
- pandas: 2.2.2
- akshare: 1.17.71

## Installation
```bash
pip install -U akshare pandas numpy requests
```

## How to Run
```bash
python project2_acquire_clean.py
```

## Outputs
```text
./data/
  raw/
    stock_universe.csv
    daily_price.csv
  processed/
    daily_price_50.csv
  database/
    financial_data.db
  REPORT.md
```

## Notes
- Proxies are disabled by default to reduce network errors.
- `pct_chg` is computed from `close` when missing.
- Exactly 50 symbols are included in the single CSV as required by the PDF.
