# Data Files

This project reads data directly from the parent directory `D:\Real estate Forecast\`.

Expected files:
- `statisticsday.csv` — DABR Matrix export: Active Listings, Sales, New Listings, Sales Volume, Expired Listings (monthly, 2010–present)
- `historicalweeklydata.xlsx` — Freddie Mac PMMS: weekly 30-year and 15-year fixed mortgage rates
- `inflation.xlsx` — CPI, Inflation Rate, Unemployment, Consumer Sentiment Index, Geopolitical Risk Index (monthly)

All paths are configured in `src/config.py`.
