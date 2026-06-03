#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# setting things up
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import polars as pl
import pandas as pd
import os
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv() # This lets me use local API keys. Replace below or set up your
              # env

fred = Fred(api_key=os.environ["FRED_API_KEY"])

#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setting things up
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# CPI, monthly, not seasonally adjusted
cpi = fred.get_series("CPIAUCNS")

# Compute 2025 average CPI
cpi_2025_avg = cpi["2025"].mean()

# Create deflator
deflator = cpi_2025_avg / cpi

# WTI crude oil price, daily
wti_monthly = fred.get_series("MCOILWTICO")

wti_monthly_df = (
    wti_monthly
    .rename("wti_price")
    .reset_index()
    .rename(columns={"index": "date"})
)

print(wti_monthly_df.head())
print(wti_monthly_df.tail())

wti_daily_df.to_csv("wti_daily.csv", index=False)
