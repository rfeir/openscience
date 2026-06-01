#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# setting things up
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

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
wti_daily = fred.get_series("DCOILWTICO")

wti_daily_df = (
    wti_daily
    .rename("wti_price")
    .reset_index()
    .rename(columns={"index": "date"})
)

print(wti_daily_df.head())
print(wti_daily_df.tail())

wti_daily_df.to_csv("wti_daily.csv", index=False)
