#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# setting things up
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import polars as pl
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from fredapi import Fred
import matplotlib.pyplot as plt

load_dotenv() # This lets me use local API keys. Replace below or set up your
              # env

plt.style.use('ricky_mocha')

fred = Fred(api_key=os.environ["FRED_API_KEY"]) # replace with your key


#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# getting that data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# CPI, monthly, not seasonally adjusted
cpi = fred.get_series("CPIAUCNS")

# Null October garbo, solved with geometric mean prev and post month 
# https://www.bls.gov/cpi/factsheets/approximating-missing-data.htm

oct_date = pd.Timestamp("2025-10-01")
if pd.isna(cpi.loc[oct_date]):
  prev_val = cpi.loc[oct_date - pd.offsets.MonthBegin(1)]  # Sep
  next_val = cpi.loc[oct_date + pd.offsets.MonthBegin(1)]  # Nov
  cpi.loc[oct_date] = round(np.sqrt(prev_val * next_val), 3)

# Compute 2025 average CPI
cpi_2025_avg = cpi["2025"].mean()

# Create deflator
deflator = cpi_2025_avg / cpi

# WTI crude oil price, daily
wti_monthly = fred.get_series("MCOILWTICO")

deflator_df = (
  deflator
  .rename("deflator")
  .reset_index()
  .rename(columns={"index": "date"})
)

wti_monthly_df = (
  wti_monthly
  .rename("wti_price")
  .reset_index()
  .rename(columns={"index": "date"})
)

wti_monthly_df = wti_monthly_df.merge(deflator_df, on="date", how="left")

wti_monthly_df["real_wti"] = (
  wti_monthly_df["wti_price"] * wti_monthly_df["deflator"]
)

plt.plot(wti_monthly_df['date'], wti_monthly_df['real_wti'])
plt.show()
plt.close('all')


#%%
# https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=pet&s=f000000__3&f=a
crude_annual_df  = pd.read_csv('oil_funds/us_crude_fpp_annual.csv')
crude_annual_df  = crude_annual_df.rename(
  columns={
    'Year': 'year',
    'U.S. Crude Oil First Purchase Price Dollars per Barrel': 'price',
  }
)

crude_monthly_df = pd.read_csv('oil_funds/us_crude_fpp_monthly.csv')
crude_monthly_df  = crude_monthly_df.rename(
  columns={
    'Month': 'date',
    'U.S. Crude Oil First Purchase Price Dollars per Barrel': 'price',
  }
)

crude_monthly_df['date'] = pd.to_datetime(
  crude_monthly_df['date'].str.title(),
  format='%b %Y',
)

#%%

from statsmodels.tsa.statespace.sarimax import SARIMAX

df = (
    wti_monthly_df[["date", "real_wti"]]
    .dropna()
    .sort_values("date")
    .set_index("date")
)

df["log_real_wti"] = np.log(df["real_wti"])

mod = SARIMAX(
    df["log_real_wti"],
    order=(1, 0, 0),
    trend="c"
)

res = mod.fit(disp=False)

df["log_ar1_smoothed"] = res.smoother_results.smoothed_state[0]
df["ar1_smoothed"] = np.exp(df["log_ar1_smoothed"])

plt.plot(df.index, df["real_wti"])
plt.plot(df.index, df["ar1_smoothed"])
plt.legend(["Real WTI", "AR(1) Kalman-smoothed"])
plt.show()
plt.close("all")

#%%
print(wti_monthly_df.head())
print(wti_monthly_df.tail())

wti_monthly_df.to_csv("wti_monthly.csv", index=False)
