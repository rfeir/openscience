#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                              FOR CREATING .CSV
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#                                 DATA NOTES
# This file is how you could generate your own data file in a csv format from
# the original data source. There are ways to get the BEA tables through
# downloading tables if you don't want to bother with getting an API. I'm
# sharing this for replication/editing purposes. You will also need to insert
# your own paths or APIs in the place of anythign with "os.environ['']". 


#                                DATA SOURCES

# 1. Census Population Figures: 
#    matches with this: https://fred.stlouisfed.org/series/POPTOTUSA647NWDB
#    2000-2009
#      https://www.census.gov/data/tables/time-series/demo/popest/intercensal-2000-2010-state.html
#    2010-2020
#      https://www.census.gov/data/tables/time-series/demo/popest/intercensal-2010-2020-state.html
#    2020-2023
#      https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html
#      Annual Estimates of the Resident Population for the United States, Regions, States, District of Columbia and Puerto Rico: April 1, 2020 to July 1, 2024 (NST-EST2024-POP) [< 1.0 MB] 

# 2. Foreign-Born Population Percentage and Education Rates from ACS Public Use
#    Microdata 1-Year Files.
#    https://www2.census.gov/programs-surveys/acs/data/pums/

# 3. 2025 Economic Freedom of North America (Subnational)
#    https://www.fraserinstitute.org/studies/economic-freedom-north-america-2025

# 4. Table SAGDP9 Real GDP by industry in chained dollars for lefthand variable
#    and ind shares. Available below or with API method used in this file.
#    https://apps.bea.gov/itable/?ReqID=70&step=1&_gl=1*1on55rm*_ga*MTExMzEyNTc3OC4xNzc4MDkxNzQ2*_ga_J4698JNNFT*czE3ODA2NjY1MzMkbzEwJGcxJHQxNzgwNjY2OTcyJGo1OSRsMCRoMA..


import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv
load_dotenv()


#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                             1. CENSUS DATA
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 2005-2009
df0 = pd.read_excel('gdp_imm/st-est00int-01.xls', skiprows=3)
df0 = df0.drop(index=list(range(0, 5)) + [13] + list(range(56, len(df0))))
df0 = df0.rename(columns={'Unnamed: 0': 'STNAME'})
df0 = df0.loc[:, ~df0.columns.astype(str).str.startswith('Unnamed')]
df0['STNAME'] = df0['STNAME'].str.replace(r'^.*?\.\s*', '', regex=True)

# 2010-2019
df1 = pd.read_excel('gdp_imm/nst-est2020int-pop.xlsx', skiprows=3)
df1 = df1.drop(index=list(range(0, 5)) + [13] + list(range(56, len(df1))))
df1 = df1.rename(columns={'Unnamed: 0': 'STNAME'})
df1 = df1.loc[:, ~df1.columns.astype(str).str.startswith('Unnamed')]
df1['STNAME'] = df1['STNAME'].str.replace(r'^.*?\.\s*', '', regex=True)

# 2020-2023
df2 = pd.read_excel('gdp_imm/NST-EST2024-POP.xlsx', skiprows=3)
df2 = df2.drop(index=list(range(0, 5)) + [13] + list(range(56, len(df2))))
df2 = df2.rename(columns={'Unnamed: 0': 'STNAME'})
df2 = df2.loc[:, ~df2.columns.astype(str).str.startswith('Unnamed')]
df2['STNAME'] = df2['STNAME'].str.replace(r'^.*?\.\s*', '', regex=True)

pop_wide = df0.merge(df1, on='STNAME', how='outer')
pop_wide = pop_wide.merge(df2, on='STNAME', how='outer')
pop_wide = pop_wide.drop(columns=list(range(2000, 2005)) + [2024])

del df0, df1, df2


#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                              2. ACS DATA
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

base_path = os.environ["ACS_PUM_PATH"]

files_by_year = [
  (2005, f'{base_path}csv_pus_05/ss05pus.csv'),
  (2006, f'{base_path}csv_pus_06/ss06pusa.csv'),
  (2006, f'{base_path}csv_pus_06/ss06pusb.csv'),
  (2007, f'{base_path}csv_pus_07/ss07pusa.csv'),
  (2007, f'{base_path}csv_pus_07/ss07pusb.csv'),
  (2008, f'{base_path}csv_pus_08/ss08pusa.csv'),
  (2008, f'{base_path}csv_pus_08/ss08pusb.csv'),
  (2009, f'{base_path}csv_pus_09/ss09pusa.csv'),
  (2009, f'{base_path}csv_pus_09/ss09pusb.csv'),
  (2010, f'{base_path}csv_pus_10/ss10pusa.csv'),
  (2010, f'{base_path}csv_pus_10/ss10pusb.csv'),
  (2011, f'{base_path}csv_pus_11/ss11pusa.csv'),
  (2011, f'{base_path}csv_pus_11/ss11pusb.csv'),
  (2012, f'{base_path}csv_pus_12/ss12pusa.csv'),
  (2012, f'{base_path}csv_pus_12/ss12pusb.csv'),
  (2013, f'{base_path}csv_pus_13/ss13pusa.csv'),
  (2013, f'{base_path}csv_pus_13/ss13pusb.csv'),
  (2014, f'{base_path}csv_pus_14/ss14pusa.csv'),
  (2014, f'{base_path}csv_pus_14/ss14pusb.csv'),
  (2015, f'{base_path}csv_pus_15/ss15pusa.csv'),
  (2015, f'{base_path}csv_pus_15/ss15pusb.csv'),
  (2016, f'{base_path}csv_pus_16/ss16pusa.csv'),
  (2016, f'{base_path}csv_pus_16/ss16pusb.csv'),
  (2017, f'{base_path}csv_pus_17/psam_pusa.csv'),
  (2017, f'{base_path}csv_pus_17/psam_pusb.csv'),
  (2018, f'{base_path}csv_pus_18/psam_pusa.csv'),
  (2018, f'{base_path}csv_pus_18/psam_pusb.csv'),
  (2019, f'{base_path}csv_pus_19/psam_pusa.csv'),
  (2019, f'{base_path}csv_pus_19/psam_pusb.csv'),
  (2020, f'{base_path}csv_pus_20/psam_pusa.csv'),
  (2020, f'{base_path}csv_pus_20/psam_pusb.csv'),
  (2021, f'{base_path}csv_pus_21/psam_pusa.csv'),
  (2021, f'{base_path}csv_pus_21/psam_pusb.csv'),
  (2022, f'{base_path}csv_pus_22/psam_pusa.csv'),
  (2022, f'{base_path}csv_pus_22/psam_pusb.csv'),
  (2023, f'{base_path}csv_pus_23/psam_pusa.csv'),
  (2023, f'{base_path}csv_pus_23/psam_pusb.csv'),
#  (2024, f'{base_path}csv_pus_24/psam_pusa.csv'),
#  (2024, f'{base_path}csv_pus_24/psam_pusb.csv'),
]

def process_chunk(chunk, year):
  if 'STATE' in chunk.columns:
    chunk = chunk.rename(columns={'STATE': 'ST'})

  chunk['ST'] = chunk['ST'].astype(str).str.zfill(2)

  w = chunk['PWGTP']
  age_25_up = chunk['AGEP'].ge(25).fillna(False) # different than old WP
  immigrant = chunk['CIT'].isin([4, 5]).fillna(False)

  if year < 2008:
    bach = chunk['SCHL'].ge(13).fillna(False)
  else:
    bach = chunk['SCHL'].ge(21).fillna(False)

  out = pd.DataFrame({
    'YEAR': year,
    'ST': chunk['ST'],
    'total_pop': w,
    'pop_25_up': w.where(age_25_up, 0),
    'immigrant_pop': w.where(immigrant, 0),
    'bach_pop': w.where(age_25_up & bach, 0),
  })

  return out.groupby(['YEAR', 'ST'], as_index=False).sum()


partials = []

for year, path in files_by_year:
  state_col = 'ST' if year < 2023 else 'STATE'
  usecols = ['PWGTP', 'CIT', 'AGEP', 'SCHL', state_col]

  for chunk in pd.read_csv(
    path,
    usecols=usecols,
    chunksize=250_000,
    dtype={
      'PWGTP': 'int32',
      'CIT': 'Int8',
      'AGEP': 'Int8',
      'SCHL': 'Int8',
      state_col: 'string'
    }
  ):
    partials.append(process_chunk(chunk, year))

state_level = (
  pd.concat(partials, ignore_index=True)
  .groupby(['YEAR', 'ST'], as_index=False)
  .sum()
)

state_level['immigrant_share'] = (
  state_level['immigrant_pop'] / state_level['total_pop']
)

state_level['bach_share'] = (
  state_level['bach_pop'] / state_level['pop_25_up']
)

state_level = state_level[
  ['YEAR', 'ST', 'immigrant_share', 'bach_share']
]

fips_to_state = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas',
    '06': 'California', '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware',
    '11': 'District of Columbia', '12': 'Florida', '13': 'Georgia', '15': 'Hawaii',
    '16': 'Idaho', '17': 'Illinois', '18': 'Indiana', '19': 'Iowa',
    '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine',
    '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
    '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', '31': 'Nebraska',
    '32': 'Nevada', '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico',
    '36': 'New York', '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio',
    '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island',
    '45': 'South Carolina', '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas',
    '49': 'Utah', '50': 'Vermont', '51': 'Virginia', '53': 'Washington',
    '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming'
}

# Add state names to state_level before merging
state_level['ST'] = state_level['ST'].astype(str).str.zfill(2)
state_level['STNAME'] = state_level['ST'].map(fips_to_state)

# Wide Census population -> long
pop_long = pop_wide.melt(
    id_vars='STNAME',
    var_name='YEAR',
    value_name='population'
)

pop_long['YEAR'] = pop_long['YEAR'].astype(int)
pop_long['population'] = pd.to_numeric(pop_long['population'], errors='coerce')

# Merge population with ACS shares
reg_long = pop_long.merge(
    state_level,
    on=['STNAME', 'YEAR'],
    how='left',
)

# Create levels
reg_long['immigrants'] = (
    reg_long['population'] * reg_long['immigrant_share']
)

reg_long['domestics'] = (
    reg_long['population'] * (1 - reg_long['immigrant_share'])
)

# Cleaning types and names
reg_long['population'] = reg_long['population'].round().astype('int32')
reg_long['domestics'] = reg_long['domestics'].round().astype('int32')
reg_long['immigrants'] = reg_long['immigrants'].round().astype('int32')
reg_long.columns = reg_long.columns.str.lower()
reg_long = reg_long[
  ['st', 'stname', 'year', 'population', 'domestics', 'immigrants', 'bach_share']
]


#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                              3. EFI DATA
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

efi_file = 'gdp_imm/economic-freedom-of-north-america-2025-subn_0.xlsx'
efi_sheet = 'Overall-scores(subn)'

efi_wide = pd.read_excel(
    efi_file,
    sheet_name=efi_sheet
)

year_cols = list(range(2005, 2024))

efi_wide = efi_wide[
    ['Overall subn Scores'] + year_cols
].copy()

efi_wide = efi_wide.rename(columns={
    'Overall subn Scores': 'stname'
})

# Drop blank separator rows
efi_wide = efi_wide.dropna(subset=['stname'])

# Clean state names
efi_wide['stname'] = efi_wide['stname'].astype(str).str.strip()

# Keep only U.S. states that appear in reg_long
us_states = reg_long['stname'].dropna().unique()

efi_wide = efi_wide[
    efi_wide['stname'].isin(us_states)
].copy()

# Wide -> long
efi_long = efi_wide.melt(
    id_vars='stname',
    value_vars=year_cols,
    var_name='year',
    value_name='efi_score'
)

efi_long['year'] = efi_long['year'].astype(int)
efi_long['efi_score'] = pd.to_numeric(efi_long['efi_score'], errors='coerce')

# Merge EFI scores onto main long panel
reg_long = reg_long.merge(
    efi_long,
    on=['stname', 'year'],
    how='left',
    validate='1:1'
)


#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                              4. GDP DATA
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bea_key = os.environ["BEA_API_KEY"]
bea_url = "https://apps.bea.gov/api/data"
years = ",".join(str(y) for y in range(2005, 2024))

# BEA SAGDP9 LineCode -> SuperSector
supersector_map = {
    3: "Natural Resources and Mining",
    6: "Natural Resources and Mining",

    10: "Trade Transportation Utilities Leisure Hospitality Other Services",
    34: "Trade Transportation Utilities Leisure Hospitality Other Services",
    35: "Trade Transportation Utilities Leisure Hospitality Other Services",
    36: "Trade Transportation Utilities Leisure Hospitality Other Services",
    75: "Trade Transportation Utilities Leisure Hospitality Other Services",
    82: "Trade Transportation Utilities Leisure Hospitality Other Services",

    11: "Construction and Manufacturing",
    12: "Construction and Manufacturing",

    45: "Information Financial Activities Professional Business Services",
    50: "Information Financial Activities Professional Business Services",
    59: "Information Financial Activities Professional Business Services",

    68: "Education and Health Services",

#    83: "Public Administration", Left out as government.
}

# 2 = Private industries
# The rest are supersectors.
needed_linecodes = sorted(set(supersector_map.keys()) | {2, 3, 6, 
                                                         10, 34, 35, 36, 75, 82,
                                                         11, 12, 45, 50, 59, 68})

parts = []

for linecode in needed_linecodes:
    params = {
        "UserID": bea_key,
        "method": "GetData",
        "datasetname": "Regional",
        "TableName": "SAGDP9",
        "LineCode": linecode,
        "GeoFIPS": "STATE",
        "Year": years,
        "ResultFormat": "JSON",
    }

    r = requests.get(bea_url, params=params)
    r.raise_for_status()
    js = r.json()

    if "Results" not in js.get("BEAAPI", {}):
        print(f"Failed LineCode {linecode}")
        print(json.dumps(js, indent=2)[:1500])
        continue

    tmp = pd.DataFrame(js["BEAAPI"]["Results"]["Data"])
    tmp["linecode"] = linecode
    parts.append(tmp)

gdp_df = pd.concat(parts, ignore_index=True)

# Clean columns
gdp_df.columns = gdp_df.columns.str.lower()

gdp_df = gdp_df.rename(columns={
    "geofips": "geofips",
    "geoname": "bea_stname",
    "timeperiod": "year",
    "data_value": "gdp_chained",
    "datavalue": "gdp_chained",
})

# Clean state FIPS:
gdp_df["st"] = (
    gdp_df["geofips"]
    .astype(str)
    .str.replace('"', '', regex=False)
    .str.replace("'", "", regex=False)
    .str.strip()
    .str.zfill(5)
    .str[:2]
)

gdp_df["year"] = pd.to_numeric(gdp_df["year"], errors="coerce").astype("Int64")
gdp_df["linecode"] = pd.to_numeric(gdp_df["linecode"], errors="coerce").astype("Int64")

gdp_df["gdp_chained"] = (
    gdp_df["gdp_chained"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .replace({
        "(NA)": np.nan,
        "NA": np.nan,
        "N": np.nan,
        "...": np.nan,
        "": np.nan
    })
)

gdp_df["gdp_chained"] = pd.to_numeric(gdp_df["gdp_chained"], errors="coerce")

# Keep only states in your panel
valid_states = reg_long["st"].astype(str).str.zfill(2).unique()
gdp_df = gdp_df[gdp_df["st"].isin(valid_states)].copy()

private_gdp = (
    gdp_df[gdp_df["linecode"].eq(2)]
    [["st", "year", "gdp_chained"]]
    .rename(columns={"gdp_chained": "private_gdp"})
    .copy()
)

# Supersector GDP
sector_long = gdp_df[gdp_df["linecode"].isin(supersector_map.keys())].copy()
sector_long["supersector"] = sector_long["linecode"].map(supersector_map)

sector_long = (
    sector_long
    .groupby(["st", "year", "supersector"], as_index=False)
    ["gdp_chained"]
    .sum()
)

sector_wide = (
    sector_long
    .pivot_table(
        index=["st", "year"],
        columns="supersector",
        values="gdp_chained",
        aggfunc="sum"
    )
    .reset_index()
)

sector_wide.columns.name = None

sector_wide = sector_wide.rename(
    columns={
        c: (
            c.lower()
             .replace(" ", "_")
             .replace(",", "")
             .replace("&", "and")
             + "_gdp"
        )
        for c in sector_wide.columns
        if c not in ["st", "year"]
    }
)

# Create aggregated sum for shares, since we can't be sure where rounding
# errors in the private GDP sum value exist. Not meaningful anyway.

sector_gdp_cols = [
    c for c in sector_wide.columns
    if c.endswith("_gdp")
]

sector_wide["supersector_gdp_sum"] = sector_wide[sector_gdp_cols].sum(axis=1)

for col in sector_gdp_cols:
    sector_wide[col.replace("_gdp", "_share")] = (
        sector_wide[col] / sector_wide["supersector_gdp_sum"]
    )

# Keep only shares plus denominator
sector_share_cols = [
    c for c in sector_wide.columns
    if c.endswith("_share")
]

sector_wide = sector_wide[
    ["st", "year", "supersector_gdp_sum"] + sector_share_cols
].copy()

# Add total/private GDP
sector_wide = sector_wide.merge(
    private_gdp,
    on=["st", "year"],
    how="left",
    validate="1:1"
)

# check against private GDP
sector_wide["perc_dif"] = (
    sector_wide["supersector_gdp_sum"] / sector_wide["private_gdp"] - 1
)

reg_long = (
    reg_long
    .merge(
        sector_wide,
        on=["st", "year"],
        how="left",
        validate="1:1"
    )
    .rename(columns={
        "construction_and_manufacturing_share": "cm_share",
        "education_and_health_services_share": "ehs_share",
        "information_financial_activities_professional_business_services_share": "ifpbs_share",
        "natural_resources_and_mining_share": "nrm_share",
        "trade_transportation_utilities_leisure_hospitality_other_services_share": "ttuhos_share",
    })
    .drop(columns=["supersector_gdp_sum", "perc_dif"])
)

reg_long.to_csv('data/public/reg_data_2005_2023.csv', index = False)
