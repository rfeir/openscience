from __future__ import annotations

import gc
import math
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
from dotenv import load_dotenv


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% IMPORTS AND PROJECT SETUP
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT_DIR = Path.cwd()

FILTERED_DIR = PROJECT_DIR / "data" / "filtered"
FILTERED_DIR.mkdir(parents=True, exist_ok=True)

PLOTLY_HTML_DIR = Path("/tmp/cps_plotly")
PLOTLY_HTML_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = PROJECT_DIR / "output"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option(
  "display.float_format",
  lambda x: f"{x:.12f}".rstrip("0").rstrip("."),
)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% SHARED GRAPH SETTINGS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

colors = [
  "#00583D",  #  [0] NDSU Green
  "#FFC425",  #  [1] NDSU Yellow
  "#003524",  #  [2] Dark Green
  "#BED738",  #  [3] Lime Green
  "#51ABA0",  #  [4] Teal
  "#FEF389",  #  [5] Lemon Yellow
  "#A0BD78",  #  [6] Sage
  "#D7E5C8",  #  [7] Pale Sage
  "#BE5E27",  #  [8] Rust
  "#9DD9F7",  #  [9] Morning Sky
  "#0F374B",  # [10] Night Sky
]

graph_colors = {
  "native": colors[3],
  "foreign": colors[2],
  "grid": "#E6E6E6",
  "axis": "#6B6B6B",
  "text": "#222222",
  "paper": "#FFFFFF",
}

nativity_colors = {
  "Native born": graph_colors["native"],
  "Foreign born": graph_colors["foreign"],
}

GRAPH_FONT = "Montserrat, Arial, sans-serif"

GRAPH_TITLE_SIZE = 14
GRAPH_SUBTITLE_SIZE = 14
GRAPH_AXIS_TITLE_SIZE = 14
GRAPH_TICK_SIZE = 14
GRAPH_LEGEND_SIZE = 14
GRAPH_ANNOTATION_SIZE = 14

# Set True during development if you still want every chart to open separately.
OPEN_INDIVIDUAL_FIGURES = False


def project_axis(
  title="",
  tickformat=None,
  tickprefix=None,
  ticksuffix=None,
  showgrid=True,
  zeroline=False,
  categoryorder=None,
  categoryarray=None,
  range_=None,
  tickvals=None,
  ticktext=None,
):
  axis = {
    "title": {
      "text": title,
      "font": {
        "family": GRAPH_FONT,
        "size": GRAPH_AXIS_TITLE_SIZE,
        "color": graph_colors["text"],
      },
    },
    "tickfont": {
      "family": GRAPH_FONT,
      "size": GRAPH_TICK_SIZE,
      "color": graph_colors["text"],
    },
    "showgrid": showgrid,
    "gridcolor": graph_colors["grid"],
    "gridwidth": 1,
    "zeroline": zeroline,
    "zerolinecolor": graph_colors["axis"],
    "zerolinewidth": 1,
    "showline": True,
    "linecolor": graph_colors["axis"],
    "linewidth": 1,
    "mirror": False,
    "automargin": True,
  }

  optional_values = {
    "tickformat": tickformat,
    "tickprefix": tickprefix,
    "ticksuffix": ticksuffix,
    "categoryorder": categoryorder,
    "categoryarray": categoryarray,
    "range": range_,
    "tickvals": tickvals,
    "ticktext": ticktext,
  }

  axis.update(
    {
      key: value
      for key, value in optional_values.items()
      if value is not None
    }
  )

  return axis


def project_layout(
  fig,
  title,
  subtitle=None,
  xaxis=None,
  yaxis=None,
  legend_title=None,
  barmode=None,
  barnorm=None,
  hovermode="closest",
  margin=None,
  annotations=None,
  shapes=None,
  showlegend=True,
):
  if xaxis is None:
    xaxis = project_axis()

  if yaxis is None:
    yaxis = project_axis()

  if margin is None:
    margin = {
      "l": 80,
      "r": 40,
      "b": 70,
      "t": 95,
    }

  if annotations is None:
    annotations = []

  if shapes is None:
    shapes = []

  title_text = title

  if subtitle is not None:
    title_text = (
      f"{title}"
      f"<br><span style='font-size:{GRAPH_SUBTITLE_SIZE}px;"
      f"font-weight:normal'>{subtitle}</span>"
    )

  layout_values = {
    "title": {
      "text": title_text,
      "x": 0.02,
      "xanchor": "left",
      "y": 0.98,
      "yanchor": "top",
      "font": {
        "family": GRAPH_FONT,
        "size": GRAPH_TITLE_SIZE,
        "color": graph_colors["text"],
      },
    },
    "font": {
      "family": GRAPH_FONT,
      "size": GRAPH_TICK_SIZE,
      "color": graph_colors["text"],
    },
    "paper_bgcolor": graph_colors["paper"],
    "plot_bgcolor": graph_colors["paper"],
    "xaxis": xaxis,
    "yaxis": yaxis,
    "hovermode": hovermode,
    "margin": margin,
    "annotations": annotations,
    "shapes": shapes,
    "showlegend": showlegend,
    "legend": {
      "title": {
        "text": legend_title,
        "font": {
          "family": GRAPH_FONT,
          "size": GRAPH_LEGEND_SIZE,
        },
      },
      "font": {
        "family": GRAPH_FONT,
        "size": GRAPH_LEGEND_SIZE,
      },
      "bgcolor": "rgba(255,255,255,0.75)",
      "bordercolor": "rgba(0,0,0,0)",
      "orientation": "h",
      "x": 0,
      "xanchor": "left",
      "y": -0.17,
      "yanchor": "top",
    },
  }

  if barmode is not None:
    layout_values["barmode"] = barmode

  if barnorm is not None:
    layout_values["barnorm"] = barnorm

  fig.update_layout(**layout_values)

  return fig


def project_config(filename="figure"):
  return {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
      "lasso2d",
      "select2d",
    ],
    "toImageButtonOptions": {
      "format": "png",
      "filename": filename,
      "height": 900,
      "width": 1400,
      "scale": 2,
    },
  }


def show_figure(
  fig,
  filename="figure",
  open_browser=OPEN_INDIVIDUAL_FIGURES,
):
  output_file = PLOTLY_HTML_DIR / f"{filename}.html"

  fig.write_html(
    output_file,
    config=project_config(filename),
    include_plotlyjs=True,
    full_html=True,
    auto_open=False,
  )

  print(
    f"Wrote Plotly figure: {output_file}",
    flush=True,
  )

  if open_browser:
    try:
      subprocess.Popen(
        [
          "xdg-open",
          str(output_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
      )

    except FileNotFoundError:
      print(
        "Could not find xdg-open. "
        f"Open this file manually: {output_file}"
      )


def project_source_annotation(
  text,
  y=-0.25,
):
  return {
    "text": text,
    "x": 0,
    "xref": "paper",
    "xanchor": "left",
    "y": y,
    "yref": "paper",
    "yanchor": "top",
    "showarrow": False,
    "align": "left",
    "font": {
      "family": GRAPH_FONT,
      "size": GRAPH_ANNOTATION_SIZE,
      "color": "#666666",
    },
  }


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% PROJECT CONTROLS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CPS_START_YEAR = 2005
CPS_END_YEAR = date.today().year

DOWNLOAD_CPS = True

# PWSSWGT contains four implied decimal places.
CPS_WEIGHT_DIVISOR = 10000

# Number of complete calendar years for the seasonality graph.
SEASONAL_YEAR_COUNT = 3


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CPS VARIABLES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CPS_VARIABLES = [
  "HRYEAR4",  # survey year
  "HRMONTH",  # survey month
  "PRTAGE",  # age
  "PESEX",  # sex
  "PRCITSHP",  # citizenship / nativity
  "PEMNTVTY",  # mother's country of birth
  "PEFNTVTY",  # father's country of birth
  "PRINUYER",  # interval-coded year of entry
  "PEMLR",  # monthly labor-force status
  "PRCIVLF",  # civilian labor-force recode
  "PWSSWGT",  # final person weight
]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CPS DATA: DOWNLOAD AND LOAD
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _month_code(month_number):
  return [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
  ][month_number - 1]


def download_cps_monthly_files(resume_date="2025-10-01"):
  """Download CPS Basic monthly files. Call manually when you want to refresh."""
  if not DOWNLOAD_CPS:
    return

  load_dotenv(".env")
  census_api_key = os.getenv("CENSUS_API_KEY", "")

  if not census_api_key:
    raise RuntimeError(
      f"CENSUS_API_KEY was not found in {PROJECT_DIR / '.env'}."
    )

  resume_date = pd.Timestamp(resume_date)
  current_month = pd.Timestamp(date.today()).replace(day=1)
  request_calendar = []

  for year in range(CPS_START_YEAR, CPS_END_YEAR + 1):
    for month_number in range(1, 13):
      month_date = pd.Timestamp(
        year=year,
        month=month_number,
        day=1,
      )
      if resume_date <= month_date <= current_month:
        request_calendar.append(
          (year, month_number, month_date)
        )

  failed_months = 0
  session = requests.Session()

  for year, month_number, _ in request_calendar:
    print(f"Downloading CPS {year:04d}-{month_number:02d}")

    url = (
      f"https://api.census.gov/data/{year:04d}/cps/basic/"
      f"{_month_code(month_number)}"
    )

    response = None

    for attempt in range(8):
      try:
        candidate = session.get(
          url,
          params={
            "get": ",".join(CPS_VARIABLES),
            "key": census_api_key,
          },
          timeout=120,
        )

        if candidate.status_code in {400, 401, 403, 404}:
          response = candidate
          break

        candidate.raise_for_status()
        response = candidate
        break

      except requests.RequestException:
        if attempt == 7:
          response = None
        else:
          time.sleep(min(2 * (2**attempt), 30))

    if response is None or response.status_code != 200:
      failed_months += 1
      print(f"Skipped CPS {year:04d}-{month_number:02d}")

      if failed_months >= 2:
        print(
          "Two consecutive months were unavailable. "
          "Stopping download."
        )
        break

      continue

    data = response.json()

    if not data or len(data) < 2:
      failed_months += 1

      if failed_months >= 2:
        print(
          "Two consecutive months were unavailable. "
          "Stopping download."
        )
        break

      continue

    month_data = pd.DataFrame(
      data[1:],
      columns=data[0],
    )

    month_data = month_data[CPS_VARIABLES].apply(
      pd.to_numeric,
      errors="coerce",
    )

    month_file = (
      PROJECT_DIR
      / "data"
      / "cps_api_cache"
      / f"cps_basic_{year:04d}_{month_number:02d}.csv"
    )

    month_file.parent.mkdir(
      parents=True,
      exist_ok=True,
    )

    month_data.to_csv(
      month_file,
      index=False,
    )

    failed_months = 0
    time.sleep(0.5)


ENTRY_LOWER = {
  1: 1900, 2: 1950, 3: 1960, 4: 1965, 5: 1970,
  6: 1975, 7: 1980, 8: 1982, 9: 1984, 10: 1986,
  11: 1988, 12: 1990, 13: 1992, 14: 1994, 15: 1996,
  16: 1998, 17: 2000, 18: 2002, 19: 2004, 20: 2006,
  21: 2008, 22: 2010, 23: 2012, 24: 2014, 25: 2016,
  26: 2018, 27: 2020, 28: 2022, 29: 2024,
}

ENTRY_UPPER = {
  1: 1949, 2: 1959, 3: 1964, 4: 1969, 5: 1974,
  6: 1979, 7: 1981, 8: 1983, 9: 1985, 10: 1987,
  11: 1989, 12: 1991, 13: 1993, 14: 1995, 15: 1997,
  16: 1999, 17: 2001, 18: 2003, 19: 2005, 20: 2007,
  21: 2009, 22: 2011, 23: 2013, 24: 2015, 25: 2017,
  26: 2019, 27: 2021, 28: 2023, 29: 2026,
}


def add_derived_variables_to_cps_files():
  """One-time CPS microdata edit. Call manually; it overwrites monthly files."""
  cps_files = sorted(
    (PROJECT_DIR / "data" / "cps_api_cache").glob(
      "cps_basic_????_??.csv"
    )
  )

  cps_files = [
    file
    for file in cps_files
    if file.name != "cps_basic_2025_10.csv"
  ]

  for file in cps_files:
    month_data = pd.read_csv(file)

    month_data["PERSON_WEIGHT"] = (
      month_data["PWSSWGT"]
      / CPS_WEIGHT_DIVISOR
    )

    month_data["FOREIGN_BORN"] = np.select(
      [
        month_data["PRCITSHP"].isin([1, 2, 3]),
        month_data["PRCITSHP"].isin([4, 5]),
      ],
      [0.0, 1.0],
      default=np.nan,
    )

    # Retained for compatibility with your inspection workflow.
    # Graph 4B below uses the raw parental birthplace variables directly.
    if {"PEMNTVTY", "PEFNTVTY"}.issubset(
      month_data.columns
    ):
      mother_non_us = (
        month_data["PEMNTVTY"].notna()
        & (month_data["PEMNTVTY"] > 0)
        & (month_data["PEMNTVTY"] != 57)
      )

      father_non_us = (
        month_data["PEFNTVTY"].notna()
        & (month_data["PEFNTVTY"] > 0)
        & (month_data["PEFNTVTY"] != 57)
      )

      mother_us = month_data["PEMNTVTY"] == 57
      father_us = month_data["PEFNTVTY"] == 57

      month_data["FOREIGN_BORN_PARENT"] = np.select(
        [
          mother_non_us | father_non_us,
          mother_us & father_us,
        ],
        [1.0, 0.0],
        default=np.nan,
      )

    else:
      month_data["FOREIGN_BORN_PARENT"] = np.nan

    month_data["ENTRY_YEAR_LOWER"] = (
      month_data["PRINUYER"].map(ENTRY_LOWER)
    )

    month_data["ENTRY_YEAR_UPPER"] = (
      month_data["PRINUYER"].map(ENTRY_UPPER)
    )

    month_data["ENTRY_YEAR_UPPER"] = np.minimum(
      month_data["ENTRY_YEAR_UPPER"],
      month_data["HRYEAR4"],
    )

    month_data["ESTIMATED_ENTRY_YEAR"] = (
      month_data["ENTRY_YEAR_LOWER"]
      + month_data["ENTRY_YEAR_UPPER"]
    ) / 2

    month_data["YEARS_SINCE_ENTRY"] = (
      month_data["HRYEAR4"]
      - month_data["ESTIMATED_ENTRY_YEAR"]
    )

    recent_condition = (
      (month_data["FOREIGN_BORN"] == 1)
      & np.isfinite(
        month_data["YEARS_SINCE_ENTRY"]
      )
      & (month_data["YEARS_SINCE_ENTRY"] >= 0)
      & (month_data["YEARS_SINCE_ENTRY"] <= 10)
    )

    long_term_condition = (
      (month_data["FOREIGN_BORN"] == 1)
      & np.isfinite(
        month_data["YEARS_SINCE_ENTRY"]
      )
      & (month_data["YEARS_SINCE_ENTRY"] > 10)
    )

    month_data["RECENT_IMMIGRANT_10YR"] = np.select(
      [
        recent_condition,
        long_term_condition,
      ],
      [
        1.0,
        0.0,
      ],
      default=np.nan,
    )

    month_data["IN_CIVILIAN_POPULATION"] = np.where(
      (
        (month_data["PRTAGE"] >= 16)
        & month_data["PEMLR"].isin(
          range(1, 8)
        )
      ),
      1,
      0,
    )

    month_data["IN_LABOR_FORCE"] = np.where(
      (
        (month_data["PRTAGE"] >= 16)
        & month_data["PEMLR"].isin(
          range(1, 5)
        )
      ),
      1,
      0,
    )

    month_data.to_csv(
      file,
      index=False,
    )

    del month_data
    gc.collect()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CPS FILE HELPERS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_cps_files():
  files = sorted(
    (PROJECT_DIR / "data" / "cps_api_cache").glob(
      "cps_basic_????_??.csv"
    )
  )

  return [
    file
    for file in files
    if file.name != "cps_basic_2025_10.csv"
  ]


def file_has_columns(
  file,
  required_columns,
):
  available_columns = pd.read_csv(
    file,
    nrows=0,
  ).columns

  return set(
    required_columns
  ).issubset(
    available_columns
  )


def cps_month_date(data):
  return pd.to_datetime(
    {
      "year": data["HRYEAR4"].astype(int),
      "month": data["HRMONTH"].astype(int),
      "day": 1,
    }
  )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% 0. PREPARE CPS INSPECTION DATA
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cps_files = get_cps_files()

if len(cps_files) == 0:
  raise RuntimeError(
    "No CPS monthly files were found in data/cps_api_cache."
  )

inspection_variables = [
  "PRTAGE",
  "PESEX",
  "FOREIGN_BORN",
  "FOREIGN_BORN_PARENT",
  "RECENT_IMMIGRANT_10YR",
  "IN_CIVILIAN_POPULATION",
  "IN_LABOR_FORCE",
  "PERSON_WEIGHT",
]


def inspection_file_is_usable(
  file,
  variable_name,
):
  available_columns = set(
    pd.read_csv(
      file,
      nrows=0,
    ).columns
  )

  required_columns = {
    variable_name,
    "PERSON_WEIGHT",
  }

  if not required_columns.issubset(
    available_columns
  ):
    return False

  if variable_name == "FOREIGN_BORN_PARENT":
    parent_check = pd.read_csv(
      file,
      usecols=[
        "FOREIGN_BORN_PARENT"
      ],
    )

    values = pd.to_numeric(
      parent_check["FOREIGN_BORN_PARENT"],
      errors="coerce",
    )

    return np.isfinite(values).any()

  return True


def find_first_usable_file(
  files,
  variable_name,
):
  for file in files:
    if inspection_file_is_usable(
      file,
      variable_name,
    ):
      return file

  return None


def find_last_usable_file(
  files,
  variable_name,
):
  for file in reversed(files):
    if inspection_file_is_usable(
      file,
      variable_name,
    ):
      return file

  return None


inspection_data_parts = []

for variable_number, variable_name in enumerate(
  inspection_variables
):
  print(
    f"Loading {variable_name} "
    f"({variable_number + 1}/{len(inspection_variables)})"
  )

  first_file = find_first_usable_file(
    cps_files,
    variable_name,
  )

  last_file = find_last_usable_file(
    cps_files,
    variable_name,
  )

  selected_files = [
    file
    for file in [
      first_file,
      last_file,
    ]
    if file is not None
  ]

  selected_files = list(
    dict.fromkeys(
      selected_files
    )
  )

  if len(selected_files) == 0:
    print(
      f"  No usable files found for {variable_name}."
    )
    continue

  selected_periods = [
    "First month",
    "Last month",
  ][:len(selected_files)]

  for selected_file_number, file in enumerate(
    selected_files
  ):
    print(
      f"  Reading {file.name}"
    )

    match = re.match(
      r"cps_basic_(\d{4})_(\d{2})\.csv$",
      file.name,
    )

    if match is None:
      raise RuntimeError(
        f"Unexpected CPS filename: {file.name}"
      )

    month_label = (
      f"{match.group(1)}-{match.group(2)}"
    )

    read_columns = list(
      dict.fromkeys(
        [
          variable_name,
          "PERSON_WEIGHT",
        ]
      )
    )

    month_data = pd.read_csv(
      file,
      usecols=read_columns,
    )

    month_data = month_data.sample(
      frac=0.10,
      random_state=(
        12345
        + (variable_number + 1) * 100
        + (selected_file_number + 1)
      ),
    )

    inspection_weight = pd.to_numeric(
      month_data["PERSON_WEIGHT"],
      errors="coerce",
    )

    if variable_name == "PERSON_WEIGHT":
      inspection_weight = pd.Series(
        np.ones(
          len(month_data)
        ),
        index=month_data.index,
      )

    sampled_data = pd.DataFrame(
      {
        "variable": variable_name,
        "Period": (
          selected_periods[
            selected_file_number
          ]
        ),
        "Month": month_label,
        "value": pd.to_numeric(
          month_data[
            variable_name
          ],
          errors="coerce",
        ),
        "weight": inspection_weight,
      }
    )

    sampled_data = sampled_data[
      np.isfinite(
        sampled_data["value"]
      )
      & np.isfinite(
        sampled_data["weight"]
      )
      & (
        sampled_data["weight"]
        > 0
      )
    ]

    inspection_data_parts.append(
      sampled_data
    )

    del month_data, sampled_data


if len(inspection_data_parts) > 0:
  cps_inspection_data = pd.concat(
    inspection_data_parts,
    ignore_index=True,
  )

  cps_inspection_data.to_csv(
    FILTERED_DIR
    / "cps_inspection_data.csv",
    index=False,
  )

else:
  cps_inspection_data = pd.DataFrame()

del inspection_data_parts
gc.collect()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% 1. REGIONAL UNEMPLOYED PER JOB OPENING
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

load_dotenv(".env")
BLS_API_KEY = os.getenv(
  "BLS_API_KEY",
  "",
)

if not BLS_API_KEY:
  raise RuntimeError(
    "BLS_API_KEY was not found in .env."
  )

regions = [
  "Midwest",
  "Northeast",
  "South",
  "West",
]

regional_series = pd.DataFrame(
  {
    "Region": np.repeat(
      regions,
      2,
    ),
    "Measure": [
      "Job openings",
      "Unemployed",
    ] * 4,
    "series_id": [
      "JTS000000MW0000000JOL",
      "LASRD920000000000004",
      "JTS000000NE0000000JOL",
      "LASRD910000000000004",
      "JTS000000SO0000000JOL",
      "LASRD930000000000004",
      "JTS000000WE0000000JOL",
      "LASRD940000000000004",
    ],
  }
)


def fetch_bls_region(
  start_year,
  end_year,
):
  payload = {
    "seriesid": (
      regional_series[
        "series_id"
      ].tolist()
    ),
    "startyear": str(
      start_year
    ),
    "endyear": str(
      end_year
    ),
    "registrationkey": BLS_API_KEY,
  }

  response = requests.post(
    "https://api.bls.gov/publicAPI/v2/timeseries/data/",
    json=payload,
    timeout=60,
  )

  response.raise_for_status()
  result = response.json()

  if (
    result.get("status")
    != "REQUEST_SUCCEEDED"
  ):
    message = "; ".join(
      result.get(
        "message",
        [],
      )
    )

    raise RuntimeError(
      message
      or "BLS API request failed."
    )

  rows = []

  for series in (
    result.get(
      "Results",
      {},
    ).get(
      "series",
      [],
    )
  ):
    series_id = series.get(
      "seriesID"
    )

    for obs in series.get(
      "data",
      [],
    ):
      period = obs.get(
        "period",
        "",
      )

      if not re.fullmatch(
        r"M(0[1-9]|1[0-2])",
        period,
      ):
        continue

      value = pd.to_numeric(
        obs.get(
          "value"
        ),
        errors="coerce",
      )

      if not np.isfinite(
        value
      ):
        continue

      rows.append(
        {
          "series_id": series_id,
          "Date": pd.Timestamp(
            year=int(
              obs[
                "year"
              ]
            ),
            month=int(
              period[1:]
            ),
            day=1,
          ),
          "value": float(
            value
          ),
        }
      )

  return pd.DataFrame(
    rows
  )


regional_start_year = 2001
regional_end_year = date.today().year

regional_year_windows = [
  (
    start_year,
    min(
      start_year + 9,
      regional_end_year,
    ),
  )
  for start_year in range(
    regional_start_year,
    regional_end_year + 1,
    10,
  )
]

print(
  f"Downloading {len(regional_year_windows)} "
  "BLS year windows..."
)

regional_parts = []

with ThreadPoolExecutor(
  max_workers=min(
    4,
    len(regional_year_windows),
  )
) as executor:
  futures = {
    executor.submit(
      fetch_bls_region,
      start_year,
      end_year,
    ): (
      start_year,
      end_year,
    )
    for start_year, end_year
    in regional_year_windows
  }

  completed = 0

  for future in as_completed(
    futures
  ):
    start_year, end_year = (
      futures[
        future
      ]
    )

    window_data = future.result()

    regional_parts.append(
      window_data
    )

    completed += 1

    print(
      f"[{completed}/{len(regional_year_windows)}] "
      f"Finished {start_year}–{end_year}: "
      f"{len(window_data):,} observations",
      flush=True,
    )


regional_raw = pd.concat(
  regional_parts,
  ignore_index=True,
)

del regional_parts

regional_raw = (
  regional_raw
  .drop_duplicates(
    [
      "series_id",
      "Date",
    ]
  )
  .merge(
    regional_series,
    on="series_id",
    how="left",
  )
)

regional_unemployed_per_opening = (
  regional_raw[
    [
      "Region",
      "Measure",
      "Date",
      "value",
    ]
  ]
  .pivot_table(
    index=[
      "Region",
      "Date",
    ],
    columns="Measure",
    values="value",
    aggfunc="first",
  )
  .reset_index()
  .dropna(
    subset=[
      "Job openings",
      "Unemployed",
    ]
  )
)

regional_unemployed_per_opening[
  "unemployed_per_job_opening"
] = (
  (
    regional_unemployed_per_opening[
      "Unemployed"
    ]
    / 1000
  )
  / regional_unemployed_per_opening[
    "Job openings"
  ]
)

regional_unemployed_per_opening[
  "Region"
] = pd.Categorical(
  regional_unemployed_per_opening[
    "Region"
  ],
  categories=regions,
  ordered=True,
)

regional_unemployed_per_opening = (
  regional_unemployed_per_opening[
    np.isfinite(
      regional_unemployed_per_opening[
        "unemployed_per_job_opening"
      ]
    )
  ]
  .sort_values(
    [
      "Region",
      "Date",
    ]
  )
)

regional_unemployed_per_opening.to_csv(
  FILTERED_DIR
  / "regional_unemployed_per_job_opening.csv",
  index=False,
)

region_colors = {
  "Midwest": colors[6],
  "Northeast": colors[10],
  "South": colors[1],
  "West": colors[9],
}

fig_regional_openings = go.Figure()

for region_name in regions:
  region_data = (
    regional_unemployed_per_opening[
      regional_unemployed_per_opening[
        "Region"
      ]
      == region_name
    ]
  )

  fig_regional_openings.add_trace(
    go.Scatter(
      x=region_data[
        "Date"
      ],
      y=region_data[
        "unemployed_per_job_opening"
      ],
      mode="lines",
      name=region_name,
      opacity=1,
      line={
        "color": region_colors[
          region_name
        ],
        "width": 3,
      },
      hovertemplate=(
        f"<b>{region_name}</b><br>"
        "Unemployed per opening: <b>%{y:.2f}</b>"
        "<extra></extra>"
      ),
    )
  )

project_layout(
  fig=fig_regional_openings,
  title=(
    "Monthly Unemployed Persons "
    "per Job Opening by Region"
  ),
  subtitle=(
    "BLS JOLTS job openings and "
    "regional unemployment estimates"
  ),
  xaxis=project_axis(
    title="Year"
  ),
  yaxis=project_axis(
    title=(
      "Unemployed persons "
      "per job opening"
    ),
    zeroline=True,
  ),
  legend_title="Region",
  hovermode="x unified",
  margin={
    "l": 90,
    "r": 40,
    "b": 115,
    "t": 105,
  },
)

fig_regional_openings.update_layout(
  hoversort="value descending",
)

show_figure(
  fig_regional_openings,
  filename=(
    "regional_unemployed_per_job_opening"
  ),
)

del regional_raw
gc.collect()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% 2. STATE JOB-OPENING-RATE MAP
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

STATE_INFO = [
  ("Alabama", "AL", "01"),
  ("Alaska", "AK", "02"),
  ("Arizona", "AZ", "04"),
  ("Arkansas", "AR", "05"),
  ("California", "CA", "06"),
  ("Colorado", "CO", "08"),
  ("Connecticut", "CT", "09"),
  ("Delaware", "DE", "10"),
  ("Florida", "FL", "12"),
  ("Georgia", "GA", "13"),
  ("Hawaii", "HI", "15"),
  ("Idaho", "ID", "16"),
  ("Illinois", "IL", "17"),
  ("Indiana", "IN", "18"),
  ("Iowa", "IA", "19"),
  ("Kansas", "KS", "20"),
  ("Kentucky", "KY", "21"),
  ("Louisiana", "LA", "22"),
  ("Maine", "ME", "23"),
  ("Maryland", "MD", "24"),
  ("Massachusetts", "MA", "25"),
  ("Michigan", "MI", "26"),
  ("Minnesota", "MN", "27"),
  ("Mississippi", "MS", "28"),
  ("Missouri", "MO", "29"),
  ("Montana", "MT", "30"),
  ("Nebraska", "NE", "31"),
  ("Nevada", "NV", "32"),
  ("New Hampshire", "NH", "33"),
  ("New Jersey", "NJ", "34"),
  ("New Mexico", "NM", "35"),
  ("New York", "NY", "36"),
  ("North Carolina", "NC", "37"),
  ("North Dakota", "ND", "38"),
  ("Ohio", "OH", "39"),
  ("Oklahoma", "OK", "40"),
  ("Oregon", "OR", "41"),
  ("Pennsylvania", "PA", "42"),
  ("Rhode Island", "RI", "44"),
  ("South Carolina", "SC", "45"),
  ("South Dakota", "SD", "46"),
  ("Tennessee", "TN", "47"),
  ("Texas", "TX", "48"),
  ("Utah", "UT", "49"),
  ("Vermont", "VT", "50"),
  ("Virginia", "VA", "51"),
  ("Washington", "WA", "53"),
  ("West Virginia", "WV", "54"),
  ("Wisconsin", "WI", "55"),
  ("Wyoming", "WY", "56"),
  ("District of Columbia", "DC", "11"),
]

states = pd.DataFrame(
  STATE_INFO,
  columns=[
    "State",
    "Abbreviation",
    "state_code",
  ],
)

states["series_id"] = (
  "JTS000000"
  + states["state_code"]
  + "0000000JOR"
)


def fetch_bls_state_window(
  ids,
  start_year,
  end_year,
  batch_size=25,
):
  all_rows = []
  ids = list(
    ids
  )

  for start in range(
    0,
    len(ids),
    batch_size,
  ):
    batch = ids[
      start:start + batch_size
    ]

    payload = {
      "seriesid": batch,
      "startyear": str(
        start_year
      ),
      "endyear": str(
        end_year
      ),
      "registrationkey": BLS_API_KEY,
    }

    response = requests.post(
      "https://api.bls.gov/publicAPI/v2/timeseries/data/",
      json=payload,
      timeout=60,
    )

    response.raise_for_status()
    result = response.json()

    if (
      result.get("status")
      != "REQUEST_SUCCEEDED"
    ):
      message = "; ".join(
        result.get(
          "message",
          [],
        )
      )

      raise RuntimeError(
        message
        or "BLS API request failed."
      )

    for series in (
      result.get(
        "Results",
        {},
      ).get(
        "series",
        [],
      )
    ):
      series_id = series.get(
        "seriesID"
      )

      for obs in series.get(
        "data",
        [],
      ):
        period = obs.get(
          "period",
          "",
        )

        if not re.fullmatch(
          r"M(0[1-9]|1[0-2])",
          period,
        ):
          continue

        value = pd.to_numeric(
          obs.get(
            "value"
          ),
          errors="coerce",
        )

        if not np.isfinite(
          value
        ):
          continue

        all_rows.append(
          {
            "series_id": series_id,
            "Date": pd.Timestamp(
              year=int(
                obs[
                  "year"
                ]
              ),
              month=int(
                period[
                  1:
                ]
              ),
              day=1,
            ),
            "job_opening_rate": float(
              value
            ),
          }
        )

  return pd.DataFrame(
    all_rows
  )


map_start_year = 2000
map_end_year = date.today().year

map_year_windows = [
  (
    start_year,
    min(
      start_year + 9,
      map_end_year,
    ),
  )
  for start_year in range(
    map_start_year,
    map_end_year + 1,
    10,
  )
]

state_job_opening_history = pd.concat(
  [
    fetch_bls_state_window(
      ids=states[
        "series_id"
      ],
      start_year=start_year,
      end_year=end_year,
    )
    for start_year, end_year
    in map_year_windows
  ],
  ignore_index=True,
)

state_job_opening_history = (
  state_job_opening_history
  .drop_duplicates(
    [
      "series_id",
      "Date",
    ]
  )
  .merge(
    states,
    on="series_id",
    how="left",
  )
)

state_job_opening_history = (
  state_job_opening_history[
    state_job_opening_history[
      "Date"
    ].notna()
    & np.isfinite(
      state_job_opening_history[
        "job_opening_rate"
      ]
    )
    & state_job_opening_history[
      "Abbreviation"
    ].notna()
  ]
  .sort_values(
    [
      "Date",
      "State",
    ]
  )
)

if len(
  state_job_opening_history
) == 0:
  raise RuntimeError(
    "The BLS API returned no usable "
    "state job-opening-rate observations."
  )

state_job_opening_history.to_csv(
  FILTERED_DIR
  / "state_job_opening_rate_history.csv",
  index=False,
)

state_map_data = (
  state_job_opening_history.copy()
)

state_map_data[
  "frame_date"
] = state_map_data[
  "Date"
].dt.strftime(
  "%Y-%m"
)

state_map_data[
  "date_label"
] = state_map_data[
  "Date"
].dt.strftime(
  "%B %Y"
)


# ------------------------------------------------------------
# Discrete job-opening-rate categories
# ------------------------------------------------------------

job_opening_rate_order = [
  "1.3–3%",
  "3–4%",
  "4–6%",
  "6%+",
]

job_opening_rate_colors = {
  "1.3–3%": colors[7],
  "3–4%": colors[6],
  "4–6%": colors[0],
  "6%+": colors[2],
}

state_map_data[
  "job_opening_rate_category"
] = pd.cut(
  state_map_data[
    "job_opening_rate"
  ],
  bins=[
    1.3,
    3.0,
    4.0,
    6.0,
    np.inf,
  ],
  labels=job_opening_rate_order,
  right=False,
  include_lowest=True,
)

state_map_data[
  "job_opening_rate_category"
] = pd.Categorical(
  state_map_data[
    "job_opening_rate_category"
  ],
  categories=job_opening_rate_order,
  ordered=True,
)


# ------------------------------------------------------------
# Animation frame order
# ------------------------------------------------------------

map_frames = (
  state_map_data[
    [
      "Date",
      "frame_date",
      "date_label",
    ]
  ]
  .drop_duplicates()
  .sort_values(
    "Date"
  )
)

map_frame_order = (
  map_frames[
    "frame_date"
  ].tolist()
)

map_frame_labels = dict(
  zip(
    map_frames[
      "frame_date"
    ],
    map_frames[
      "date_label"
    ],
  )
)


# ------------------------------------------------------------
# Animated choropleth traces
# ------------------------------------------------------------

def build_state_map_trace(
  frame_data,
  category,
):
  trace_data = frame_data[
    frame_data[
      "job_opening_rate_category"
    ]
    == category
  ].copy()

  return go.Choropleth(
    locations=trace_data[
      "Abbreviation"
    ],
    locationmode="USA-states",
    z=np.ones(
      len(
        trace_data
      )
    ),
    colorscale=[
      [
        0,
        job_opening_rate_colors[
          category
        ],
      ],
      [
        1,
        job_opening_rate_colors[
          category
        ],
      ],
    ],
    zmin=0,
    zmax=1,
    showscale=False,
    showlegend=False,
    marker={
      "line": {
        "color": "white",
        "width": 2,
      },
    },
    text=trace_data[
      "State"
    ],
    customdata=trace_data[
      [
        "job_opening_rate",
        "date_label",
      ]
    ].to_numpy(),
    hovertemplate=(
      "<b>%{text}</b><br>"
      "JOR: "
      "<b>%{customdata[0]:.1f}%</b><br>"
      "<extra></extra>"
    ),
    hoverlabel={
      "bgcolor": "white",
      "bordercolor": "#D9D9D9",
      "font": {
        "family": GRAPH_FONT,
        "size": GRAPH_TICK_SIZE,
        "color": graph_colors[
          "text"
        ],
      },
    },
  )


# ------------------------------------------------------------
# Permanent legend traces
#
# These traces are never updated by the animation, so all four
# legend categories remain visible for every month.
# ------------------------------------------------------------

def build_state_map_legend_trace(
  category,
):
  return go.Scattergeo(
    lon=[
      None
    ],
    lat=[
      None
    ],
    mode="markers",
    marker={
      "size": 14,
      "color": (
        job_opening_rate_colors[
          category
        ]
      ),
      "symbol": "square",
    },
    name=category,
    showlegend=True,
    hoverinfo="skip",
    visible=True,
  )


# ------------------------------------------------------------
# Initial frame
# ------------------------------------------------------------

first_map_frame = map_frame_order[
  0
]

first_map_data = state_map_data[
  state_map_data[
    "frame_date"
  ]
  == first_map_frame
]


# ------------------------------------------------------------
# Figure
#
# Traces 0–3 are animated map traces.
# Traces 4–7 are permanent legend traces.
# ------------------------------------------------------------

fig_state_map = go.Figure(
  data=[
    *[
      build_state_map_trace(
        first_map_data,
        category,
      )
      for category
      in job_opening_rate_order
    ],
    *[
      build_state_map_legend_trace(
        category
      )
      for category
      in job_opening_rate_order
    ],
  ],
  frames=[
    go.Frame(
      name=frame_date,
      traces=[
        0,
        1,
        2,
        3,
      ],
      data=[
        build_state_map_trace(
          state_map_data[
            state_map_data[
              "frame_date"
            ]
            == frame_date
          ],
          category,
        )
        for category
        in job_opening_rate_order
      ],
    )
    for frame_date
    in map_frame_order
  ],
)


# ------------------------------------------------------------
# Geography
# ------------------------------------------------------------

fig_state_map.update_geos(
  scope="usa",
  projection_type="albers usa",
  bgcolor=graph_colors[
    "paper"
  ],
  showlakes=False,
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

fig_state_map.update_layout(
  title={
    "text": (
      "State Job Opening Rate Over Time"
    ),
    "x": 0.02,
    "xanchor": "left",
    "font": {
      "family": GRAPH_FONT,
      "size": GRAPH_TITLE_SIZE,
      "color": graph_colors[
        "text"
      ],
    },
  },
  font={
    "family": GRAPH_FONT,
    "size": GRAPH_TICK_SIZE,
    "color": graph_colors[
      "text"
    ],
  },
  paper_bgcolor=graph_colors[
    "paper"
  ],
  plot_bgcolor=graph_colors[
    "paper"
  ],
  margin={
    "l": 10,
    "r": 10,
    "b": 145,
    "t": 70,
  },
  hoverlabel={
    "bgcolor": "white",
    "bordercolor": "#D9D9D9",
    "font": {
      "family": GRAPH_FONT,
      "size": GRAPH_TICK_SIZE,
      "color": graph_colors[
        "text"
      ],
    },
  },
  legend={
    "title": {
      "text": (
        "Job Opening Rate"
      ),
      "font": {
        "family": GRAPH_FONT,
        "size": GRAPH_TICK_SIZE,
      },
    },
    "font": {
      "family": GRAPH_FONT,
      "size": GRAPH_TICK_SIZE,
    },
    "traceorder": "reversed",
    "itemclick": False,
    "itemdoubleclick": False,
  },
  updatemenus=[
    {
      "type": "buttons",
      "direction": "left",
      "x": 0,
      "xanchor": "left",
      "y": -0.20,
      "yanchor": "top",
      "showactive": False,
      "buttons": [
        {
          "label": "Play",
          "method": "animate",
          "args": [
            None,
            {
              "frame": {
                "duration": 100,
                "redraw": True,
              },
              "transition": {
                "duration": 0,
              },
              "fromcurrent": True,
              "mode": "immediate",
            },
          ],
        },
        {
          "label": "Pause",
          "method": "animate",
          "args": [
            [
              None
            ],
            {
              "frame": {
                "duration": 0,
                "redraw": False,
              },
              "transition": {
                "duration": 0,
              },
              "mode": "immediate",
            },
          ],
        },
      ],
    }
  ],
  sliders=[
    {
      "active": 0,
      "currentvalue": {
        "prefix": "Month: ",
        "font": {
          "family": GRAPH_FONT,
          "size": GRAPH_TICK_SIZE,
        },
      },
      "pad": {
        "t": 35,
        "b": 10,
      },
      "steps": [
        {
          "label": map_frame_labels[
            frame_date
          ],
          "method": "animate",
          "args": [
            [
              frame_date
            ],
            {
              "mode": "immediate",
              "frame": {
                "duration": 100,
                "redraw": True,
              },
              "transition": {
                "duration": 0,
              },
            },
          ],
        }
        for frame_date
        in map_frame_order
      ],
    }
  ],
)

show_figure(
  fig_state_map,
  filename=(
    "state_job_opening_rate_over_time"
  ),
)

gc.collect()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% 3. MONTHLY LABOR-FORCE PARTICIPATION RATE BY NATIVITY
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

lfpr_files = get_cps_files()
lfpr_parts = []

for file in lfpr_files:
  required_columns = [
    "HRYEAR4",
    "HRMONTH",
    "FOREIGN_BORN",
    "IN_CIVILIAN_POPULATION",
    "IN_LABOR_FORCE",
    "PERSON_WEIGHT",
  ]

  if not file_has_columns(
    file,
    required_columns,
  ):
    continue

  data = pd.read_csv(
    file,
    usecols=required_columns,
  )

  data["Date"] = cps_month_date(
    data
  )

  data["Nativity"] = np.select(
    [
      data[
        "FOREIGN_BORN"
      ] == 0,
      data[
        "FOREIGN_BORN"
      ] == 1,
    ],
    [
      "Native born",
      "Foreign born",
    ],
    default=None,
  )

  data = data[
    data[
      "Nativity"
    ].notna()
    & (
      data[
        "IN_CIVILIAN_POPULATION"
      ] == 1
    )
    & np.isfinite(
      data[
        "PERSON_WEIGHT"
      ]
    )
    & (
      data[
        "PERSON_WEIGHT"
      ] > 0
    )
  ]

  if len(data) > 0:
    grouped_rows = []

    for (
      month_date,
      nativity,
    ), group in data.groupby(
      [
        "Date",
        "Nativity",
      ],
      sort=False,
    ):
      civilian_population = group[
        "PERSON_WEIGHT"
      ].sum()

      labor_force = group.loc[
        group[
          "IN_LABOR_FORCE"
        ] == 1,
        "PERSON_WEIGHT",
      ].sum()

      grouped_rows.append(
        {
          "Date": month_date,
          "Nativity": nativity,
          "civilian_population": civilian_population,
          "labor_force": labor_force,
          "lfpr": (
            100
            * labor_force
            / civilian_population
          ),
        }
      )

    lfpr_parts.append(
      pd.DataFrame(
        grouped_rows
      )
    )

  del data


if len(lfpr_parts) == 0:
  raise RuntimeError(
    "No usable LFPR observations were found."
  )

lfpr_monthly = pd.concat(
  lfpr_parts,
  ignore_index=True,
).sort_values(
  [
    "Date",
    "Nativity",
  ]
)

lfpr_monthly.to_csv(
  FILTERED_DIR
  / "lfpr_monthly.csv",
  index=False,
)

fig_lfpr_monthly = go.Figure()

for nativity_group in nativity_colors:
  nativity_data = (
    lfpr_monthly[
      lfpr_monthly[
        "Nativity"
      ]
      == nativity_group
    ]
  )

  fig_lfpr_monthly.add_trace(
    go.Scatter(
      x=nativity_data[
        "Date"
      ],
      y=nativity_data[
        "lfpr"
      ],
      mode="lines",
      name=nativity_group,
      line={
        "color": (
          nativity_colors[
            nativity_group
          ]
        ),
        "width": 3,
      },
      hovertemplate=(
        f"<b>{nativity_group}</b><br>"
        "LFPR: <b>%{y:.1f}%</b>"
        "<extra></extra>"
      ),
    )
  )

project_layout(
  fig=fig_lfpr_monthly,
  title=(
    "Monthly Labor-Force Participation Rate by Nativity"
  ),
  subtitle=(
    "Population-weighted civilian population age 16 or older"
  ),
  xaxis=project_axis(
    title=""
  ),
  yaxis=project_axis(
    title=(
      "Labor-force participation rate"
    ),
    ticksuffix="%",
    range_=[
      0,
      100,
    ],
  ),
  legend_title="",
  hovermode="x unified",
)

fig_lfpr_monthly.update_layout(
  hoversort="value descending",
)

show_figure(
  fig_lfpr_monthly,
  filename=(
    "monthly_lfpr_by_nativity"
  ),
)

del lfpr_parts
gc.collect()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% 4. INTERACTIVE POPULATION PYRAMID
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

population_age_labels = [
  *[
    f"{start}–{start + 4}"
    for start in range(
      0,
      85,
      5,
    )
  ],
  "85+",
]

population_pyramid_files = get_cps_files()
population_pyramid_parts = []

for file in population_pyramid_files:
  required_columns = [
    "HRYEAR4",
    "HRMONTH",
    "PRTAGE",
    "PESEX",
    "FOREIGN_BORN",
    "PERSON_WEIGHT",
  ]

  if not file_has_columns(
    file,
    required_columns,
  ):
    continue

  data = pd.read_csv(
    file,
    usecols=required_columns,
  )

  data["Date"] = cps_month_date(
    data
  )

  data["Sex"] = np.select(
    [
      data["PESEX"] == 1,
      data["PESEX"] == 2,
    ],
    [
      "Male",
      "Female",
    ],
    default=None,
  )

  data["Nativity"] = np.select(
    [
      data[
        "FOREIGN_BORN"
      ] == 0,
      data[
        "FOREIGN_BORN"
      ] == 1,
    ],
    [
      "Native born",
      "Foreign born",
    ],
    default=None,
  )

  age_breaks = (
    [-np.inf]
    + list(
      range(
        4,
        85,
        5,
      )
    )
    + [np.inf]
  )

  data[
    "age_band"
  ] = pd.cut(
    data[
      "PRTAGE"
    ],
    bins=age_breaks,
    labels=population_age_labels,
    right=True,
    ordered=True,
  )

  data = data[
    np.isfinite(
      data[
        "PERSON_WEIGHT"
      ]
    )
    & (
      data[
        "PERSON_WEIGHT"
      ] > 0
    )
    & data[
      "Sex"
    ].notna()
    & data[
      "Nativity"
    ].notna()
    & data[
      "age_band"
    ].notna()
  ]

  if len(data) > 0:
    monthly = (
      data
      .groupby(
        [
          "Date",
          "age_band",
          "Sex",
          "Nativity",
        ],
        observed=True,
        sort=False,
      )[
        "PERSON_WEIGHT"
      ]
      .sum()
      .rename(
        "population"
      )
      .reset_index()
    )

    population_pyramid_parts.append(
      monthly
    )

  del data


if len(
  population_pyramid_parts
) == 0:
  raise RuntimeError(
    "No usable observations were available "
    "for the population pyramid."
  )

population_pyramid_monthly = pd.concat(
  population_pyramid_parts,
  ignore_index=True,
)

population_pyramid_monthly.to_csv(
  FILTERED_DIR
  / "population_pyramid_monthly.csv",
  index=False,
)

population_pyramid_monthly[
  "age_band"
] = pd.Categorical(
  population_pyramid_monthly[
    "age_band"
  ],
  categories=population_age_labels,
  ordered=True,
)

pyramid_rolling_parts = []

for keys, group in (
  population_pyramid_monthly.groupby(
    [
      "age_band",
      "Sex",
      "Nativity",
    ],
    observed=True,
    sort=False,
  )
):
  group = group.sort_values(
    "Date"
  ).copy()

  group[
    "population_12m"
  ] = group[
    "population"
  ].rolling(
    window=12,
    min_periods=12,
  ).mean()

  group[
    "window_start"
  ] = group[
    "Date"
  ].shift(
    11
  )

  pyramid_rolling_parts.append(
    group
  )

pyramid_rolling = pd.concat(
  pyramid_rolling_parts,
  ignore_index=True,
).dropna(
  subset=[
    "population_12m",
    "window_start",
  ]
)

pyramid_rolling[
  "window_label"
] = (
  pyramid_rolling[
    "window_start"
  ].dt.strftime(
    "%b %Y"
  )
  + " – "
  + pyramid_rolling[
    "Date"
  ].dt.strftime(
    "%b %Y"
  )
)

pyramid_rolling[
  "plot_population"
] = np.where(
  pyramid_rolling[
    "Sex"
  ] == "Male",
  -pyramid_rolling[
    "population_12m"
  ],
  pyramid_rolling[
    "population_12m"
  ],
)

pyramid_windows = (
  pyramid_rolling[
    [
      "Date",
      "window_label",
    ]
  ]
  .drop_duplicates()
  .sort_values(
    "Date"
  )
)

window_order = (
  pyramid_windows[
    "window_label"
  ].tolist()
)

pyramid_rolling[
  "window_label"
] = pd.Categorical(
  pyramid_rolling[
    "window_label"
  ],
  categories=window_order,
  ordered=True,
)

pyramid_rolling.to_csv(
  FILTERED_DIR
  / "population_pyramid_rolling_12_month.csv",
  index=False,
)

pyramid_limit = (
  pyramid_rolling[
    "plot_population"
  ].abs().max()
)

if pyramid_limit >= 10_000_000:
  pyramid_rounding_unit = 1_000_000

elif pyramid_limit >= 1_000_000:
  pyramid_rounding_unit = 100_000

elif pyramid_limit >= 100_000:
  pyramid_rounding_unit = 10_000

elif pyramid_limit >= 10_000:
  pyramid_rounding_unit = 1_000

else:
  pyramid_rounding_unit = 100

pyramid_limit = (
  math.ceil(
    pyramid_limit
    / pyramid_rounding_unit
  )
  * pyramid_rounding_unit
)

pyramid_tickvals = np.linspace(
  -pyramid_limit,
  pyramid_limit,
  7,
)

if pyramid_limit >= 1_000_000:
  pyramid_ticktext = [
    f"{abs(value) / 1_000_000:.1f}M"
    for value
    in pyramid_tickvals
  ]

else:
  pyramid_ticktext = [
    f"{abs(value) / 1_000:.1f}K"
    for value
    in pyramid_tickvals
  ]

pyramid_trace_specs = [
  {
    "nativity": "Native born",
    "sex": "Male",
    "width": .95,
    "color": "#0F374B",
    "name": "Native-born male",
    "opacity": 1,
  },
  {
    "nativity": "Native born",
    "sex": "Female",
    "width": .95,
    "color": "#4B0F27",
    "name": "Native-born female",
    "opacity": 1,
  },
  {
    "nativity": "Foreign born",
    "sex": "Male",
    "width": 0.4,
    "color": "#9DD9F7",
    "name": "Foreign-born male",
    "opacity": 1,
  },
  {
    "nativity": "Foreign born",
    "sex": "Female",
    "width": 0.4,
    "color": "#F79DDB",
    "name": "Foreign-born female",
    "opacity": 1,
  },
]


def build_pyramid_trace(
  frame_data,
  trace_spec,
):
  trace_data = frame_data[
    (
      frame_data[
        "Nativity"
      ]
      == trace_spec[
        "nativity"
      ]
    )
    & (
      frame_data[
        "Sex"
      ]
      == trace_spec[
        "sex"
      ]
    )
  ].copy()

  trace_data[
    "age_band"
  ] = pd.Categorical(
    trace_data[
      "age_band"
    ],
    categories=population_age_labels,
    ordered=True,
  )

  trace_data = trace_data.sort_values(
    "age_band"
  )

  return go.Bar(
    x=trace_data[
      "plot_population"
    ],
    y=trace_data[
      "age_band"
    ].astype(
      str
    ),
    orientation="h",
    width=trace_spec[
      "width"
    ],
    name=trace_spec[
      "name"
    ],
    marker={
      "color": trace_spec[
        "color"
      ],
      "line": {
        "color": "white",
        "width": 0.6,
      },
    },
    opacity=trace_spec[
      "opacity"
    ],
    customdata=trace_data[
      [
        "population_12m"
      ]
    ].to_numpy(),
    hovertemplate=(
      f"{trace_spec['name']}<br>"
      "Age: %{y}<br>"
      "Average monthly population: "
      "%{customdata[0]:,.0f}"
      "<extra></extra>"
    ),
  )


first_window = window_order[
  0
]

first_frame_data = (
  pyramid_rolling[
    pyramid_rolling[
      "window_label"
    ]
    == first_window
  ]
)

fig_pyramid = go.Figure(
  data=[
    build_pyramid_trace(
      first_frame_data,
      trace_spec,
    )
    for trace_spec
    in pyramid_trace_specs
  ],
  frames=[
    go.Frame(
      name=window_label,
      data=[
        build_pyramid_trace(
          pyramid_rolling[
            pyramid_rolling[
              "window_label"
            ]
            == window_label
          ],
          trace_spec,
        )
        for trace_spec
        in pyramid_trace_specs
      ],
    )
    for window_label
    in window_order
  ],
)

project_layout(
  fig=fig_pyramid,
  title=(
    "U.S. Population Pyramid by Nativity"
  ),
  subtitle=(
    "Average of 12 available monthly estimates; "
    "use play or the slider to change the ending month"
  ),
  xaxis=project_axis(
    title="Population",
    zeroline=True,
    range_=[
      -pyramid_limit,
      pyramid_limit,
    ],
    tickvals=(
      pyramid_tickvals.tolist()
    ),
    ticktext=pyramid_ticktext,
  ),
  yaxis=project_axis(
    title="Age band",
    showgrid=False,
    categoryorder="array",
    categoryarray=(
      population_age_labels
    ),
  ),
  barmode="overlay",
  hovermode="closest",
  margin={
    "l": 95,
    "r": 40,
    "b": 130,
    "t": 105,
  },
)

fig_pyramid.update_layout(
  height=980,
  legend={
    "y": -0.2,
    "yanchor": "top",
  },
  updatemenus=[
    {
      "type": "buttons",
      "direction": "left",
      "x": 1,
      "xanchor": "right",
      "y": -0.04, # spacebottom
      "yanchor": "top",
      "showactive": False,
      "buttons": [
        {
          "label": "Play",
          "method": "animate",
          "args": [
            None,
            {
              "frame": {
                "duration": 100,
                "redraw": True,
              },
              "transition": {
                "duration": 0,
              },
              "fromcurrent": True,
              "mode": "immediate",
            },
          ],
        },
        {
          "label": "Pause",
          "method": "animate",
          "args": [
            [
              None
            ],
            {
              "frame": {
                "duration": 0,
                "redraw": False,
              },
              "transition": {
                "duration": 0,
              },
              "mode": "immediate",
            },
          ],
        },
      ],
    }
  ],
  sliders=[
    {
      "active": 0,
      "x": 0,
      "xanchor": "left",
      "y": 0,
      "yanchor": "top",
      "currentvalue": {
        "prefix": (
          "12-month window: "
        ),
        "font": {
          "family": GRAPH_FONT,
          "size": GRAPH_TICK_SIZE,
        },
      },
      "pad": {
        "t": 35 #pad ticks
      },
      "steps": [
        {
          "label": window_label,
          "method": "animate",
          "args": [
            [
              window_label
            ],
            {
              "mode": "immediate",
              "frame": {
                "duration": 100,
                "redraw": True,
              },
              "transition": {
                "duration": 0,
              },
            },
          ],
        }
        for window_label
        in window_order
      ],
    }
  ],
)

show_figure(
  fig_pyramid,
  filename=(
    "population_pyramid_rolling_slider"
  ),
)

del population_pyramid_parts
del pyramid_rolling_parts
gc.collect()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% 4B. POPULATION PYRAMID BY IMMIGRANT GENERATION
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Mutually exclusive populations:
#   1. Native-born with two native-born parents
#   2. Second generation: native-born with >=1 foreign-born parent
#   3. Foreign born
#
# Bars remain overlaid, with native/native-parent bars widest,
# second generation narrower, and foreign-born narrowest.

parent_native_birth_codes = {
  57,  # United States
  60,  # American Samoa
  66,  # Guam
  73,  # Puerto Rico
  78,  # U.S. Virgin Islands
  96,  # Other U.S. Island Area
}

population_pyramid_gen_overlay_files = get_cps_files()
population_pyramid_gen_overlay_parts = []

for file in population_pyramid_gen_overlay_files:
  required_columns = [
    "HRYEAR4",
    "HRMONTH",
    "PRTAGE",
    "PESEX",
    "FOREIGN_BORN",
    "PEMNTVTY",
    "PEFNTVTY",
    "PERSON_WEIGHT",
  ]

  if not file_has_columns(
    file,
    required_columns,
  ):
    continue

  data = pd.read_csv(
    file,
    usecols=required_columns,
  )

  data["Date"] = cps_month_date(
    data
  )

  data["Sex"] = np.select(
    [
      data[
        "PESEX"
      ] == 1,
      data[
        "PESEX"
      ] == 2,
    ],
    [
      "Male",
      "Female",
    ],
    default=None,
  )

  mother_foreign_born = (
    data[
      "PEMNTVTY"
    ].between(
      100,
      554,
      inclusive="both",
    )
  )

  father_foreign_born = (
    data[
      "PEFNTVTY"
    ].between(
      100,
      554,
      inclusive="both",
    )
  )

  mother_native_born = (
    data[
      "PEMNTVTY"
    ].isin(
      parent_native_birth_codes
    )
  )

  father_native_born = (
    data[
      "PEFNTVTY"
    ].isin(
      parent_native_birth_codes
    )
  )

  data["Series"] = np.select(
    [
      data[
        "FOREIGN_BORN"
      ] == 1,
      (
        (
          data[
            "FOREIGN_BORN"
          ] == 0
        )
        & (
          mother_foreign_born
          | father_foreign_born
        )
      ),
      (
        (
          data[
            "FOREIGN_BORN"
          ] == 0
        )
        & mother_native_born
        & father_native_born
      ),
    ],
    [
      "Foreign born",
      "Second generation",
      "Native born, native parents",
    ],
    default=None,
  )

  age_breaks = (
    [-np.inf]
    + list(
      range(
        4,
        85,
        5,
      )
    )
    + [np.inf]
  )

  data[
    "age_band"
  ] = pd.cut(
    data[
      "PRTAGE"
    ],
    bins=age_breaks,
    labels=population_age_labels,
    right=True,
    ordered=True,
  )

  data = data[
    np.isfinite(
      data[
        "PERSON_WEIGHT"
      ]
    )
    & (
      data[
        "PERSON_WEIGHT"
      ] > 0
    )
    & data[
      "Sex"
    ].notna()
    & data[
      "Series"
    ].notna()
    & data[
      "age_band"
    ].notna()
  ]

  if len(data) > 0:
    monthly_combined = (
      data
      .groupby(
        [
          "Date",
          "age_band",
          "Sex",
          "Series",
        ],
        observed=True,
        sort=False,
      )[
        "PERSON_WEIGHT"
      ]
      .sum()
      .rename(
        "population"
      )
      .reset_index()
    )

    population_pyramid_gen_overlay_parts.append(
      monthly_combined
    )

  del data


if len(
  population_pyramid_gen_overlay_parts
) == 0:
  raise RuntimeError(
    "No usable observations were available "
    "for the immigrant-generation population pyramid."
  )

population_pyramid_gen_overlay_monthly = pd.concat(
  population_pyramid_gen_overlay_parts,
  ignore_index=True,
)

population_pyramid_gen_overlay_monthly[
  "age_band"
] = pd.Categorical(
  population_pyramid_gen_overlay_monthly[
    "age_band"
  ],
  categories=population_age_labels,
  ordered=True,
)

series_order = [
  "Native born, native parents",
  "Second generation",
  "Foreign born",
]

population_pyramid_gen_overlay_monthly[
  "Series"
] = pd.Categorical(
  population_pyramid_gen_overlay_monthly[
    "Series"
  ],
  categories=series_order,
  ordered=True,
)

population_pyramid_gen_overlay_monthly.to_csv(
  FILTERED_DIR
  / "population_pyramid_generation_monthly.csv",
  index=False,
)

pyramid_gen_overlay_rolling_parts = []

for keys, group in (
  population_pyramid_gen_overlay_monthly.groupby(
    [
      "age_band",
      "Sex",
      "Series",
    ],
    observed=True,
    sort=False,
  )
):
  group = group.sort_values(
    "Date"
  ).copy()

  group[
    "population_12m"
  ] = group[
    "population"
  ].rolling(
    window=12,
    min_periods=12,
  ).mean()

  group[
    "window_start"
  ] = group[
    "Date"
  ].shift(
    11
  )

  pyramid_gen_overlay_rolling_parts.append(
    group
  )

pyramid_gen_overlay_rolling = pd.concat(
  pyramid_gen_overlay_rolling_parts,
  ignore_index=True,
).dropna(
  subset=[
    "population_12m",
    "window_start",
  ]
)

pyramid_gen_overlay_rolling[
  "window_label"
] = (
  pyramid_gen_overlay_rolling[
    "window_start"
  ].dt.strftime(
    "%b %Y"
  )
  + " – "
  + pyramid_gen_overlay_rolling[
    "Date"
  ].dt.strftime(
    "%b %Y"
  )
)

pyramid_gen_overlay_rolling[
  "plot_population"
] = np.where(
  pyramid_gen_overlay_rolling[
    "Sex"
  ] == "Male",
  -pyramid_gen_overlay_rolling[
    "population_12m"
  ],
  pyramid_gen_overlay_rolling[
    "population_12m"
  ],
)

pyramid_gen_overlay_windows = (
  pyramid_gen_overlay_rolling[
    [
      "Date",
      "window_label",
    ]
  ]
  .drop_duplicates()
  .sort_values(
    "Date"
  )
)

generation_overlay_window_order = (
  pyramid_gen_overlay_windows[
    "window_label"
  ].tolist()
)

pyramid_gen_overlay_rolling[
  "window_label"
] = pd.Categorical(
  pyramid_gen_overlay_rolling[
    "window_label"
  ],
  categories=(
    generation_overlay_window_order
  ),
  ordered=True,
)

pyramid_gen_overlay_rolling.to_csv(
  FILTERED_DIR
  / "population_pyramid_generation_rolling_12_month.csv",
  index=False,
)

pyramid_gen_overlay_limit = (
  pyramid_gen_overlay_rolling[
    "plot_population"
  ].abs().max()
)

if pyramid_gen_overlay_limit >= 10_000_000:
  pyramid_gen_overlay_rounding_unit = 1_000_000

elif pyramid_gen_overlay_limit >= 1_000_000:
  pyramid_gen_overlay_rounding_unit = 100_000

elif pyramid_gen_overlay_limit >= 100_000:
  pyramid_gen_overlay_rounding_unit = 10_000

elif pyramid_gen_overlay_limit >= 10_000:
  pyramid_gen_overlay_rounding_unit = 1_000

else:
  pyramid_gen_overlay_rounding_unit = 100

pyramid_gen_overlay_limit = (
  math.ceil(
    pyramid_gen_overlay_limit
    / pyramid_gen_overlay_rounding_unit
  )
  * pyramid_gen_overlay_rounding_unit
)

pyramid_gen_overlay_tickvals = np.linspace(
  -pyramid_gen_overlay_limit,
  pyramid_gen_overlay_limit,
  7,
)

if pyramid_gen_overlay_limit >= 1_000_000:
  pyramid_gen_overlay_ticktext = [
    f"{abs(value) / 1_000_000:.1f}M"
    for value
    in pyramid_gen_overlay_tickvals
  ]

else:
  pyramid_gen_overlay_ticktext = [
    f"{abs(value) / 1_000:.1f}K"
    for value
    in pyramid_gen_overlay_tickvals
  ]

# Adjust these colors here as desired.
pyramid_gen_overlay_trace_specs = [
  {
    "series": (
      "Native born, native parents"
    ),
    "sex": "Male",
    "width": .95,
    "color": "#0F374B",
    "name": (
      "Native-born male, native parents"
    ),
    "opacity": 1,
  },
  {
    "series": (
      "Native born, native parents"
    ),
    "sex": "Female",
    "width": .95,
    "color": "#4B0F27",
    "name": (
      "Native-born female, native parents"
    ),
    "opacity": 1,
  },
  {
    "series": (
      "Second generation"
    ),
    "sex": "Male",
    "width": 0.8,
    "color": "#38baa4",
    "name": (
      "Second-generation male"
    ),
    "opacity": 1,
  },
  {
    "series": (
      "Second generation"
    ),
    "sex": "Female",
    "width": 0.8,
    "color": "#cf516a",
    "name": (
      "Second-generation female"
    ),
    "opacity": 1,
  },
  {
    "series": (
      "Foreign born"
    ),
    "sex": "Male",
    "width": 0.4,
    "color": "#9DD9F7",
    "name": (
      "Foreign-born male"
    ),
    "opacity": 1,
  },
  {
    "series": (
      "Foreign born"
    ),
    "sex": "Female",
    "width": 0.4,
    "color": "#F79DDB",
    "name": (
      "Foreign-born female"
    ),
    "opacity": 1,
  },
]


def build_pyramid_gen_overlay_trace(
  frame_data,
  trace_spec,
):
  trace_data = frame_data[
    (
      frame_data[
        "Series"
      ]
      == trace_spec[
        "series"
      ]
    )
    & (
      frame_data[
        "Sex"
      ]
      == trace_spec[
        "sex"
      ]
    )
  ].copy()

  trace_data[
    "age_band"
  ] = pd.Categorical(
    trace_data[
      "age_band"
    ],
    categories=population_age_labels,
    ordered=True,
  )

  trace_data = trace_data.sort_values(
    "age_band"
  )

  return go.Bar(
    x=trace_data[
      "plot_population"
    ],
    y=trace_data[
      "age_band"
    ].astype(
      str
    ),
    orientation="h",
    width=trace_spec[
      "width"
    ],
    name=trace_spec[
      "name"
    ],
    marker={
      "color": trace_spec[
        "color"
      ],
      "line": {
        "color": "white",
        "width": 0.6,
      },
    },
    opacity=trace_spec[
      "opacity"
    ],
    customdata=trace_data[
      [
        "population_12m"
      ]
    ].to_numpy(),
    hovertemplate=(
      f"{trace_spec['name']}<br>"
      "Age: %{y}<br>"
      "Average monthly population: "
      "%{customdata[0]:,.0f}"
      "<extra></extra>"
    ),
  )


first_generation_window = (
  generation_overlay_window_order[
    0
  ]
)

first_generation_frame_data = (
  pyramid_gen_overlay_rolling[
    pyramid_gen_overlay_rolling[
      "window_label"
    ]
    == first_generation_window
  ]
)

fig_pyramid_gen_overlay = go.Figure(
  data=[
    build_pyramid_gen_overlay_trace(
      first_generation_frame_data,
      trace_spec,
    )
    for trace_spec
    in pyramid_gen_overlay_trace_specs
  ],
  frames=[
    go.Frame(
      name=window_label,
      data=[
        build_pyramid_gen_overlay_trace(
          pyramid_gen_overlay_rolling[
            pyramid_gen_overlay_rolling[
              "window_label"
            ]
            == window_label
          ],
          trace_spec,
        )
        for trace_spec
        in pyramid_gen_overlay_trace_specs
      ],
    )
    for window_label
    in generation_overlay_window_order
  ],
)

project_layout(
  fig=fig_pyramid_gen_overlay,
  title=(
    "U.S. Population Pyramid by Immigrant Generation"
  ),
  subtitle=(
    "Native-born with native-born parents, second generation, "
    "and foreign-born population; 12-month averages"
  ),
  xaxis=project_axis(
    title="Population",
    zeroline=True,
    range_=[
      -pyramid_gen_overlay_limit,
      pyramid_gen_overlay_limit,
    ],
    tickvals=(
      pyramid_gen_overlay_tickvals.tolist()
    ),
    ticktext=(
      pyramid_gen_overlay_ticktext
    ),
  ),
  yaxis=project_axis(
    title="Age band",
    showgrid=False,
    categoryorder="array",
    categoryarray=(
      population_age_labels
    ),
  ),
  barmode="overlay",
  hovermode="closest",
  margin={
    "l": 95,
    "r": 40,
    "b": 155,
    "t": 105,
  },
)

fig_pyramid_gen_overlay.update_layout(
  height=1020,
  legend={
    "y": -0.2,
    "yanchor": "top",
  },
  updatemenus=[
    {
      "type": "buttons",
      "direction": "left",
      "x": 1,
      "xanchor": "right",
      "y": -0.04,
      "yanchor": "top",
      "showactive": False,
      "buttons": [
        {
          "label": "Play",
          "method": "animate",
          "args": [
            None,
            {
              "frame": {
                "duration": 100,
                "redraw": True,
              },
              "transition": {
                "duration": 0,
              },
              "fromcurrent": True,
              "mode": "immediate",
            },
          ],
        },
        {
          "label": "Pause",
          "method": "animate",
          "args": [
            [
              None
            ],
            {
              "frame": {
                "duration": 0,
                "redraw": False,
              },
              "transition": {
                "duration": 0,
              },
              "mode": "immediate",
            },
          ],
        },
      ],
    }
  ],
  sliders=[
    {
      "active": 0,
      "x": 0,
      "xanchor": "left",
      "y": 0,
      "currentvalue": {
        "prefix": (
          "12-month window: "
        ),
        "font": {
          "family": GRAPH_FONT,
          "size": GRAPH_TICK_SIZE,
        },
      },
      "pad": {
        "t": 35
      },
      "steps": [
        {
          "label": window_label,
          "method": "animate",
          "args": [
            [
              window_label
            ],
            {
              "mode": "immediate",
              "frame": {
                "duration": 100,
                "redraw": True,
              },
              "transition": {
                "duration": 0,
              },
            },
          ],
        }
        for window_label
        in generation_overlay_window_order
      ],
    }
  ],
)

show_figure(
  fig_pyramid_gen_overlay,
  filename=(
    "population_pyramid_by_immigrant_generation"
  ),
)

del population_pyramid_gen_overlay_parts
del pyramid_gen_overlay_rolling_parts
gc.collect()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% 5. LABOR-FORCE PARTICIPATION SEASONALITY
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if len(
  lfpr_monthly
) == 0:
  raise RuntimeError(
    "No usable LFPR observations were found. "
    "Check that the derived CPS variables exist."
  )

lfpr_seasonality_base = (
  lfpr_monthly.copy()
)

lfpr_seasonality_base[
  "year"
] = lfpr_seasonality_base[
  "Date"
].dt.year

lfpr_seasonality_base[
  "month_number"
] = lfpr_seasonality_base[
  "Date"
].dt.month

month_abbr = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
]

lfpr_seasonality_base[
  "month_name"
] = pd.Categorical(
  [
    month_abbr[
      month - 1
    ]
    for month
    in lfpr_seasonality_base[
      "month_number"
    ]
  ],
  categories=month_abbr,
  ordered=True,
)

month_counts = (
  lfpr_seasonality_base[
    [
      "year",
      "Date",
      "Nativity",
    ]
  ]
  .drop_duplicates()
  .groupby(
    [
      "year",
      "Nativity",
    ]
  )
  .size()
  .rename(
    "month_count"
  )
  .reset_index()
)

complete_lfpr_years = []

for year, group in month_counts.groupby(
  "year"
):
  if (
    group[
      "Nativity"
    ].nunique()
    == 2
    and (
      group[
        "month_count"
      ]
      == 12
    ).all()
  ):
    complete_lfpr_years.append(
      year
    )

if len(
  complete_lfpr_years
) == 0:
  raise RuntimeError(
    "No complete calendar years were available "
    "for both nativity groups."
  )

seasonality_years = (
  complete_lfpr_years[
    -SEASONAL_YEAR_COUNT:
  ]
)

seasonality_base = (
  lfpr_seasonality_base[
    lfpr_seasonality_base[
      "year"
    ].isin(
      seasonality_years
    )
  ].copy()
)

annual_totals = (
  seasonality_base.groupby(
    [
      "year",
      "Nativity",
    ],
    as_index=False,
  )
  .agg(
    annual_labor_force=(
      "labor_force",
      "sum",
    ),
    annual_civilian_population=(
      "civilian_population",
      "sum",
    ),
  )
)

annual_totals[
  "annual_lfpr"
] = (
  100
  * annual_totals[
    "annual_labor_force"
  ]
  / annual_totals[
    "annual_civilian_population"
  ]
)

seasonality_base = (
  seasonality_base.merge(
    annual_totals[
      [
        "year",
        "Nativity",
        "annual_lfpr",
      ]
    ],
    on=[
      "year",
      "Nativity",
    ],
    how="left",
  )
)

seasonality_base[
  "deviation"
] = (
  seasonality_base[
    "lfpr"
  ]
  - seasonality_base[
    "annual_lfpr"
  ]
)

lfpr_seasonality = (
  seasonality_base.groupby(
    [
      "month_number",
      "month_name",
      "Nativity",
    ],
    observed=True,
    as_index=False,
  )[
    "deviation"
  ]
  .mean()
)

lfpr_seasonality.to_csv(
  FILTERED_DIR
  / "lfpr_seasonality.csv",
  index=False,
)

fig_lfpr_seasonality = (
  go.Figure()
)

for nativity_group in [
  "Native born",
  "Foreign born",
]:
  group_data = (
    lfpr_seasonality[
      lfpr_seasonality[
        "Nativity"
      ]
      == nativity_group
    ]
  )

  fig_lfpr_seasonality.add_trace(
    go.Bar(
      x=group_data[
        "month_name"
      ].astype(
        str
      ),
      y=group_data[
        "deviation"
      ],
      base=0,
      name=nativity_group,
      offsetgroup=nativity_group,
      marker={
        "color": nativity_colors[
          nativity_group
        ],
        "line": {
          "color": "white",
          "width": 0.5,
        },
      },
      hovertemplate=(
        f"<b>{nativity_group}</b><br>"
        "Deviation: %{y:,.2f} percentage points"
        "<extra></extra>"
      ),
    )
  )

project_layout(
  fig=fig_lfpr_seasonality,
  title=(
    "Seasonal Deviation in Labor-Force Participation"
  ),
  subtitle=(
    "Average monthly percentage-point deviation from each annual rate, "
    f"{min(seasonality_years)}–{max(seasonality_years)}"
  ),
  xaxis=project_axis(
    title="Month",
    showgrid=False,
    categoryorder="array",
    categoryarray=month_abbr,
  ),
  yaxis=project_axis(
    title=(
      "Deviation from annual LFPR"
    ),
    ticksuffix=" pp",
    zeroline=True,
  ),
  barmode="group",
  legend_title="",
)

show_figure(
  fig_lfpr_seasonality,
  filename=(
    "labor_force_participation_seasonality"
  ),
)

gc.collect()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% 6. SHARE OF THE FOREIGN-BORN LABOR FORCE HERE 10 OR FEWER YEARS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

recent_share_files = get_cps_files()
recent_share_parts = []

for file in recent_share_files:
  required_columns = [
    "HRYEAR4",
    "HRMONTH",
    "FOREIGN_BORN",
    "RECENT_IMMIGRANT_10YR",
    "IN_LABOR_FORCE",
    "PERSON_WEIGHT",
  ]

  if not file_has_columns(
    file,
    required_columns,
  ):
    continue

  data = pd.read_csv(
    file,
    usecols=required_columns,
  )

  data["Date"] = cps_month_date(
    data
  )

  data = data[
    (
      data[
        "FOREIGN_BORN"
      ] == 1
    )
    & (
      data[
        "IN_LABOR_FORCE"
      ] == 1
    )
    & np.isfinite(
      data[
        "PERSON_WEIGHT"
      ]
    )
    & (
      data[
        "PERSON_WEIGHT"
      ] > 0
    )
  ]

  if len(data) > 0:
    rows = []

    for month_date, group in data.groupby(
      "Date",
      sort=False,
    ):
      foreign_born_labor_force = (
        group[
          "PERSON_WEIGHT"
        ].sum()
      )

      recent_foreign_born_labor_force = (
        group.loc[
          group[
            "RECENT_IMMIGRANT_10YR"
          ] == 1,
          "PERSON_WEIGHT",
        ].sum()
      )

      rows.append(
        {
          "Date": month_date,
          "foreign_born_labor_force": (
            foreign_born_labor_force
          ),
          "recent_foreign_born_labor_force": (
            recent_foreign_born_labor_force
          ),
          "recent_share": (
            100
            * recent_foreign_born_labor_force
            / foreign_born_labor_force
          ),
        }
      )

    recent_share_parts.append(
      pd.DataFrame(
        rows
      )
    )

  del data


if len(
  recent_share_parts
) == 0:
  raise RuntimeError(
    "No usable observations were available "
    "for the recent-immigrant share."
  )

recent_foreign_born_share = (
  pd.concat(
    recent_share_parts,
    ignore_index=True,
  )
  .sort_values(
    "Date"
  )
)

recent_foreign_born_share.to_csv(
  FILTERED_DIR
  / "foreign_born_labor_force_recent_10yr_share.csv",
  index=False,
)

fig_recent_foreign_born_share = (
  go.Figure(
    go.Scatter(
      x=recent_foreign_born_share[
        "Date"
      ],
      y=recent_foreign_born_share[
        "recent_share"
      ],
      mode="lines",
      line={
        "color": colors[
          0
        ],
        "width": 3,
      },
      hovertemplate=(
        "Share here 10 years or fewer: <b>%{y:.1f}%</b>"
        "<extra></extra>"
      ),
    )
  )
)

project_layout(
  fig=fig_recent_foreign_born_share,
  title=(
    "Share of the Foreign-Born Labor Force Here 10 or Fewer Years"
  ),
  subtitle=(
    "Population-weighted monthly share of the foreign-born labor force"
  ),
  xaxis=project_axis(
    title=""
  ),
  yaxis=project_axis(
    title=(
      "Share of foreign-born labor force"
    ),
    ticksuffix="%",
    range_=[
      0,
      recent_foreign_born_share[
        "recent_share"
      ].max()
      * 1.10,
    ],
  ),
  showlegend=False,
  hovermode="x unified",
)

show_figure(
  fig_recent_foreign_born_share,
  filename=(
    "foreign_born_labor_force_recent_10yr_share"
  ),
)

del recent_share_parts
gc.collect()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% 7. LABOR-FORCE PARTICIPATION RATE BY AGE AND NATIVITY
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

age_files = get_cps_files()
latest_age_files = age_files[
  -12:
]
age_parts = []

for file in latest_age_files:
  required_columns = [
    "HRYEAR4",
    "HRMONTH",
    "PRTAGE",
    "FOREIGN_BORN",
    "IN_CIVILIAN_POPULATION",
    "IN_LABOR_FORCE",
    "PERSON_WEIGHT",
  ]

  if not file_has_columns(
    file,
    required_columns,
  ):
    continue

  data = pd.read_csv(
    file,
    usecols=required_columns,
  )

  data["Date"] = cps_month_date(
    data
  )

  data["Nativity"] = np.select(
    [
      data[
        "FOREIGN_BORN"
      ] == 0,
      data[
        "FOREIGN_BORN"
      ] == 1,
    ],
    [
      "Native born",
      "Foreign born",
    ],
    default=None,
  )

  data = data[
    data[
      "Nativity"
    ].notna()
    & (
      data[
        "IN_CIVILIAN_POPULATION"
      ] == 1
    )
    & (
      data[
        "PRTAGE"
      ] >= 16
    )
    & (
      data[
        "PRTAGE"
      ] <= 85
    )
    & np.isfinite(
      data[
        "PERSON_WEIGHT"
      ]
    )
    & (
      data[
        "PERSON_WEIGHT"
      ] > 0
    )
  ]

  if len(data) > 0:
    rows = []

    for (
      month_date,
      age,
      nativity,
    ), group in data.groupby(
      [
        "Date",
        "PRTAGE",
        "Nativity",
      ],
      sort=False,
    ):
      rows.append(
        {
          "Date": month_date,
          "PRTAGE": age,
          "Nativity": nativity,
          "civilian_population": (
            group[
              "PERSON_WEIGHT"
            ].sum()
          ),
          "labor_force": (
            group.loc[
              group[
                "IN_LABOR_FORCE"
              ] == 1,
              "PERSON_WEIGHT",
            ].sum()
          ),
        }
      )

    age_parts.append(
      pd.DataFrame(
        rows
      )
    )

  del data


if len(
  age_parts
) == 0:
  raise RuntimeError(
    "No usable observations were available "
    "for LFPR by age."
  )

lfpr_by_age_monthly = pd.concat(
  age_parts,
  ignore_index=True,
)

latest_age_start = (
  lfpr_by_age_monthly[
    "Date"
  ].min()
)

latest_age_end = (
  lfpr_by_age_monthly[
    "Date"
  ].max()
)

lfpr_by_age_latest = (
  lfpr_by_age_monthly.groupby(
    [
      "PRTAGE",
      "Nativity",
    ],
    as_index=False,
  )
  .agg(
    civilian_population=(
      "civilian_population",
      "mean",
    ),
    labor_force=(
      "labor_force",
      "mean",
    ),
  )
  .rename(
    columns={
      "PRTAGE": "age"
    }
  )
)

lfpr_by_age_latest[
  "lfpr"
] = (
  100
  * lfpr_by_age_latest[
    "labor_force"
  ]
  / lfpr_by_age_latest[
    "civilian_population"
  ]
)

lfpr_by_age_latest = (
  lfpr_by_age_latest[
    np.isfinite(
      lfpr_by_age_latest[
        "lfpr"
      ]
    )
    & (
      lfpr_by_age_latest[
        "civilian_population"
      ] > 0
    )
  ]
)

lfpr_by_age_latest.to_csv(
  FILTERED_DIR
  / "lfpr_by_age_latest_12_months.csv",
  index=False,
)

lfpr_by_age_latest[
  "Nativity"
] = pd.Categorical(
  lfpr_by_age_latest[
    "Nativity"
  ],
  categories=[
    "Native born",
    "Foreign born",
  ],
  ordered=True,
)

bubble_size_reference = (
  2
  * lfpr_by_age_latest[
    "civilian_population"
  ].max()
  / (
    42**2
  )
)

fig_lfpr_age = (
  go.Figure()
)

for nativity_group in [
  "Native born",
  "Foreign born",
]:
  age_points = (
    lfpr_by_age_latest[
      lfpr_by_age_latest[
        "Nativity"
      ]
      == nativity_group
    ]
  )

  hover_text = [
    (
      f"{nativity_group}<br>"
      f"Age: {int(age)}<br>"
      f"Labor-force participation: {lfpr:.1f}%<br>"
      f"Average monthly population: {population:,.0f}"
    )
    for age, lfpr, population
    in zip(
      age_points[
        "age"
      ],
      age_points[
        "lfpr"
      ],
      age_points[
        "civilian_population"
      ],
    )
  ]

  fig_lfpr_age.add_trace(
    go.Scatter(
      x=age_points[
        "age"
      ],
      y=age_points[
        "lfpr"
      ],
      mode="markers",
      name=nativity_group,
      legendgroup=nativity_group,
      marker={
        "color": nativity_colors[
          nativity_group
        ],
        "size": age_points[
          "civilian_population"
        ],
        "sizemode": "area",
        "sizeref": (
          bubble_size_reference
        ),
        "sizemin": 4,
        "opacity": 0.85,
        "line": {
          "color": "white",
          "width": 0.7,
        },
      },
      text=hover_text,
      hoverinfo="text",
    )
  )

project_layout(
  fig=fig_lfpr_age,
  title=(
    "Labor-Force Participation Rate by Age and Nativity"
  ),
  subtitle=(
    "Population-weighted estimates using 12 available months, "
    f"{latest_age_start.strftime('%b %Y')}–"
    f"{latest_age_end.strftime('%b %Y')}"
  ),
  xaxis=project_axis(
    title="Age",
    range_=[
      15,
      86,
    ],
  ),
  yaxis=project_axis(
    title=(
      "Labor-force participation rate"
    ),
    ticksuffix="%",
    range_=[
      0,
      100,
    ],
  ),
  legend_title="",
  hovermode="closest",
)

show_figure(
  fig_lfpr_age,
  filename=(
    "labor_force_participation_by_age"
  ),
)

del age_parts
del lfpr_by_age_monthly
gc.collect()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% 8. BUILD SINGLE HTML REPORT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

REPORT_FILE = (
  REPORT_DIR
  / "cps_labor_market_report.html"
)

# Edit the HTML text below whenever you want to add narrative between graphs.
report_figures = [
  (
    "Regional Labor Market Tightness",
    """
    <p>
      Labor-market tightness varies substantially across U.S. regions.
      This figure compares the number of unemployed workers with the
      number of available job openings over time.
    </p>
    """,
    fig_regional_openings,
  ),
  (
    "State Job Opening Rates",
    """
    <p>
      The animated map shows how state job-opening rates have evolved
      month by month.
    </p>
    """,
    fig_state_map,
  ),
  (
    "Labor-Force Participation by Nativity",
    """
    <p>
      Native-born and foreign-born populations differ in their rates
      of labor-force participation. The figure below tracks those
      differences using monthly CPS estimates. However, as the population
      pyramid and participation rate by age graph show, this doesn't
      capture the whole story.
    </p>
    """,
    fig_lfpr_monthly,
  ),
  (
    "Population Pyramid by Nativity",
    """
    <p>
      Labor-force outcomes partly reflect differences in the age
      structure of the native-born and foreign-born populations.
      This population pyramid shows those demographic differences directly.
    </p>
    """,
    fig_pyramid,
  ),
  (
    "Population Pyramid by Immigrant Generation",
    """
    <p>
      The native-born population is split into two groups here:
      people with two native-born parents and second-generation
      Americans with at least one foreign-born parent. The foreign-born
      population is also shown separately.
    </p>
    <p>
      Male populations extend to the left and female populations to the
      right. The bars are overlaid so that the three age profiles can be
      compared directly.
    </p>
    """,
    fig_pyramid_gen_overlay,
  ),
  (
    "Labor-Force Participation Seasonality",
    """
    <p>
      Participation varies across the calendar year. These bars show
      average monthly deviations from each nativity group's annual
      participation rate.
    </p>
    """,
    fig_lfpr_seasonality,
  ),
  (
    "Recent Immigrant Share of the Foreign-Born Labor Force",
    """
    <p>
      The composition of the foreign-born labor force changes as new
      immigrants arrive. This figure tracks the share estimated to have
      entered the United States within the previous ten years. This figure
      needs double-checking, as it's coded with entry spans and take the 
      median year. I also haven't uncovered the seasonality reasons here.
    </p>
    """,
    fig_recent_foreign_born_share,
  ),
  (
    "Labor-Force Participation by Age and Nativity",
    """
    <p>
      The relationship between nativity and labor-force participation
      varies considerably by age. Bubble area represents the average
      monthly civilian population associated with each age-specific estimate.
      It is likely that spouses are often unable to work legally, leading
      to the lower participation rate for immigrants < 45 years of age. 
      However, above this age, immigrants generally surpass the native-born
      population for labor force participation rates. One issue may likely 
      be needing to attain more wealth for retirement, in part due to a lack
      of social security benefits.
    </p>
    """,
    fig_lfpr_age,
  ),
]

html_parts = []

for figure_number, (
  section_title,
  section_text,
  fig,
) in enumerate(
  report_figures
):
  include_plotlyjs = (
    "inline"
    if figure_number == 0
    else False
  )

  figure_html = pio.to_html(
    fig,
    full_html=False,
    include_plotlyjs=(
      include_plotlyjs
    ),
    config=project_config(
      f"figure_{figure_number + 1}"
    ),
    default_width="100%",
    default_height="700px",
  )

  html_parts.append(
    f"""
    <section
      class="figure-section"
      id="figure-{figure_number + 1}"
    >
      <div class="section-intro">
        <h2>{section_title}</h2>
        {section_text.strip()}
      </div>

      <div class="figure-container">
        {figure_html}
      </div>
    </section>
    """
  )

report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">

  <meta
    name="viewport"
    content="width=device-width, initial-scale=1"
  >

  <title>
    U.S. Labor Force and Immigration
  </title>

  <style>
    * {{
      box-sizing: border-box;
    }}

    html {{
      scroll-behavior: smooth;
    }}

    body {{
      margin: 0;
      background: #FFFFFF;
      color: #222222;
      font-family: Montserrat, Arial, sans-serif;
    }}

    .report-header {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 48px 40px 20px 40px;
    }}

    .report-header h1 {{
      margin: 0 0 10px 0;
      font-size: 22px;
      font-weight: 600;
    }}

    .report-header p {{
      margin: 0;
      max-width: 850px;
      font-size: 14px;
      line-height: 1.6;
      color: #666666;
    }}

    .report-nav {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 10px 40px 30px 40px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px 18px;
    }}

    .report-nav a {{
      color: #0F374B;
      text-decoration: none;
      font-size: 12px;
    }}

    .report-nav a:hover {{
      text-decoration: underline;
    }}

    .figure-section {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px 30px 58px 30px;
      border-top: 1px solid #E6E6E6;
    }}

    .section-intro {{
      max-width: 900px;
      margin: 0 0 20px 10px;
    }}

    .section-intro h2 {{
      margin: 0 0 10px 0;
      font-size: 18px;
      font-weight: 600;
      line-height: 1.25;
      color: #222222;
    }}

    .section-intro p {{
      margin: 0 0 12px 0;
      font-size: 14px;
      line-height: 1.65;
      color: #444444;
    }}

    .figure-container {{
      width: 100%;
    }}

    .plotly-graph-div {{
      width: 100% !important;
    }}

    @media (max-width:700px) {{
      .report-header {{
        padding: 30px 18px 15px 18px;
      }}

      .report-nav {{
        padding: 10px 18px 25px 18px;
      }}

      .figure-section {{
        padding: 24px 5px 45px 5px;
      }}

      .section-intro {{
        margin-left: 13px;
        margin-right: 13px;
      }}
    }}
  </style>
</head>

<body>

  <header class="report-header">
    <h1>
      U.S. Labor Force and Immigration
    </h1>

    <p>
      Interactive labor-market and demographic indicators constructed
      from Census CPS Basic Monthly Microdata and Bureau of Labor
      Statistics data.
    </p>
  </header>

  <nav class="report-nav">
    <a href="#figure-1">Regional labor markets</a>
    <a href="#figure-2">State openings</a>
    <a href="#figure-3">LFPR by nativity</a>
    <a href="#figure-4">Population pyramid</a>
    <a href="#figure-5">Immigrant generations</a>
    <a href="#figure-6">Seasonality</a>
    <a href="#figure-7">Recent immigrants</a>
    <a href="#figure-8">LFPR by age</a>
  </nav>

  {''.join(html_parts)}

</body>
</html>
"""

REPORT_FILE.write_text(
  report_html,
  encoding="utf-8",
)

print(
  "Combined HTML report written to: "
  f"{REPORT_FILE.resolve()}",
  flush=True,
)

try:
  subprocess.Popen(
    [
      "xdg-open",
      str(
        REPORT_FILE.resolve()
      ),
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
  )

except FileNotFoundError:
  print(
    "Could not find xdg-open. "
    "Open this file manually: "
    f"{REPORT_FILE.resolve()}"
  )


