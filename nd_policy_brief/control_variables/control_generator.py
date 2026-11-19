"""Generate independent, merge-ready state controls for 2013, 2018, 2023.

Data sources (all independent of the regression dataset):
* Population and ages 0-4: two Census PEP vintages from sources/census_pep/.
* Female LFPR: BLS Geographic Profile Table 14 (Women) XLSX files.
* Median household income: ACS 1-year B19013 via the Census API.
* Binding state/federal minimum wage and CPI-U: FRED API.

Produces ONLY control_variables_state_year_2013_2018_2023.csv
(51 states/DC * 3 years = 153 rows). Does not merge regulations or costs,
fit models, generate figures, or require a previous controls CSV.

Run as a file: python control_variables/control_generator.py
IPython: %run control_variables/control_generator.py
When pasting individual cells, use the control_variables/ working directory
(or project root with a control_variables/ subdirectory).

Dependencies: pandas numpy requests openpyxl; CENSUS_API_KEY and FRED_API_KEY
from environment or a local .env file outside version control. For example,
~/openscience/.env may hold both keys; the script uses Path.home() to use it.

Reuses existing source CSV/XLSX files and caches API responses under .cache/.
"""
from __future__ import annotations

from pathlib import Path
import os
import re
import time
import json

import numpy as np
import pandas as pd
import requests
from openpyxl import load_workbook


# %% PATHS / SETTINGS --------------------------------------------------------
def _control_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    cwd = Path.cwd()
    return cwd / "control_variables" if (cwd / "control_variables").is_dir() else cwd


CONTROL_DIR = _control_dir()
SOURCE_DIR = CONTROL_DIR / "sources"
BLS_DIR = SOURCE_DIR / "bls_female_lfpr"
PEP_DIR = SOURCE_DIR / "census_pep"
CACHE_DIR = CONTROL_DIR / ".cache" / "control_generator"
OUTPUT_FILE = CONTROL_DIR / "control_variables_state_year_2013_2018_2023.csv"

STUDY_YEARS = (2013, 2018, 2023)
YEARS = STUDY_YEARS
REAL_DOLLAR_YEAR = 2025
REFRESH_CACHE = False

STATE_FIPS = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia",
    "15": "Hawaii", "16": "Idaho", "17": "Illinois", "18": "Indiana",
    "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana",
    "23": "Maine", "24": "Maryland", "25": "Massachusetts",
    "26": "Michigan", "27": "Minnesota", "28": "Mississippi",
    "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada",
    "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
    "36": "New York", "37": "North Carolina", "38": "North Dakota",
    "39": "Ohio", "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania",
    "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota",
    "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont",
    "51": "Virginia", "53": "Washington", "54": "West Virginia",
    "55": "Wisconsin", "56": "Wyoming",
}
STATE_NAME_TO_FIPS = {v: k for k, v in STATE_FIPS.items()}
STATE_UPPER = {v.upper(): v for v in STATE_FIPS.values()}
KEYS = ["state", "year"]
FULL_KEYS = ["state", "state_fips", "year"]
FINAL_COLUMNS = [
    "state", "state_fips", "year", "pop_total", "pop_under5",
    "percent_0_4", "female_lfpr", "median_household_income",
    "r_household_income", "r_min_wage",
]

PEP_URLS = {
    "old": (
        "https://www2.census.gov/programs-surveys/popest/datasets/"
        "2010-2020/intercensal/state/asrh/sc-est2020int-alldata6.csv"
    ),
    "new": (
        "https://www2.census.gov/programs-surveys/popest/datasets/"
        "2020-2025/state/asrh/sc-est2025-alldata6.csv"
    ),
}


# %% COMMON TOOLS -----------------------------------------------------------
def load_env() -> None:
    """Read local API keys without hardcoded personal paths or dotenv dependency.

    Check the project first, then ~/openscience/.env as a user-local fallback.
    External reviewers can use their own environment or project .env file.
    """
    candidates = [
        CONTROL_DIR.parent / ".env",
        CONTROL_DIR / ".env",
        Path.cwd() / ".env",
        Path.home() / "openscience" / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def request(url: str, *, params: dict | None = None) -> requests.Response:
    failure = None
    for attempt in range(5):
        try:
            response = requests.get(
                url, params=params, timeout=(20, 150),
                headers={"User-Agent": "ND childcare research data generator"},
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            failure = exc
            if attempt < 4:
                time.sleep(min(2 ** (attempt + 1), 16))
    raise RuntimeError(f"HTTP request failed: {url}: {failure}")


def json_cached(url: str, params: dict, filename: str) -> dict | list:
    dest = CACHE_DIR / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not REFRESH_CACHE:
        return json.loads(dest.read_text(encoding="utf-8"))
    print(f"Requesting {filename}")
    response = request(url, params=params)
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        content_type = response.headers.get("Content-Type", "unknown")
        raise RuntimeError(
            f"Expected JSON from {url}, but received {content_type} "
            f"(HTTP {response.status_code}). "
            "Check that the API key is present and valid, and that the "
            "service is responding normally. The response was not cached."
        ) from exc
    # Do not cache API errors masquerading as successful responses.
    if isinstance(data, dict) and "error_code" in data:
        raise RuntimeError(f"API error: {data}")
    dest.write_text(json.dumps(data), encoding="utf-8")
    return data


def state_fips(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str).str.zfill(2)


def check_panel(frame: pd.DataFrame, label: str, years=YEARS) -> pd.DataFrame:
    expected = len(STATE_FIPS) * len(tuple(years))
    if len(frame) != expected or frame.duplicated(KEYS).any():
        raise ValueError(
            f"{label}: expected {expected} unique state-years, got {len(frame)}"
        )
    if frame["state"].isna().any() or set(frame.state) != set(STATE_FIPS.values()):
        raise ValueError(f"{label}: state/DC coverage mismatch")
    if set(frame.year) != set(years):
        raise ValueError(f"{label}: year coverage mismatch")
    return frame.sort_values(KEYS).reset_index(drop=True)


def numeric(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (float, int, np.number)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, str):
        s = value.replace(",", "").replace("%", "").strip()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
            return float(s)
    return None


def tidy_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()



# Preserve the original PEP files as public reproducibility inputs.
def pep_source(vintage: str, filename: str) -> Path:
    path = PEP_DIR / filename
    if path.is_file() and path.stat().st_size > 0 and not REFRESH_CACHE:
        print(f"  Census PEP (local): {path.name}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading Census PEP: {filename}")
    data = request(PEP_URLS[vintage]).content
    if not data:
        raise ValueError(f"Empty PEP download: {PEP_URLS[vintage]}")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)
    return path


# %% CENSUS PEP -------------------------------------------------------------
def pep_from_file(path: Path, years: tuple[int, ...], vintage: str) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip().str.upper()
    for col in ("STATE", "SEX", "AGE", "ORIGIN", "RACE"):
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    required = {"STATE", "SEX", "AGE"}
    if not required.issubset(data.columns):
        raise ValueError(f"{path}: missing PEP columns {required - set(data.columns)}")
    # Mirrors R: total sex, aggregate ethnicity/race and ages 0 through 85,
    # where age 85 is the open-ended 85+ category in the source.
    mask = data.SEX.eq(0) & data.AGE.between(0, 85)
    if "ORIGIN" in data:
        mask &= data.ORIGIN.eq(0)
    if "RACE" in data:
        mask &= data.RACE.eq(0) if data.RACE.eq(0).any() else data.RACE.isin(range(1, 7))
    data = data.loc[mask].copy()
    data["state_fips"] = state_fips(data.STATE)
    data = data[data.state_fips.isin(STATE_FIPS)]
    pieces = []
    for year in years:
        col = f"POPESTIMATE{year}"
        if col not in data.columns:
            raise ValueError(f"{path.name}: missing {col}")
        working = data[["state_fips", "AGE"]].copy()
        working["population"] = pd.to_numeric(data[col], errors="coerce")
        if working.population.isna().any():
            raise ValueError(f"{path.name}: missing population for {year}")
        pop = working.groupby("state_fips").population.sum()
        under5 = working[working.AGE.lt(5)].groupby("state_fips").population.sum()
        part = pd.DataFrame({"pop_total": pop, "pop_under5": under5}).reset_index()
        part["state"] = part.state_fips.map(STATE_FIPS)
        part["year"] = year
        part["percent_0_4"] = part.pop_under5 / part.pop_total
        part["population_source"] = vintage
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def build_population() -> pd.DataFrame:
    print("Building Census PEP population")
    old = pep_from_file(
        pep_source("old", "sc-est2020int-alldata6.csv"),
        (2013, 2018), "Census PEP 2010-2020 intercensal",
    )
    new = pep_from_file(
        pep_source("new", "sc-est2025-alldata6.csv"),
        (2023,), "Census PEP 2020-2025",
    )
    out = check_panel(pd.concat([old, new], ignore_index=True), "PEP")
    if out[["pop_total", "pop_under5", "percent_0_4"]].isna().any().any():
        raise ValueError("PEP missing observations")
    if (out.pop_total <= out.pop_under5).any() or not out.percent_0_4.between(0.01, 0.15).all():
        raise ValueError("PEP invalid totals or under-five population share")
    return out


# %% BLS PUBLISHED FEMALE LFPR ----------------------------------------------
def extract_bls_table(path: Path, year: int) -> pd.DataFrame:
    """Read published female LFPR from BLS Table 14 (one row per state/group).

    Observed workbook layout (zero-based Excel positions):
      0 state FIPS, 1 group code, 2 state name, 3 group,
      4 civilian noninstitutional population (thousands),
      5 civilian labor force (thousands),
      6 civilian labor force / percent of population.
    Select the top-level group 'Women' (group code 03), not a race subgroup.
    """
    wb = load_workbook(path, read_only=True, data_only=True)
    found: dict[str, float] = {}
    try:
        for sheet in wb.worksheets:
            # Validate that the sheet has the expected Table 14 columns.
            header_found = False
            for header in sheet.iter_rows(min_row=1, max_row=min(12, sheet.max_row), values_only=True):
                fields = [tidy_text(x) for x in header]
                if (len(fields) >= 7 and fields[2] == "STATE"
                        and fields[3] in {"GROUP", "POPULATION GROUP"}
                        and "CIVILIAN LABOR FORCE" in fields[5]):
                    header_found = True
                    break
            if not header_found:
                continue  # Ignore cover sheets or unrelated worksheets.

            for excel_row, cells in enumerate(sheet.iter_rows(values_only=True), start=1):
                if len(cells) < 7 or tidy_text(cells[3]) != "WOMEN":
                    continue
                state = STATE_UPPER.get(tidy_text(cells[2]))
                if state is None:
                    continue

                fips = numeric(cells[0])
                group_code = numeric(cells[1])
                population = numeric(cells[4])
                labor_force = numeric(cells[5])
                rate = numeric(cells[6])
                if fips is None or int(fips) != int(STATE_NAME_TO_FIPS[state]):
                    raise ValueError(
                        f"{path.name}/{sheet.title}/row {excel_row}: "
                        f"unexpected FIPS for {state}: {cells[0]!r}"
                    )
                if group_code is None or int(group_code) != 3:
                    raise ValueError(
                        f"{path.name}/{sheet.title}/row {excel_row}: "
                        f"unexpected Women group code: {cells[1]!r}"
                    )
                if (population is None or labor_force is None or rate is None
                        or population <= 0 or not 20 < rate < 90):
                    raise ValueError(
                        f"{path.name}/{sheet.title}/row {excel_row}: "
                        f"invalid female LFPR inputs for {state}"
                    )
                # Published counts are rounded to thousands; permit rounding
                # differences when checking that column G is an LFPR, not an
                # employment or unemployment percentage.
                implied_rate = 100 * labor_force / population
                if abs(implied_rate - rate) > 1.5:
                    raise ValueError(
                        f"{path.name}/{sheet.title}/row {excel_row}: "
                        f"female LFPR {rate} is inconsistent with counts "
                        f"({labor_force}/{population} -> {implied_rate:.2f})"
                    )
                if state in found:
                    if abs(found[state] - rate) > 1e-9:
                        raise ValueError(f"{path.name}: conflicting rates for {state}")
                    raise ValueError(f"{path.name}: duplicate Women row for {state}")
                found[state] = rate
    finally:
        wb.close()

    missing = sorted(set(STATE_FIPS.values()) - set(found))
    if missing:
        raise ValueError(
            f"{path.name}: extracted {len(found)}/51 top-level Women rows; "
            f"missing: {missing[:15]}. Check whether this year's workbook "
            "uses different columns from the 2010 layout."
        )
    return pd.DataFrame([
        {
            "state": state, "year": year, "female_lfpr": rate / 100,
            "female_lfpr_source": "BLS Geographic Profile Table 14, Women LFPR",
            "female_lfpr_source_url": f"https://www.bls.gov/opub/geographic-profile/{year}/",
            "female_lfpr_workbook": path.name,
        }
        for state, rate in found.items()
    ])

def build_bls_lfpr() -> pd.DataFrame:
    print("Reading original BLS Table 14 annual workbooks")
    parts = []
    for year in YEARS:
        filename = (
            "ptable14full2025.xlsx" if year == 2025
            else f"table14full{year % 100:02d}.xlsx"
        )
        path = BLS_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing BLS Table 14 workbook: {path}. "
                "Rebuilding published LFPR requires these original source workbooks."
            )
        parts.append(extract_bls_table(path, year))
        print(f"  {year}: {len(parts[-1])} states/DC")
    out = check_panel(pd.concat(parts, ignore_index=True), "BLS female LFPR")
    if not out.female_lfpr.between(0.2, 0.9).all():
        raise ValueError("BLS female LFPR outside reasonable bounds")
    return out



# %% ACS HOUSEHOLD INCOME
def build_acs_income() -> pd.DataFrame:
    print("Building ACS 1-year state median household income")
    census_key = os.getenv("CENSUS_API_KEY", "").strip()
    if not census_key:
        raise RuntimeError(
            "CENSUS_API_KEY is required for ACS downloads. "
            "The Census Bureau requires API keys for all data requests. "
            "Add CENSUS_API_KEY=... to the project's untracked .env file, "
            "or export it in your shell before running this script."
        )
    pieces = []
    for year in STUDY_YEARS:
        url = f"https://api.census.gov/data/{year}/acs/acs1"
        params = {"get": "NAME,B19013_001E", "for": "state:*", "key": census_key}
        raw = json_cached(url, params, f"acs/acs_income_{year}.json")
        if not isinstance(raw, list) or len(raw) < 52:
            raise ValueError(f"{year} Census ACS returned too few observations")
        df = pd.DataFrame(raw[1:], columns=raw[0])
        df["state_fips"] = state_fips(df["state"])
        df = df[df.state_fips.isin(STATE_FIPS)]
        out = pd.DataFrame({
            "state": df.state_fips.map(STATE_FIPS),
            "state_fips": df.state_fips,
            "year": year,
            "median_household_income": pd.to_numeric(df.B19013_001E, errors="coerce"),
        })
        out["income_source"] = "ACS 1-year B19013"
        out["income_source_url"] = url
        pieces.append(out)
    result = check_panel(pd.concat(pieces, ignore_index=True), "ACS")
    if result.median_household_income.isna().any() or (result.median_household_income <= 0).any():
        raise ValueError("ACS income missing or nonpositive")
    return result


# %% FRED CPI AND MINIMUM WAGES ---------------------------------------------
def fred_call(endpoint: str, params: dict, cache: str) -> dict:
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        raise RuntimeError("FRED_API_KEY must be set to rebuild FRED source values")
    return json_cached(
        f"https://api.stlouisfed.org/fred/{endpoint}",
        {**params, "api_key": key, "file_type": "json"},
        f"fred/{cache}",
    )


def fred_observations(series_id: str, start: int, end: int) -> pd.DataFrame:
    raw = fred_call(
        "series/observations",
        {
            "series_id": series_id,
            "observation_start": f"{start}-01-01",
            "observation_end": f"{end}-12-31",
        },
        f"{series_id}_{start}_{end}.json",
    )
    if not raw.get("observations"):
        raise ValueError(f"No FRED observations for {series_id}")
    df = pd.DataFrame(raw["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df.date.dt.year
    df["value"] = pd.to_numeric(df.value, errors="coerce")
    return df[["date", "year", "value"]]


def fred_release_series(release_id: int) -> pd.DataFrame:
    rows, offset, limit = [], 0, 1000
    while True:
        raw = fred_call(
            "release/series", {"release_id": release_id, "limit": limit, "offset": offset},
            f"release_{release_id}_series_{offset}.json",
        )
        rows.extend(raw.get("seriess", []))
        offset += limit
        if offset >= int(raw.get("count", 0)):
            break
    if not rows:
        raise ValueError(f"FRED release {release_id} returned no series")
    return pd.DataFrame(rows)


def build_cpi() -> pd.DataFrame:
    """Use published CPI-U observations to construct annual 2025-dollar factors.

    To reproduce the original analysis, use 11 published months of 2025,
    excluding October, and all 12 months in each requested study year.
    """
    print("Building annual CPI-U to real 2025 dollars")
    df = fred_observations("CPIAUCNS", min(YEARS), REAL_DOLLAR_YEAR)
    df["month"] = df.date.dt.month
    df = df.dropna(subset=["value"]).copy()
    # Retain the original paper's 2025-dollar convention: October 2025
    # is excluded from the base-year CPI average, even if later revised.
    df = df.loc[~(df.year.eq(2025) & df.month.eq(10))].copy()

    required = set(YEARS) | {REAL_DOLLAR_YEAR}
    for year in sorted(required):
        months = df.loc[df.year.eq(year), "month"].astype(int)
        expected = set(range(1, 13)) - ({10} if year == 2025 else set())
        if set(months) != expected or len(months) != len(expected):
            raise ValueError(
                f"CPI {year}: expected exactly the published months "
                f"{sorted(expected)}, found {sorted(months.tolist())}"
            )

    annual = (
        df.groupby("year", as_index=False)
        .agg(cpi=("value", "mean"), cpi_months=("month", "nunique"))
    )
    base = float(annual.loc[annual.year.eq(REAL_DOLLAR_YEAR), "cpi"].iloc[0])
    annual["cpi_base"] = base
    annual["real_dollar_year"] = REAL_DOLLAR_YEAR
    annual["to_real_dollars"] = base / annual.cpi
    print(f"  2025 published-month CPI-U average: {base:.6f} (11 months)")
    return annual


def build_minimum_wages() -> pd.DataFrame:
    print("Building binding state/federal annual minimum wages")
    series = fred_release_series(387)
    mask = (
        series.title.astype(str).str.startswith("State Minimum Wage Rate for ")
        & series.frequency.astype(str).eq("Annual")
        & series.units.astype(str).eq("Dollars per Hour")
    )
    series = series[mask].copy()
    series["state"] = series.title.str.removeprefix("State Minimum Wage Rate for ").str.strip()
    series = series[series.state.isin(STATE_NAME_TO_FIPS)]
    if series.state.duplicated().any():
        raise ValueError("FRED release 387 has duplicate state wage series")
    if series.empty:
        raise ValueError("No annual state minimum wage series in FRED release 387")
    parts = []
    for r in series.itertuples(index=False):
        data = fred_observations(str(r.id), min(YEARS), max(YEARS))
        data = data[data.year.isin(YEARS)]
        parts.append(pd.DataFrame({
            "state": r.state, "year": data.year,
            "state_min_wage": data.value, "state_min_wage_series": r.id,
        }))
    state_wages = pd.concat(parts, ignore_index=True)
    if state_wages.duplicated(KEYS).any():
        raise ValueError("Duplicate FRED state minimum-wage observations")
    grid = pd.MultiIndex.from_product(
        [list(STATE_NAME_TO_FIPS), list(YEARS)], names=KEYS
    ).to_frame(index=False)
    out = grid.merge(state_wages, on=KEYS, how="left", validate="one_to_one")
    federal = fred_observations("STTMINWGFG", min(YEARS), max(YEARS))
    federal = federal[federal.year.isin(YEARS)].rename(columns={"value": "federal_min_wage"})
    if federal.year.duplicated().any():
        raise ValueError("Duplicate FRED federal minimum-wage years")
    out = out.merge(federal[["year", "federal_min_wage"]], on="year", validate="many_to_one")
    out["binding_min_wage"] = out[["state_min_wage", "federal_min_wage"]].max(axis=1)
    if out[["federal_min_wage", "binding_min_wage"]].isna().any().any():
        raise ValueError("Missing FRED federal/binding wage")
    out["state_fips"] = out.state.map(STATE_NAME_TO_FIPS)
    out["min_wage_source"] = "FRED release 387 / DOL; binding=max(state,federal)"
    return check_panel(out, "FRED wages")


# %% INDEPENDENT BUILD AND CHECK
def build_controls() -> tuple[pd.DataFrame, pd.DataFrame]:
    population = build_population()
    bls = build_bls_lfpr()
    income = build_acs_income()
    cpi = build_cpi()
    wages = build_minimum_wages()
    audit = (
        population.merge(bls, on=KEYS, how="inner", validate="one_to_one")
        .merge(income, on=FULL_KEYS, how="inner", validate="one_to_one")
        .merge(wages, on=FULL_KEYS, how="inner", validate="one_to_one")
        .merge(
            cpi[["year", "cpi", "cpi_base", "to_real_dollars", "real_dollar_year"]],
            on="year", how="inner", validate="many_to_one",
        )
    )
    audit["r_household_income"] = audit.median_household_income * audit.to_real_dollars
    audit["r_min_wage"] = audit.binding_min_wage * audit.to_real_dollars
    audit = check_panel(audit, "Final rebuilt controls")
    controls = audit[FINAL_COLUMNS].copy()
    if controls[FINAL_COLUMNS[3:]].isna().any().any():
        raise ValueError("Missing final control values")
    if not np.allclose(controls.percent_0_4, controls.pop_under5 / controls.pop_total, atol=1e-12):
        raise ValueError("Under-five proportion mismatch")
    if not (controls[["r_household_income", "r_min_wage"]] > 0).all().all():
        raise ValueError("Nonpositive real income or minimum wage")
    return controls, audit




# %% GENERATE CONTROL PANEL --------------------------------------------------
def main() -> None:
    load_env()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    controls, _audit = build_controls()
    selected = check_panel(controls, "Regression control panel", STUDY_YEARS)
    if len(selected) != 153:
        raise ValueError("Expected exactly 153 control rows")
    temp_path = OUTPUT_FILE.with_suffix(".csv.partial")
    selected[FINAL_COLUMNS].to_csv(temp_path, index=False)
    temp_path.replace(OUTPUT_FILE)
    print(f"\nGenerated {len(selected)} control rows: {OUTPUT_FILE}")
    print("Columns:", ", ".join(FINAL_COLUMNS))
    print("North Dakota rows:")
    print(selected.loc[selected.state.eq("North Dakota"), [
        "year", "pop_total", "percent_0_4", "female_lfpr",
        "r_household_income", "r_min_wage"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()

