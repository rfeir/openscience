"""North Dakota center childcare: two OLS models and publication-style plots.

Fit one infant-price OLS and one four-year-old-price OLS using original
Child Care Aware of America reported center prices for 2013, 2018 and 2023.
Retain CCA-reported values without dropping, cleaning or interpolating prices.
Convert original nominal prices to real 2025 dollars using the published
monthly CPI-U (CPIAUCNS); use the same 11 published months in 2025 as the
controls generator. Separate 2025 ND price benchmarks anchor simulations.

Regulatory codes and study-year controls come from their own project folders;
this script performs the joins and all subsequent statistical analysis.
The demographic and workforce sections below are separate manuscript analyses.
"""

from pathlib import Path
import os
import re
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb, TwoSlopeNorm
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from scipy.stats import t
from scipy import stats
from statsmodels.stats.outliers_influence import OLSInfluence
from matplotlib.ticker import MaxNLocator, FuncFormatter
from openpyxl import load_workbook


# %% SETTINGS ---------------------------------------------------------------
# Resolve the project from this file, not the shell's launch directory.
# In IPython cells, first: %cd .../nd_policy_brief/center_ols
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if SCRIPT_DIR.name != "center_ols":
  raise RuntimeError(
    "Run this script from center_ols/ (with %run nd_center_ols.py in IPython), "
    "or change IPython's working directory to center_ols before running cells."
  )
PROJECT_DIR = SCRIPT_DIR.parent
COST_FILE = PROJECT_DIR / "cost_left_hand_variable/center_cost_python.csv"
REG_FILE = PROJECT_DIR / "center_regulations/center_reg_vars.csv"
CONTROL_FILE = PROJECT_DIR / "control_variables/control_variables_state_year_2013_2018_2023.csv"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for input_file in (COST_FILE, REG_FILE, CONTROL_FILE):
  if not input_file.is_file():
    raise FileNotFoundError(f"Required input not found: {input_file}")

YEARS = (2013, 2018, 2023)
AGES = ("infant", "four_year_old")
TREATMENTS = ("original",)
KEYS = ["state", "year"]
ND_YEAR = 2023

ND_REFERENCE_COST = {"infant": 12_281.0, "four_year_old": 10_855.0}
ND_THREE_YEAR_SCR = 7.0
ND_THREE_YEAR_MGS = 20.0
ND_THREE_YEAR_NEW_SCR = 10.0

FORMULA = (
  "ln_cost ~ ln_scr + no_mgs + mgs_ratio + ed_t + ed_d "
  "+ ln_inc + ln_min_wage + percent_0_4 + female_lfpr "
  "+ ln_pop + C(year)"
)
REG_COLS = {
  "infant": ("scr_inf", "no_mgs_inf", "mgs_ratio_inf"),
  "four_year_old": ("scr_4", "no_mgs_4", "mgs_ratio_4"),
}
TERM_NAMES = {
  "ln_scr": "Log staff-child ratio",
  "no_mgs": "No maximum group size",
  "mgs_ratio": "Maximum group size ratio",
  "ed_t": "Teacher education requirement",
  "ed_d": "Director education requirement",
  "ln_inc": "Log household income",
  "ln_min_wage": "Log minimum wage",
  "female_lfpr": "Female labor force participation rate",
  "percent_0_4": "Share of population under age 5",
  "ln_pop": "Log population",
  "Intercept": "Constant",
  "C(year)[T.2018]": "Year: 2018",
  "C(year)[T.2023]": "Year: 2023",
}
COLORS = {
  "infant": "#51ABA0",
  "four_year_old": "#00583D",
  "three_year": "#BE5E27",
}
CI_BAND_ALPHA = {"infant": .16, "four_year_old": .13, "three_year_proxy": .13}

try:
  plt.style.use("challey_pub")
except OSError:
  pass


# %% TABLE EXPORT -----------------------------------------------------------
def export_table(table, name, caption, label, column_format=None):
  """Save a ready-to-paste HTML table, LaTeX table, and editable CSV."""
  table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
  latex = table.to_latex(
    index=False, escape=True, caption=caption,
    label=label, column_format=column_format
  )
  if name.startswith("regression_"):
    note = ("Standard errors clustered by state in parentheses. "
            "* p < 0.10, ** p < 0.05, *** p < 0.01. "
            "Year effects included; 2013 omitted.")
    latex = latex.replace(
      "\\end{table}",
      "\\par\\footnotesize " + note + "\n\\end{table}"
    )
  (OUTPUT_DIR / f"{name}.tex").write_text(latex, encoding="utf-8")
  html = f"""<!doctype html><html><head><meta charset="utf-8">
  <style>body{{font-family:Georgia,serif;margin:35px}}
  h2{{font-size:18px}}table{{border-collapse:collapse;font-size:13px}}
  th{{border-top:2px solid #222;border-bottom:1px solid #222;
  padding:9px 13px;text-align:right}}
  td{{padding:5px 13px;text-align:right}}
  td:first-child,th:first-child{{text-align:left}}
  tr:last-child td{{border-bottom:2px solid #222}}</style></head>
  <body><h2>{caption}</h2>{table.to_html(index=False, border=0)}</body></html>"""
  (OUTPUT_DIR / f"{name}.html").write_text(html, encoding="utf-8")


def stars(p):
  return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


# %% LOAD COSTS; INTERPOLATE FLAGGED AND MISSING PRICES --------------------
# %% LOAD ORIGINAL CENTER PRICES AND CONVERT TO REAL 2025 DOLLARS -------
# The source scraper generates nominal reported prices, not r_* columns.
# No imputation flags, interpolations or clean/linear price treatments here.
# Use the 11 published months of 2025 CPI-U (October was not published),
# matching the real-dollar convention used by control_generator.py.
CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS"
CPI_CACHE = PROJECT_DIR / "cost_left_hand_variable/sources/cpi_u_monthly.csv"


def load_annual_cpi():
  if CPI_CACHE.is_file():
    raw = pd.read_csv(CPI_CACHE)
    print(f"Using cached CPI-U source: {CPI_CACHE}")
  else:
    print("Downloading monthly CPI-U from FRED")
    response = requests.get(CPI_URL, timeout=(20, 120))
    response.raise_for_status()
    from io import StringIO
    raw = pd.read_csv(StringIO(response.text))
    CPI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(CPI_CACHE, index=False)
    print(f"Saved CPI-U source: {CPI_CACHE}")

  if "CPIAUCNS" not in raw.columns:
    raise ValueError(f"CPI source lacks CPIAUCNS column: {CPI_CACHE}")
  date_col = "observation_date" if "observation_date" in raw else "DATE"
  if date_col not in raw:
    raise ValueError("CPI source has no observation_date or DATE column")
  raw["date"] = pd.to_datetime(raw[date_col], errors="coerce")
  raw["cpi"] = pd.to_numeric(raw["CPIAUCNS"], errors="coerce")
  raw = raw.dropna(subset=["date", "cpi"]).copy()
  raw["year"] = raw.date.dt.year
  raw["month"] = raw.date.dt.month

  for year in (*YEARS, 2025):
    published_months = raw.loc[raw.year.eq(year), "month"].astype(int).tolist()
    expected = set(range(1, 13)) - ({10} if year == 2025 else set())
    if set(published_months) != expected or len(published_months) != len(expected):
      raise ValueError(
        f"CPI-U {year}: expected months {sorted(expected)}, "
        f"found {sorted(published_months)}. Check FRED source."
      )

  annual = raw.groupby("year")["cpi"].mean()
  return annual


costs = pd.read_csv(COST_FILE)
costs["state"] = costs["state"].astype("string").str.strip()
costs["year"] = pd.to_numeric(costs["year"], errors="raise").astype(int)
costs = costs.sort_values(KEYS).reset_index(drop=True)
if costs.duplicated(KEYS).any():
  raise ValueError("Duplicate Child Care Aware cost state-years")
if not {"infant", "four_year_old"}.issubset(costs.columns):
  raise ValueError("Expected infant and four_year_old in scraper output")

annual_cpi = load_annual_cpi()
costs = costs.loc[costs.year.isin(YEARS)].copy()
costs["cpi_u"] = costs.year.map(annual_cpi)
costs["to_2025_dollars"] = float(annual_cpi.loc[2025]) / costs.cpi_u
if costs["to_2025_dollars"].isna().any():
  raise ValueError("Missing CPI-U conversion for one or more cost years")

for age in AGES:
  nominal = pd.to_numeric(costs[age], errors="coerce")
  costs[f"r_{age}"] = nominal.where(nominal.gt(0)) * costs.to_2025_dollars
  print(f"{age}: {costs[f'r_{age}'].notna().sum()} reported prices in study years")

costs[[*KEYS, "cpi_u", "to_2025_dollars", "infant", "four_year_old",
       "r_infant", "r_four_year_old"]].to_csv(
  OUTPUT_DIR / "center_original_prices_2025_audit.csv", index=False
)


# %% MERGE REGULATIONS AND CONTROLS ----------------------------------------
reg = pd.read_csv(REG_FILE)
controls = pd.read_csv(CONTROL_FILE)

for name, frame in (("Regulations", reg), ("Controls", controls)):
  frame["state"] = frame["state"].astype("string").str.strip()
  frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
  if frame.duplicated(KEYS).any():
    raise ValueError(f"Duplicate {name} state-years")

reg = reg.loc[reg.year.isin(YEARS)].copy()
controls = controls.loc[controls.year.isin(YEARS)].copy()
if len(reg) != 153 or len(controls) != 153:
  raise ValueError("Expected 51 jurisdictions x 3 years in each input")

panel = reg.merge(controls, on=KEYS, how="left", validate="one_to_one", indicator=True)
if not panel["_merge"].eq("both").all():
  raise ValueError("Unmatched regulation/control rows")
panel = panel.drop(columns="_merge")

cost_cols = [*KEYS]
for age in AGES:
  cost_cols += [f"r_{age}"]
panel = panel.merge(costs[cost_cols], on=KEYS, how="left", validate="one_to_one")

for source, target in (
  ("r_household_income", "ln_inc"),
  ("r_min_wage", "ln_min_wage"),
  ("pop_total", "ln_pop"),
):
  value = pd.to_numeric(panel[source], errors="coerce")
  panel[target] = np.log(value.where(value > 0))

panel.to_csv(OUTPUT_DIR / "center_model_panel_2025.csv", index=False)

coverage = pd.DataFrame([
  {
    "Age": age, "Year": year,
    "Reported prices": int(panel.loc[panel.year.eq(year), f"r_{age}"].notna().sum()),
  }
  for age in AGES for year in YEARS
])
export_table(
  coverage, "sample_coverage", "Available original childcare prices by year",
  "tab:childcare_coverage", "llr"
)
print("\nORIGINAL PRICE COVERAGE BY YEAR\n", coverage.to_string(index=False))


# %% FIT TWO MODELS: ORIGINAL INFANT AND FOUR-YEAR-OLD PRICES ----------
# Preserve the (age, "original") keys consumed by simulation/figure code.
models = {}
model_states = {}
model_info = []
coefficients = []

required = [
  "ln_cost", "ln_scr", "no_mgs", "mgs_ratio", "ed_t", "ed_d",
  "ln_inc", "ln_min_wage", "percent_0_4", "female_lfpr", "ln_pop",
  "state", "year"
]

for age in AGES:
  scr_col, no_mgs_col, ratio_col = REG_COLS[age]
  sample = panel.copy()
  scr = pd.to_numeric(sample[scr_col], errors="coerce")
  sample["ln_scr"] = np.log(scr.where(scr > 0))
  sample["no_mgs"] = pd.to_numeric(sample[no_mgs_col], errors="coerce")
  sample["mgs_ratio"] = pd.to_numeric(sample[ratio_col], errors="coerce")
  price = pd.to_numeric(sample[f"r_{age}"], errors="coerce")
  sample["ln_cost"] = np.log(price.where(price > 0))
  sample = sample.dropna(subset=required).copy()
  if len(sample) < 20 or sample.state.nunique() < 2:
    raise ValueError(f"Insufficient original-price observations for {age}")

  model = smf.ols(FORMULA, data=sample).fit(
    cov_type="cluster", cov_kwds={"groups": sample.state}, use_t=True
  )
  models[age, "original"] = model
  model_states[age, "original"] = sample.state.nunique()
  model_info.append({
    "Age": age, "Treatment": "original",
    "Observations": int(model.nobs), "States": sample.state.nunique(),
    "R-squared": round(model.rsquared, 3),
    "Adj. R-squared": round(model.rsquared_adj, 3),
  })
  ci = model.conf_int()
  for term in model.params.index:
    coefficients.append({
      "age": age, "treatment": "original", "term": term,
      "estimate": model.params[term], "se_cluster": model.bse[term],
      "p_value": model.pvalues[term],
      "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1]
    })

  print(f"\n{'=' * 70}\n{age.upper()} — ORIGINAL\n{'=' * 70}")
  print(model.summary())

pd.DataFrame(coefficients).to_csv(
  OUTPUT_DIR / "all_regression_coefficients.csv", index=False
)
pd.DataFrame(model_info).to_csv(
  OUTPUT_DIR / "model_fit_comparison.csv", index=False
)


# %% REGRESSION DIAGNOSTICS ------------------------------------------------
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostics"
DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

for age in AGES:
  for treatment in TREATMENTS:
    model = models[age, treatment]

    fitted = np.asarray(model.fittedvalues, dtype=float)
    residuals = np.asarray(model.resid, dtype=float)

    influence = OLSInfluence(model)
    standardized = np.asarray(influence.resid_studentized_internal, dtype=float)
    cooks = np.asarray(influence.cooks_distance[0], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    color = COLORS[age]

    ax = axes[0, 0]
    ax.scatter(fitted, residuals, color=color, alpha=.7, s=25)
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set(title="Residuals vs. Fitted", xlabel="Fitted log annual cost", ylabel="OLS residual")

    ax = axes[0, 1]
    stats.probplot(standardized, dist="norm", plot=ax)
    ax.get_lines()[0].set_color(color)
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[1].set_color("black")
    ax.get_lines()[1].set_linewidth(1)
    ax.set_title("Normal Q-Q Plot")
    ax.set_ylabel("Standardized residual")

    ax = axes[1, 0]
    scale = np.sqrt(np.abs(standardized))
    ax.scatter(fitted, scale, color=color, alpha=.7, s=25)
    ax.set(title="Scale-Location", xlabel="Fitted log annual cost",
           ylabel="Square root of |standardized residual|")

    ax = axes[1, 1]
    observations = np.arange(len(cooks))
    ax.vlines(observations, 0, cooks, color=color, lw=1.2)
    ax.axhline(4 / len(cooks), color="black", ls="--", lw=1,
               label="4 / N reference")
    ax.set(title="Cook's Distance", xlabel="Observation index", ylabel="Cook's distance")
    ax.legend(fontsize=8)

    for ax in axes.flat:
      ax.grid(alpha=.2)
      ax.set_axisbelow(True)

    title = "Infant" if age == "infant" else "Four-year-old"
    fig.suptitle(
      f"{title} Childcare Regression Diagnostics — {treatment.title()}\n"
      f"N = {int(model.nobs)}, "
      f"State clusters = {model_states[age, treatment]}", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, .93))

    stem = f"diagnostics_{age}_{treatment}"
    fig.savefig(DIAGNOSTIC_DIR / f"{stem}.png", dpi=300)
    fig.savefig(DIAGNOSTIC_DIR / f"{stem}.pdf")
    plt.show()
    plt.close(fig)
    print(f"Saved diagnostics: {stem}")


# %% REGRESSION CORRELATION MATRICES ---------------------------------------
CORR_DIR = OUTPUT_DIR / "correlations"
CORR_DIR.mkdir(parents=True, exist_ok=True)

CORR_VARS = [
  "ln_scr", "no_mgs", "mgs_ratio", "ed_t", "ed_d",
  "ln_inc", "ln_min_wage", "percent_0_4", "female_lfpr", "ln_pop"
]
CORR_LABELS = [
  "Log SCR", "No MGS", "MGS/SCR", "Teacher education", "Director education",
  "Log income", "Log minimum wage", "Population 0–4", "Female LFPR", "Log population"
]

for age in AGES:
  scr_col, no_mgs_col, ratio_col = REG_COLS[age]
  base = panel.copy()
  scr = pd.to_numeric(base[scr_col], errors="coerce")
  base["ln_scr"] = np.log(scr.where(scr > 0))
  base["no_mgs"] = pd.to_numeric(base[no_mgs_col], errors="coerce")
  base["mgs_ratio"] = pd.to_numeric(base[ratio_col], errors="coerce")

  for treatment, source in (("original", f"r_{age}"),):
    data = base.copy()
    price = pd.to_numeric(data[source], errors="coerce")
    data["ln_cost"] = np.log(price.where(price > 0))
    sample = data.dropna(subset=required).copy()

    corr = sample[CORR_VARS].corr(method="pearson")
    corr.index = CORR_LABELS
    corr.columns = CORR_LABELS

    stem = f"correlation_{age}_{treatment}"
    corr.round(4).to_csv(CORR_DIR / f"{stem}.csv")

    fig, ax = plt.subplots(figsize=(10, 8.5))
    image = ax.imshow(
      corr.to_numpy(), cmap="RdBu_r",
      norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
      interpolation="nearest"
    )
    ax.set_xticks(np.arange(len(CORR_LABELS)))
    ax.set_yticks(np.arange(len(CORR_LABELS)))
    ax.set_xticklabels(CORR_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(CORR_LABELS)

    for i in range(len(CORR_LABELS)):
      for j in range(len(CORR_LABELS)):
        value = corr.iloc[i, j]
        if np.isfinite(value):
          ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                  fontsize=8, color="white" if abs(value) > .65 else "black")

    ax.set_xticks(np.arange(-.5, len(CORR_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(CORR_LABELS), 1), minor=True)
    ax.grid(False)
    ax.tick_params(which="minor", length=0)

    colorbar = fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    colorbar.set_label("Pearson correlation coefficient")

    title = "Infant" if age == "infant" else "Four-year-old"
    ax.set_title(
      f"{title} Regression Correlation Matrix\n"
      f"{treatment.title()} prices — N = {len(sample)}", fontsize=13, pad=16
    )
    fig.tight_layout()
    fig.savefig(CORR_DIR / f"{stem}.png", dpi=300)
    fig.savefig(CORR_DIR / f"{stem}.pdf")
    plt.show()
    plt.close(fig)
    print(f"\n{title.upper()} — {treatment.upper()}")
    print(corr.round(3).to_string())
    print(f"Saved: {stem}")


# %% PUBLICATION REGRESSION TABLES -----------------------------------------
for age in AGES:
  rows = []
  for term, name in TERM_NAMES.items():
    if term not in models[age, "original"].params:
      continue
    estimates = {"Term": name}
    errors = {"Term": ""}
    for treatment in TREATMENTS:
      model = models[age, treatment]
      if term in model.params:
        estimates[treatment.title()] = f"{model.params[term]:.3f}{stars(model.pvalues[term])}"
        errors[treatment.title()] = f"({model.bse[term]:.3f})"
      else:
        estimates[treatment.title()] = "—"
        errors[treatment.title()] = ""
    rows += [estimates, errors]

  for name, getter in (
    ("Observations", lambda m, a, z: str(int(m.nobs))),
    ("States (clusters)", lambda m, a, z: str(model_states[a, z])),
    ("R-squared", lambda m, a, z: f"{m.rsquared:.3f}"),
    ("Adjusted R-squared", lambda m, a, z: f"{m.rsquared_adj:.3f}"),
  ):
    row = {"Term": name}
    for treatment in TREATMENTS:
      row[treatment.title()] = getter(models[age, treatment], age, treatment)
    rows.append(row)

  table = pd.DataFrame(rows)
  title = "Infant" if age == "infant" else "Four-year-old"
  export_table(
    table, f"regression_{age}",
    f"{title} center childcare prices: original-price OLS (2025 dollars)",
    f"tab:center_{age}", "lc"
  )
  print(f"\n{title.upper()} PUBLICATION TABLE\n{table.to_string(index=False)}")


# %% ND SIMULATION: CHANGE SCR, HOLD GROUP SIZE OR SPECIFY ITS CHANGE ------
def effect(model, old_scr, old_mgs, new_scr, new_mgs, clusters):
  """Difference in log cost from ln(SCR) and MGS/SCR, holding other covariates fixed."""
  delta = pd.Series(0.0, index=model.params.index)
  delta["ln_scr"] = np.log(new_scr / old_scr)
  delta["mgs_ratio"] = new_mgs / new_scr - old_mgs / old_scr
  log_change = float(delta @ model.params)
  covariance = model.cov_params().loc[delta.index, delta.index]
  se = float(np.sqrt(max(0.0, delta @ covariance @ delta)))
  critical = t.ppf(.975, df=clusters - 1)
  return (
    float(np.exp(log_change)),
    float(100 * np.expm1(log_change)),
    float(100 * np.expm1(log_change - critical * se)),
    float(100 * np.expm1(log_change + critical * se)),
  )

nd = panel.loc[panel.state.eq("North Dakota") & panel.year.eq(ND_YEAR)]
if len(nd) != 1:
  raise ValueError(f"Expected one North Dakota {ND_YEAR} regulation row")
nd = nd.iloc[0]

nd_policy = {}
for age in AGES:
  scr_col, no_mgs_col, ratio_col = REG_COLS[age]
  if nd[no_mgs_col] != 0:
    raise ValueError(f"ND {ND_YEAR} needs a numerical MGS for {age} simulation")
  nd_policy[age] = (float(nd[scr_col]), float(nd[scr_col] * nd[ratio_col]))

print("\nND 2023 policy anchors: ", nd_policy)
print("ND 2025-dollar cost benchmarks: ", ND_REFERENCE_COST)

simulation_rows = []
for treatment in TREATMENTS:
  for age, scenario, old_scr, old_mgs, new_scr, new_mgs in (
    (
      "infant", f"Infant: {nd_policy['infant'][0]:g} to 5 children",
      nd_policy["infant"][0], nd_policy["infant"][1],
      5.0, 5.0 * nd_policy["infant"][1] / nd_policy["infant"][0]
    ),
    (
      "four_year_old", "4y at 3y staffing rules",
      nd_policy["four_year_old"][0], nd_policy["four_year_old"][1],
      ND_THREE_YEAR_SCR, ND_THREE_YEAR_MGS
    ),
    (
      "four_year_old", "3y proxy: 7 to 10 children",
      ND_THREE_YEAR_SCR, ND_THREE_YEAR_MGS,
      ND_THREE_YEAR_NEW_SCR,
      ND_THREE_YEAR_NEW_SCR * ND_THREE_YEAR_MGS / ND_THREE_YEAR_SCR
    ),
    (
      "four_year_old", "3y proxy: 7 to 10 (MGS remains 20; sensitivity)",
      ND_THREE_YEAR_SCR, ND_THREE_YEAR_MGS,
      ND_THREE_YEAR_NEW_SCR, ND_THREE_YEAR_MGS
    ),
  ):
    model = models[age, treatment]
    clusters = model_states[age, treatment]
    baseline = ND_REFERENCE_COST[age]

    if scenario.startswith("3y proxy"):
      four_scr, four_mgs = nd_policy["four_year_old"]
      factor, *_ = effect(model, four_scr, four_mgs, old_scr, old_mgs, clusters)
      baseline *= factor

    multiplier, pct, pct_low, pct_high = effect(
      model, old_scr, old_mgs, new_scr, new_mgs, clusters
    )
    new_price = baseline * multiplier
    simulation_rows.append({
      "Treatment": treatment.title(), "Scenario": scenario,
      "Age model": age, "Old SCR": old_scr, "New SCR": new_scr,
      "Old MGS": old_mgs, "New MGS": new_mgs,
      "Baseline cost": baseline, "Scenario cost": new_price,
      "Change dollars": new_price - baseline,
      "Change percent": pct, "95% low": pct_low, "95% high": pct_high,
    })

simulations = pd.DataFrame(simulation_rows)
simulations.to_csv(OUTPUT_DIR / "north_dakota_simulations_full.csv", index=False)

paper_sim = pd.DataFrame({
  "Model": simulations["Treatment"],
  "Scenario": simulations["Scenario"],
  "Baseline ($)": simulations["Baseline cost"].map(lambda x: f"{x:,.0f}"),
  "Simulated ($)": simulations["Scenario cost"].map(lambda x: f"{x:,.0f}"),
  "Difference ($)": simulations["Change dollars"].map(lambda x: f"{x:+,.0f}"),
  "Change (%)": simulations["Change percent"].map(lambda x: f"{x:+.1f}"),
  "95% CI (%)": [
    f"[{lo:+.1f}, {hi:+.1f}]"
    for lo, hi in zip(simulations["95% low"], simulations["95% high"])
  ],
})
export_table(
  paper_sim, "north_dakota_simulations_table",
  "North Dakota childcare regulatory counterfactuals, 2025 price anchors / 2023 policy (2025 dollars)",
  "tab:nd_childcare_counterfactuals", "llrrrrr"
)
print("\nNORTH DAKOTA SIMULATIONS\n", paper_sim.to_string(index=False))


# %% ORIGINAL PUBLICATION GRAPH SETTINGS ------------------------------------
FIGSIZE = (6.5, 5)
DISPLAY_DPI = 100
SAVE_DPI = 300
PUBLICATION_DPI = 600
FONT_SIZE = 10
ANNOTATION_SIZE = 12
LINE_LW = 2.6
AXIS_LW = 1.0
GUIDE_LW = 2.0
POINT_SIZE = 36
LEGEND_EDGE_COLOR = "#D0DAC6"
GRID_LS = "--"
GRID_DASHES = (0, (3.2, 2.4))
LAYOUT = dict(left=.14, right=.97, bottom=.13, top=.97)
SCR_HEIGHT_RATIOS = (18, 1.2)
BREAK = dict(offset=-0.02, gap=.04, pad=.001, dx=.012, dy=.005, lw=1)
GUIDE_DOTS = (0, (.1, 2.4))
Z = dict(band=1, grid=2, guide=2.5, curve=3, axis=20, point=25, label=26)

INFANT_SCR_TICKS = np.arange(3, 7)
PRESCHOOL_SCR_TICKS = np.arange(6, 21)
FOUR_YEAR_INTEGER_SCR = np.arange(8, 21)
THREE_YEAR_INTEGER_SCR = np.arange(6, 16)

plt.rcParams.update({
  "figure.figsize": FIGSIZE,
  "figure.dpi": DISPLAY_DPI,
  "font.size": FONT_SIZE,
  "axes.labelsize": FONT_SIZE,
  "xtick.labelsize": FONT_SIZE,
  "ytick.labelsize": FONT_SIZE,
  "legend.fontsize": FONT_SIZE,
  "axes.edgecolor": "black",
  "axes.linewidth": AXIS_LW,
  "grid.color": LEGEND_EDGE_COLOR,
  "xtick.major.size": 2,
  "ytick.major.size": 2,
  "xtick.major.width": 1,
  "ytick.major.width": 1,
})


# %% SCR PLOT HELPERS -------------------------------------------------------
def scr_axes(xlim, ticks, ymin, ymax):
  fig = plt.figure(figsize=FIGSIZE)
  gs = fig.add_gridspec(2, 1, height_ratios=SCR_HEIGHT_RATIOS, hspace=0)
  ax = fig.add_subplot(gs[0])
  ax_zero = fig.add_subplot(gs[1], sharex=ax)
  ax_zero.patch.set_visible(False)
  fig.subplots_adjust(**LAYOUT)

  ax.set(xlim=xlim, ylim=(ymin, ymax))
  ax_zero.set(xlim=xlim, ylim=(0, min(700, max(100, .08 * ymin))))
  ax_zero.set_xticks(ticks)
  ax_zero.set_yticks([0], labels=["$0"])
  ax.tick_params(axis="x", bottom=False, labelbottom=False)
  ax_zero.set_xlabel("Children Per Teacher")
  fig.supylabel("Annual Price", x=.015, fontsize=FONT_SIZE)

  if ymax - ymin <= 30_000:
    y_ticks = np.arange(1000 * np.ceil(ymin / 1000), ymax + 1, 1000)
  else:
    y_ticks = MaxNLocator(nbins=6).tick_values(ymin, ymax)
    y_ticks = y_ticks[(y_ticks >= ymin) & (y_ticks <= ymax)]
  ax.set_yticks(y_ticks[1:])
  ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:,.0f}"))
  return fig, ax, ax_zero


def scr_grid_and_frame(fig, ax, ax_zero):
  for axis in (ax, ax_zero):
    axis.grid(False)
    axis.set_axisbelow(True)
    axis.tick_params(color="black", labelcolor="black")
    for spine in axis.spines.values():
      spine.set_color("black")
      spine.set_linewidth(AXIS_LW)
      spine.set_zorder(Z["axis"])

  for y in ax.get_yticks():
    if ax.get_ylim()[0] <= y <= ax.get_ylim()[1]:
      ax.axhline(y, color=LEGEND_EDGE_COLOR, lw=.8, alpha=.9,
                 ls=GRID_LS, dashes=GRID_DASHES[1], zorder=Z["grid"])

  ax.spines["bottom"].set_visible(False)
  ax_zero.spines["top"].set_visible(False)
  ax.spines["left"].set_visible(False)
  ax_zero.spines["left"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax_zero.spines["right"].set_visible(False)

  pos = ax.get_position()
  lower = ax_zero.get_position()
  x_left, x_right = pos.x0, pos.x1
  stop = BREAK["offset"] - BREAK["pad"]
  start = BREAK["offset"] + BREAK["gap"] + BREAK["pad"]
  y1 = pos.y0 + stop * pos.height
  y2 = pos.y0 + start * pos.height

  for x, a, b in (
    (x_left, lower.y0, y1),
    (x_left, y2, pos.y1),
    (x_right, lower.y0, pos.y1),
  ):
    ax.add_artist(Line2D(
      [x, x], [a, b], transform=fig.transFigure,
      color="black", lw=AXIS_LW, clip_on=False, zorder=Z["axis"]
    ))

  for y in (BREAK["offset"], BREAK["offset"] + BREAK["gap"]):
    ax.plot(
      (-BREAK["dx"], BREAK["dx"]),
      (y - BREAK["dy"], y + BREAK["dy"]),
      transform=ax.transAxes, color="black", lw=BREAK["lw"],
      clip_on=False, zorder=Z["axis"] + 1
    )

  for axis in (ax, ax_zero):
    axis.xaxis.set_zorder(Z["axis"])
    axis.yaxis.set_zorder(Z["axis"])


def scr_vertical_guide(fig, ax, ax_zero, x, y, color):
  from matplotlib.transforms import blended_transform_factory
  y_top = fig.transFigure.inverted().transform(ax.transData.transform((x, y)))[1]
  ax.add_line(Line2D(
    [x, x], [ax_zero.get_position().y0, y_top],
    transform=blended_transform_factory(ax.transData, fig.transFigure),
    color=color, lw=GUIDE_LW, ls=GUIDE_DOTS,
    dash_capstyle="round", clip_on=False, zorder=Z["guide"]
  ))


def scr_cost_label(ax, x, y, color, right_edge):
  right = np.isclose(x, right_edge)
  ax.annotate(
    f"${y:,.0f}", (x, y),
    xytext=(-4, 15) if right else (4, 8),
    textcoords="offset points",
    ha="right" if right else "left", va="bottom",
    color=color, fontsize=ANNOTATION_SIZE, fontweight=600,
    clip_on=False, zorder=Z["label"]
  )


# %% BENCHMARK-ANCHORED PRICE CURVES AND CLUSTERED CONFIDENCE BANDS --------
def publication_point(age, treatment, scr, target_ratio):
  """Counterfactual price and 95% CI using 2025 ND price as a fixed anchor."""
  old_scr, old_mgs = nd_policy[age]
  mult, pct, pct_low, pct_high = effect(
    models[age, treatment], old_scr, old_mgs,
    float(scr), float(scr) * float(target_ratio),
    model_states[age, treatment]
  )
  baseline = ND_REFERENCE_COST[age]
  return (
    baseline * mult,
    baseline * (1 + pct_low / 100),
    baseline * (1 + pct_high / 100),
  )


def publication_curve(age, treatment, scr_values, target_ratio):
  values = np.array([
    publication_point(age, treatment, scr, target_ratio)
    for scr in scr_values
  ])
  return values[:, 0], values[:, 1], values[:, 2]


# %% TWO PUBLICATION FIGURES: ORIGINAL INFANT AND FOUR-YEAR-OLD ---------
def plot_scr(treatment, age):
  infant = age == "infant"
  ref_scr, ref_mgs = nd_policy[age]
  ref_ratio = ref_mgs / ref_scr

  if infant:
    series = [(np.linspace(3, 6, 181), ref_ratio, COLORS["infant"], age)]
    highlighted = [(ref_scr, ND_REFERENCE_COST[age], COLORS[age], False)]
    xlim = (3, 6)
    xticks = INFANT_SCR_TICKS
  else:
    series = [
      (np.linspace(8, 20, 241), ref_ratio, COLORS["four_year_old"], "four_year_old"),
      (np.linspace(6, 15, 181), ND_THREE_YEAR_MGS / ND_THREE_YEAR_SCR,
       COLORS["three_year"], "three_year_proxy"),
    ]
    predicted_3y, *_ = publication_point(
      age, treatment, ND_THREE_YEAR_SCR,
      ND_THREE_YEAR_MGS / ND_THREE_YEAR_SCR
    )
    highlighted = [
      (ref_scr, ND_REFERENCE_COST[age], COLORS[age], True),
      (ND_THREE_YEAR_SCR, predicted_3y, COLORS["three_year"], True),
    ]
    xlim = (6, 20)
    xticks = PRESCHOOL_SCR_TICKS

  plotted = []
  for xs, ratio, color, name in series:
    central, low, high = publication_curve(age, treatment, xs, ratio)
    plotted.append((xs, central, low, high, color, name))

  all_values = np.concatenate([
    np.concatenate([low, high])
    for _, _, low, high, _, _ in plotted
  ] + [np.array([p[1] for p in highlighted])])
  ymin = max(500, 500 * np.floor((np.nanmin(all_values) - 750) / 500))
  ymax = max(ymin + 1000, 500 * np.ceil((np.nanmax(all_values) + 250) / 500))
  fig, ax, ax_zero = scr_axes(xlim, xticks, ymin, ymax)

  for xs, central, low, high, color, name in plotted:
    ax.fill_between(xs, low, high, color=color,
                    alpha=CI_BAND_ALPHA[name], linewidth=0, zorder=Z["band"])
    ax.plot(xs, central, color=color, lw=LINE_LW,
            zorder=Z["curve"] + (1 if name == "three_year_proxy" else 0))

  if infant:
    ratio = ref_ratio
    for scr in INFANT_SCR_TICKS:
      predicted, *_ = publication_point(age, treatment, scr, ratio)
      ax.scatter([scr], [predicted], s=POINT_SIZE, color=COLORS["infant"],
                 clip_on=False, zorder=Z["point"])
      scr_cost_label(ax, scr, predicted, COLORS["infant"], xlim[1])

  # %% PRESCHOOL ENDPOINT DOTS + LOWEST PRICE LABELS ------------------------
  # Preserve the two existing benchmark dots/labels/vertical guides below.
  # Add only one dot at the rightmost (lowest-price) endpoint of each curve.
  if not infant:
    for endpoint_scr, ratio, color, xytext, ha in (
      # Four-year-old: label immediately left and below the SCR=20 dot.
      (20, ref_ratio, COLORS["four_year_old"], (-4, -6), "right"),
      # Three-year-old proxy: label centered just below the SCR=15 dot.
      (15, ND_THREE_YEAR_MGS / ND_THREE_YEAR_SCR,
       COLORS["three_year"], (0, -10), "center"),
    ):
      endpoint_price, *_ = publication_point(
        age, treatment, endpoint_scr, ratio
      )
      ax.scatter([endpoint_scr], [endpoint_price], s=POINT_SIZE, color=color,
                 clip_on=False, zorder=Z["point"])
      ax.annotate(
        f"${endpoint_price:,.0f}", (endpoint_scr, endpoint_price),
        xytext=xytext, textcoords="offset points",
        ha=ha, va="top",
        color=color, fontsize=ANNOTATION_SIZE, fontweight=600,
        clip_on=False, zorder=Z["label"]
      )
  # %% END PRESCHOOL ENDPOINT DOTS + LOWEST PRICE LABELS --------------------

  for scr, price, color, annotate in highlighted:
    scr_vertical_guide(fig, ax, ax_zero, scr, price, color)
    ax.scatter([scr], [price], s=POINT_SIZE, color=color,
               clip_on=False, zorder=Z["point"])
    if annotate:
      scr_cost_label(ax, scr, price, color, xlim[1])

  scr_grid_and_frame(fig, ax, ax_zero)

  for tick, x in zip(ax_zero.xaxis.get_major_ticks(), ax_zero.get_xticks()):
    for ref, _, color, _ in highlighted:
      if np.isclose(x, ref):
        tick.label1.set_color(color)
        tick.label1.set_fontweight(600)
        tick.label1.set_fontsize(11)

  def ci_patch(color, alpha, label):
    pale_color = tuple(1 - alpha + alpha * channel for channel in to_rgb(color))
    return Patch(facecolor=pale_color, edgecolor="none", linewidth=0, label=label)

  if infant:
    handles = [
      Line2D([0], [0], color=COLORS["infant"], lw=LINE_LW,
             label=f"Infant, MGS/SCR = {ref_mgs:g}/{ref_scr:g}"),
      ci_patch(COLORS["infant"], CI_BAND_ALPHA["infant"], "95% confidence interval"),
    ]
  else:
    handles = [
      Line2D([0], [0], color=COLORS["four_year_old"], lw=LINE_LW,
             label=f"Four-year-old, MGS/SCR = {ref_mgs:g}/{ref_scr:g}"),
      Line2D([0], [0], color=COLORS["three_year"], lw=LINE_LW,
             label=(f"Three-year-old proxy, MGS/SCR = "
                    f"{ND_THREE_YEAR_MGS:g}/{ND_THREE_YEAR_SCR:g}")),
      ci_patch(COLORS["four_year_old"], CI_BAND_ALPHA["four_year_old"],
               "95% confidence interval (four-year-old)"),
      ci_patch(COLORS["three_year"], CI_BAND_ALPHA["three_year_proxy"],
               "95% confidence interval (three-year-old proxy)"),
    ]

  legend = ax.legend(
    handles=handles, loc="upper right", frameon=True,
    facecolor="white", edgecolor=LEGEND_EDGE_COLOR,
    framealpha=1, borderpad=.7, handlelength=2.2, fontsize=9
  )
  legend.get_frame().set_linewidth(1)
  legend.set_zorder(Z["axis"] + 2)

  stem = f"nd_{'infant' if infant else 'four_and_three_year'}_scr_{treatment}"
  fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=SAVE_DPI)
  fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
  plt.show()
  plt.close(fig)
  print(f"Saved {stem}.png and {stem}.pdf")


for treatment in TREATMENTS:
  for age in AGES:
    plot_scr(treatment, age)

print(f"\nSaved tables and two publication figures to: {OUTPUT_DIR}")
print("All main price curves hold MGS/SCR fixed; fixed MGS is a separate table sensitivity.")
print("2025 ND prices anchor the simulations; regression uses 2013/2018/2023 prices.")
print("Confidence bands reflect clustered coefficient uncertainty, not benchmark uncertainty.")


# %% MANUSCRIPT UPDATE: EXACT NORTH DAKOTA POLICY ESTIMATES -----------------
REPORT_DIR = OUTPUT_DIR / "manuscript_updates"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def manuscript_scenario(treatment, key, description, age, old, new):
  """Estimate policy change and price levels from one fixed ND anchor."""
  model = models[age, treatment]
  clusters = model_states[age, treatment]
  anchor_scr, anchor_mgs = nd_policy[age]
  anchor_price = ND_REFERENCE_COST[age]
  old_scr, old_mgs = old
  new_scr, new_mgs = new

  initial_mult, *_ = effect(model, anchor_scr, anchor_mgs, old_scr, old_mgs, clusters)
  final_mult, _, final_ci_low, final_ci_high = effect(
    model, anchor_scr, anchor_mgs, new_scr, new_mgs, clusters
  )
  _, change_pct, change_ci_low, change_ci_high = effect(
    model, old_scr, old_mgs, new_scr, new_mgs, clusters
  )

  initial_price = anchor_price * initial_mult
  final_price = anchor_price * final_mult
  return {
    "treatment": treatment, "scenario_key": key, "description": description,
    "age_model": age,
    "anchor_scr": anchor_scr, "anchor_mgs": anchor_mgs,
    "anchor_price_2025_dollars": anchor_price,
    "old_scr": old_scr, "old_mgs": old_mgs,
    "new_scr": new_scr, "new_mgs": new_mgs,
    "starting_price": initial_price, "estimated_price": final_price,
    "change_dollars": final_price - initial_price,
    "change_percent": change_pct,
    "change_percent_ci_low": change_ci_low,
    "change_percent_ci_high": change_ci_high,
    "estimated_price_ci_low": anchor_price * (1 + final_ci_low / 100),
    "estimated_price_ci_high": anchor_price * (1 + final_ci_high / 100),
    "monthly_1000_after": 1000 * (1 + change_pct / 100),
    "monthly_1000_savings": -1000 * change_pct / 100,
    "n_obs": int(model.nobs), "n_state_clusters": clusters,
  }


report_rows = []
for treatment in TREATMENTS:
  infant_scr, infant_mgs = nd_policy["infant"]
  four_scr, four_mgs = nd_policy["four_year_old"]
  infant_ratio = infant_mgs / infant_scr
  four_ratio = four_mgs / four_scr
  three_ratio = ND_THREE_YEAR_MGS / ND_THREE_YEAR_SCR
  three_start = (ND_THREE_YEAR_SCR, ND_THREE_YEAR_MGS)
  three_target_scr = ND_THREE_YEAR_NEW_SCR

  cases = [
    (
      "infant_4_to_5_ratio_fixed",
      "Infant SCR 4 to 5; MGS/SCR fixed (Figure 4)",
      "infant", (infant_scr, infant_mgs),
      (5.0, 5.0 * infant_ratio),
    ),
    (
      "four_10_to_12_ratio_fixed",
      "Four-year-old SCR 10 to 12; MGS/SCR fixed (Figure 3)",
      "four_year_old", (four_scr, four_mgs), (12.0, 12.0 * four_ratio),
    ),
    (
      "four_at_three_rules",
      "Four-year-old benchmark under three-year-old rules (7, 20)",
      "four_year_old", (four_scr, four_mgs), three_start,
    ),
    (
      "three_7_to_10_ratio_fixed",
      "Three-year-old proxy SCR 7 to 10; MGS/SCR fixed (orange curve)",
      "four_year_old", three_start,
      (three_target_scr, three_target_scr * three_ratio),
    ),
    (
      "three_7_to_10_mgs_25",
      "Three-year-old proxy SCR 7 to 10; MGS increases 20 to 25 (draft scenario)",
      "four_year_old", three_start, (three_target_scr, 25.0),
    ),
    (
      "three_7_to_10_mgs_20",
      "Three-year-old proxy SCR 7 to 10; MGS stays at 20 (sensitivity)",
      "four_year_old", three_start, (three_target_scr, ND_THREE_YEAR_MGS),
    ),
    (
      "orange_at_scr_10",
      "Orange curve at SCR 10, using the 3y MGS/SCR ratio",
      "four_year_old", (four_scr, four_mgs),
      (four_scr, four_scr * three_ratio),
    ),
  ]
  for key, description, age, old, new in cases:
    report_rows.append(manuscript_scenario(treatment, key, description, age, old, new))

report = pd.DataFrame(report_rows)
report.to_csv(REPORT_DIR / "manuscript_policy_estimates.csv", index=False)

curve_rows = []
for treatment in TREATMENTS:
  for age, label, scr_values, ratio in (
    ("infant", "Infant", range(3, 7), nd_policy["infant"][1] / nd_policy["infant"][0]),
    ("four_year_old", "Four-year-old", range(8, 21),
     nd_policy["four_year_old"][1] / nd_policy["four_year_old"][0]),
    ("four_year_old", "Three-year-old proxy", range(6, 16),
     ND_THREE_YEAR_MGS / ND_THREE_YEAR_SCR),
  ):
    for scr in scr_values:
      price, low, high = publication_point(age, treatment, scr, ratio)
      curve_rows.append({
        "treatment": treatment, "curve": label, "scr": scr,
        "mgs_ratio": ratio, "implied_mgs": scr * ratio,
        "estimated_cost": price, "cost_ci_low": low, "cost_ci_high": high,
      })

curve_data = pd.DataFrame(curve_rows)
curve_data.to_csv(REPORT_DIR / "figure_3_4_integer_scr_prices.csv", index=False)

money = lambda n: f"${n:,.0f}"
percent = lambda n: f"{n:+.2f}%"
lines = [
  "# Original-price regression numbers for Expanding Access and Affordability", "",
  "All amounts are in 2025 dollars. The regression uses 2013, 2018,",
  "and 2023 state-year observations and North Dakota's 2023 policy anchors.",
  "North Dakota reference costs are treated as fixed benchmarks.",
  "The reported 95% intervals reflect state-clustered coefficient uncertainty.",
  "", "## Regression-based figures", "",
]

for treatment in TREATMENTS:
  lines.extend([f"### {treatment.title()} model", ""])
  group = report.loc[report.treatment.eq(treatment)]
  for row in group.itertuples(index=False):
    lines.append(f"**{row.description}**")
    lines.append(
      f"  Starting {money(row.starting_price)}; "
      f"scenario {money(row.estimated_price)}; "
      f"change {money(row.change_dollars)} ({percent(row.change_percent)})."
    )
    lines.append(
      f"  95% CI for percent change: "
      f"[{percent(row.change_percent_ci_low)}, "
      f"{percent(row.change_percent_ci_high)}]."
    )
    lines.append(
      f"  95% CI for scenario price from observed 4y/infant anchor: "
      f"[{money(row.estimated_price_ci_low)}, "
      f"{money(row.estimated_price_ci_high)}]."
    )
    if row.scenario_key.startswith("three_7_to_10"):
      lines.append(
        f"  $1,000/month illustration: "
        f"{money(row.monthly_1000_after)}/month; "
        f"savings {money(row.monthly_1000_savings)}/month."
      )
    lines.append("")

lines.extend([
  "## Figures requiring separate source data (not identified by this regression)", "",
  "- Table 3: 2025 age-specific national ratio/MGS medians and state counts",
  "  require the full 51-jurisdiction regulatory data at all seven ages.",
  "- The Common Schools Trust Fund $2,508 per pupil requires updated",
  "  state appropriation/distribution figures.",
  "- Labor-market numbers (LFPR, unemployment, job openings, rankings)",
  "  require current BLS Table 14 / LAUS / JOLTS data and consistent reference periods.",
  "- Population under age five, its ranking, and births require updated PEP data.",
  "- ND vs. SD median household income and childcare price differences",
  "  require matching-year, matching-age source observations.",
  "- The anchors of $12,281/$10,855 come from ND_REFERENCE_COST,",
  "  not from a new estimate of 2025 observed annual prices.", "",
  "IMPORTANT: the draft's 3-year-old scenario uses MGS 25 at SCR 10,",
  "but the orange publication curve holds MGS/SCR fixed and therefore",
  f"uses MGS {ND_THREE_YEAR_NEW_SCR * ND_THREE_YEAR_MGS / ND_THREE_YEAR_SCR:.4f} at SCR 10.",
  "Both results are reported separately; don't interchange them.", "",
])

(REPORT_DIR / "manuscript_policy_numbers.md").write_text(
  "\n".join(lines), encoding="utf-8"
)

print("\nMANUSCRIPT NUMBERS — ORIGINAL PRICE TREATMENT")
cols = [
  "description", "starting_price", "estimated_price", "change_dollars",
  "change_percent", "change_percent_ci_low", "change_percent_ci_high"
]
print(report.loc[report.treatment.eq("original"), cols].round(2).to_string(index=False))
print(f"\nSaved manuscript statistics to {REPORT_DIR}")

# %% ADDITIONAL MANUSCRIPT FIGURES: DEMOGRAPHICS ---------------------------
# Figure 1: Under-five population share, ND versus other states, 2005-2025.
# Figure 5: BLS Geographic Profile 2025, Table 14: women, map data only.
# Figure 6: Cumulative population change from 2005 and year-over-year change.
# Figure 7: North Dakota live births, 2005-2025.
#
# Population and under-five figures use the Census Population Estimates
# Program (PEP), not ACS or CPS. No CPS microdata are used anywhere here.
# The 2013/2018/2023 regression controls are read separately from CONTROL_FILE.

DEMOGRAPHIC_DIR = OUTPUT_DIR / "demographic_figures"
PEP_CACHE_DIR = DEMOGRAPHIC_DIR / "census_pep_cache"
DEMOGRAPHIC_YEARS = range(2005, 2026)
DEMOGRAPHIC_YEAR_TICKS = np.arange(2006, 2026, 2)
DEMOGRAPHIC_STATE = "North Dakota"
DEMOGRAPHIC_FIGURES = (1, 5, 6, 7)
DEMOGRAPHIC_REFRESH = False

# Optional explicit path to your EXISTING published BLS Table 14 CSV.
# If None, discover a qualified CSV or generate it from the saved 2025
# BLS Table 14 workbook under control_variables/sources/bls_female_lfpr/.
# An optional ND_BLS_FEMALE_LFPR_CSV environment variable can override this.
BLS_FEMALE_LFPR_CSV = None
BLS_FEMALE_LFPR_SOURCE = "https://www.bls.gov/opub/geographic-profile/2025/home.htm"

DEMOGRAPHIC_COLORS = {
  "north_dakota": "#003524",
  "other_states": "#D7E5C8",
  "population": "#0F374B",
  "births": "#51ABA0",
  "baseline": "#A9BCA5",
}

DEMOGRAPHIC_STATE_FIPS = {
  "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
  "06": "California", "08": "Colorado", "09": "Connecticut",
  "10": "Delaware", "11": "District of Columbia", "12": "Florida",
  "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
  "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky",
  "22": "Louisiana", "23": "Maine", "24": "Maryland", "25": "Massachusetts",
  "26": "Michigan", "27": "Minnesota", "28": "Mississippi",
  "29": "Missouri", "30": "Montana", "31": "Nebraska",
  "32": "Nevada", "33": "New Hampshire", "34": "New Jersey",
  "35": "New Mexico", "36": "New York", "37": "North Carolina",
  "38": "North Dakota", "39": "Ohio", "40": "Oklahoma", "41": "Oregon",
  "42": "Pennsylvania", "44": "Rhode Island",
  "45": "South Carolina", "46": "South Dakota", "47": "Tennessee",
  "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia",
  "53": "Washington", "54": "West Virginia", "55": "Wisconsin",
  "56": "Wyoming",
}

ND_BIRTH_COUNTS = [
  8_381, 8_616, 8_818, 8_931, 8_974, 9_088, 9_234,
  10_072, 10_591, 11_352, 11_265, 11_364, 10_738,
  10_630, 10_447, 10_051, 10_111, 9_557, 9_639, 9_622, 10_021,
]

DEMOGRAPHIC_PEP_SOURCES = [
  (
    range(2005, 2010),
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2000-2010/intercensal/state/st-est00int-agesex.csv",
  ),
  (
    range(2010, 2020),
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2010-2020/intercensal/state/asrh/sc-est2020int-alldata6.csv",
  ),
  (
    range(2020, 2026),
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2020-2025/state/asrh/sc-est2025-alldata6.csv",
  ),
]


def demographic_ensure_paths():
  for path in (DEMOGRAPHIC_DIR, PEP_CACHE_DIR):
    path.mkdir(parents=True, exist_ok=True)


def demographic_save_plot(fig, stem):
  fig.savefig(DEMOGRAPHIC_DIR / f"{stem}.png", dpi=SAVE_DPI)
  fig.savefig(DEMOGRAPHIC_DIR / f"{stem}.pdf")
  plt.show()
  plt.close(fig)
  print(f"Saved demographic figure: {stem}.png/.pdf")


def demographic_style_axis(ax, grid=True, baseline=None):
  ax.grid(False)
  ax.tick_params(color="black", labelcolor="black")
  if grid:
    for y in ax.get_yticks():
      if baseline is not None and np.isclose(y, baseline):
        continue
      if ax.get_ylim()[0] <= y <= ax.get_ylim()[1]:
        ax.axhline(
          y, color=LEGEND_EDGE_COLOR, lw=.8,
          alpha=.9, ls=GRID_LS, dashes=GRID_DASHES[1],
          zorder=Z["grid"]
        )
  for spine in ax.spines.values():
    spine.set_color("black")
    spine.set_linewidth(AXIS_LW)
    spine.set_zorder(Z["axis"])


def demographic_legend(ax, loc="upper right", ncol=1):
  handles = [
    Line2D([0], [0], lw=LINE_LW,
           color=DEMOGRAPHIC_COLORS["north_dakota"], label=DEMOGRAPHIC_STATE),
    Line2D([0], [0], lw=LINE_LW,
           color=DEMOGRAPHIC_COLORS["other_states"], label="Other States"),
  ]
  legend = ax.legend(
    handles=handles, loc=loc, ncol=ncol, frameon=True,
    facecolor="white", edgecolor=LEGEND_EDGE_COLOR,
    framealpha=1, borderpad=.7, handlelength=2.2,
  )
  legend.get_frame().set_linewidth(1)
  legend.set_zorder(Z["axis"] + 2)


def demographic_load_env():
  """Retained for BLS API and ACS income calls, not for CPS downloads."""
  try:
    from dotenv import load_dotenv
  except ImportError:
    return
  candidates = [PROJECT_DIR / ".env", Path.home() / "openscience" / ".env", Path.cwd() / ".env"]
  for candidate in candidates:
    if candidate.exists():
      load_dotenv(candidate, override=False)
  load_dotenv(override=False)


def demographic_request(url, params=None, stream=False):
  """Retry public Census PEP or ACS downloads and return the response."""
  last_error = None
  for attempt in range(5):
    try:
      response = requests.get(url, params=params, timeout=(20, 180), stream=stream)
      response.raise_for_status()
      return response
    except requests.RequestException as exc:
      last_error = exc
      if attempt < 4:
        time.sleep(min(2 ** (attempt + 1), 16))
  raise RuntimeError(f"Census download failed: {url}") from last_error


def demographic_pep_file(url, refresh=False):
  dest = PEP_CACHE_DIR / Path(url).name
  if dest.exists() and dest.stat().st_size > 0 and not refresh:
    return dest
  print(f"Downloading Census PEP: {dest.name}")
  temporary = dest.with_suffix(dest.suffix + ".partial")
  try:
    with demographic_request(url, stream=True) as response:
      with temporary.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
          if chunk:
            file.write(chunk)
    if temporary.stat().st_size < 1000:
      raise ValueError(f"Census PEP download unexpectedly small: {url}")
    temporary.replace(dest)
  finally:
    temporary.unlink(missing_ok=True)
  return dest


def demographic_parse_pep(path, years):
  """Extract total and under-five populations from one Census PEP vintage."""
  data = pd.read_csv(path, low_memory=False)
  data.columns = data.columns.astype(str).str.strip().str.upper()
  required_cols = {"STATE", "SEX", "AGE"}
  if not required_cols.issubset(data.columns):
    raise ValueError(f"{path.name}: missing {sorted(required_cols - set(data.columns))}")

  for col in ("STATE", "SEX", "AGE", "ORIGIN", "RACE"):
    if col in data.columns:
      data[col] = pd.to_numeric(data[col], errors="coerce")

  data = data.loc[data["SEX"].eq(0)].copy()
  if "ORIGIN" in data.columns:
    data = data.loc[data["ORIGIN"].eq(0)].copy()
  if "RACE" in data.columns:
    if data["RACE"].eq(0).any():
      data = data.loc[data["RACE"].eq(0)].copy()
    else:
      data = data.loc[data["RACE"].between(1, 6)].copy()

  data["state_fips"] = data["STATE"].astype("Int64").astype("string").str.zfill(2)
  data = data.loc[data.state_fips.isin(DEMOGRAPHIC_STATE_FIPS)].copy()

  output = []
  for year in years:
    choices = [f"POPESTIMATE{year}", f"POP_ESTIMATE_{year}", f"POP{year}"]
    column = next((c for c in choices if c in data.columns), None)
    if column is None:
      raise KeyError(f"{path.name}: no population column for {year}")

    working = data[["state_fips", "AGE"]].copy()
    working["value"] = pd.to_numeric(data[column], errors="coerce")
    under5 = (
      working.loc[working["AGE"].between(0, 4)]
      .groupby("state_fips")["value"].sum(min_count=1)
    )
    if working["AGE"].eq(999).any():
      totals = (
        working.loc[working["AGE"].eq(999)]
        .groupby("state_fips")["value"].sum(min_count=1)
      )
    else:
      totals = (
        working.loc[working["AGE"].between(0, 100)]
        .groupby("state_fips")["value"].sum(min_count=1)
      )

    for fips, state in DEMOGRAPHIC_STATE_FIPS.items():
      output.append({
        "state_fips": fips, "state": state, "year": year,
        "population": float(totals.get(fips, np.nan)),
        "population_under5": float(under5.get(fips, np.nan)),
        "census_vintage": path.name,
      })
  return pd.DataFrame(output)


def demographic_pep_panel(refresh=False):
  pieces = [
    demographic_parse_pep(demographic_pep_file(url, refresh), years)
    for years, url in DEMOGRAPHIC_PEP_SOURCES
  ]
  data = pd.concat(pieces, ignore_index=True).sort_values(["state", "year"]).reset_index(drop=True)
  if data.duplicated(["state", "year"]).any():
    raise ValueError("Duplicate PEP state-years")
  if len(data) != 51 * len(DEMOGRAPHIC_YEARS):
    raise ValueError("Expected 51 state/DC rows per year from Census PEP")
  if data[["population", "population_under5"]].isna().any().any():
    raise ValueError("Missing state/year population in Census PEP data")
  if (data.population <= 0).any() or (data.population_under5 < 0).any():
    raise ValueError("Invalid Census population totals")

  data["under5_pct"] = 100 * data.population_under5 / data.population
  if not data.under5_pct.between(1, 15).all():
    raise ValueError("Unexpected under-five population percentage")
  data["under5_rank_51"] = (
    data.groupby("year")["under5_pct"]
    .rank(method="min", ascending=False).astype(int)
  )
  return data


def demographic_population_changes(pep):
  data = pep.sort_values(["state", "year"]).copy()
  base = (
    data.loc[data.year.eq(2005), ["state", "population"]]
    .rename(columns={"population": "population_2005"})
  )
  data = data.merge(base, on="state", how="left", validate="many_to_one")
  data["population_index_2005_100"] = 100 * data.population / data.population_2005
  data["population_change_from_2005_pct"] = data.population_index_2005_100 - 100
  data["population_yoy_pct"] = (
    100 * data.groupby("state")["population"].pct_change(fill_method=None)
  )
  return data


# %% FIGURE 1: PERCENTAGE UNDER FIVE ---------------------------------------
def demographic_broken_axes(ymin, ymax, zero_max, ylabel, percent=False, ylabel_x=.015):
  fig = plt.figure(figsize=FIGSIZE)
  grid = fig.add_gridspec(2, 1, height_ratios=SCR_HEIGHT_RATIOS, hspace=0)
  ax = fig.add_subplot(grid[0])
  ax_zero = fig.add_subplot(grid[1], sharex=ax)
  ax_zero.patch.set_visible(False)
  fig.subplots_adjust(**LAYOUT)
  ax.set(xlim=(2005, 2025), ylim=(ymin, ymax))
  ax_zero.set(xlim=(2005, 2025), ylim=(0, zero_max))
  ax_zero.set_yticks([0], labels=["0%" if percent else "0"])
  ax.tick_params(axis="x", bottom=False, labelbottom=False)
  ax_zero.set_xlabel("Year")
  fig.supylabel(ylabel, x=ylabel_x, fontsize=FONT_SIZE)
  ax_zero.set_xticks(DEMOGRAPHIC_YEAR_TICKS)
  return fig, ax, ax_zero


def plot_figure_1_under5(pep):
  """Show North Dakota's under-five share versus the other 49 states."""
  data = pep.loc[pep.state.ne("District of Columbia")]
  low = .5 * np.floor(2 * (data.under5_pct.min() - .35))
  high = .5 * np.ceil(2 * (data.under5_pct.max() + .35))
  fig, ax, ax_zero = demographic_broken_axes(
    max(1, low), high, .4, "Percentage Under Age 5",
    percent=True, ylabel_x=.06
  )
  for state, group in data.groupby("state"):
    is_nd = state == DEMOGRAPHIC_STATE
    color = DEMOGRAPHIC_COLORS["north_dakota" if is_nd else "other_states"]
    ax.plot(group.year, group.under5_pct,
            color=color, lw=LINE_LW if is_nd else 1.0,
            zorder=Z["curve"] + 1 if is_nd else Z["band"])
    if is_nd:
      ax.scatter(group.year, group.under5_pct, s=POINT_SIZE,
                 color=color, clip_on=False, zorder=Z["point"])
  ticks = np.arange(np.ceil(max(1, low)), high + .01, 1)
  ax.set_yticks(ticks)
  ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}%"))
  scr_grid_and_frame(fig, ax, ax_zero)
  demographic_legend(ax)
  demographic_save_plot(fig, "figure_1_under5_share_2005_2025")


# %% FIGURE 5: PUBLISHED BLS FEMALE LABOR FORCE PARTICIPATION -------------
def _is_published_bls_female_file(path):
  """Recognize the supplied 2025 published BLS Table 14 CSV, not old CPS outputs."""
  try:
    head = pd.read_csv(path, nrows=4, dtype={"state_fips": "string"})
  except (OSError, ValueError, pd.errors.ParserError, UnicodeError):
    return False
  required = {"state", "female_lfpr_source", "female_lfpr_2025"}
  if not required.issubset(head.columns) or head.empty:
    return False
  src = head["female_lfpr_source"].fillna("").astype(str)
  return bool(
    src.str.contains("BLS", case=False, regex=False).all()
    and src.str.contains("Table 14", case=False, regex=False).all()
  )


def build_published_bls_2025_from_xlsx():
  """Convert the locally saved 2025 BLS Table 14 workbook to map-ready CSV.

  Read only top-level Women (group code 03), not race-specific subgroups.
  Preserve the BLS published participation rates, rather than estimating
  female LFPR from CPS microdata or the 2013/2018/2023 controls.
  """
  source = (
    PROJECT_DIR / "control_variables/sources/bls_female_lfpr/ptable14full2025.xlsx"
  )
  if not source.is_file():
    raise FileNotFoundError(
      "No 2025 BLS Table 14 source workbook found at "
      f"{source}. Add the original ptable14full2025.xlsx to the project's "
      "control_variables/sources/bls_female_lfpr/ folder."
    )

  def _text(value):
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()

  def _number(value):
    if value is None or isinstance(value, bool):
      return None
    try:
      return float(str(value).replace(",", "").replace("%", "").strip())
    except ValueError:
      return None

  state_to_fips = {name: code for code, name in DEMOGRAPHIC_STATE_FIPS.items()}
  states_upper = {name.upper(): name for name in state_to_fips}
  rows = []
  found = set()
  wb = load_workbook(source, read_only=True, data_only=True)
  try:
    for sheet in wb.worksheets:
      # The official Table 14 workbook has a header row identifying State,
      # Group and Civilian Labor Force in columns C, D and F, respectively.
      has_header = any(
        len(row) >= 7 and _text(row[2]) == "STATE"
        and _text(row[3]) in {"GROUP", "POPULATION GROUP"}
        and "CIVILIAN LABOR FORCE" in _text(row[5])
        for row in sheet.iter_rows(
          min_row=1, max_row=min(12, sheet.max_row), values_only=True
        )
      )
      if not has_header:
        continue
      for line_number, cells in enumerate(sheet.iter_rows(values_only=True), 1):
        if len(cells) < 7 or _text(cells[3]) != "WOMEN":
          continue
        state = states_upper.get(_text(cells[2]))
        if state is None:
          continue
        fips = _number(cells[0])
        group = _number(cells[1])
        population = _number(cells[4])
        labor_force = _number(cells[5])
        rate = _number(cells[6])
        if (fips is None or int(fips) != int(state_to_fips[state])
            or group is None or int(group) != 3):
          raise ValueError(
            f"{source.name}/{sheet.title} row {line_number}: "
            f"unexpected state FIPS or Women group code for {state}"
          )
        if (population is None or labor_force is None or rate is None
            or population <= 0 or not 20 < rate < 90
            or abs(100 * labor_force / population - rate) > 1.5):
          raise ValueError(
            f"{source.name}/{sheet.title} row {line_number}: "
            f"invalid female labor-force participation rate for {state}"
          )
        if state in found:
          raise ValueError(f"Duplicate 2025 BLS Table 14 Women row: {state}")
        found.add(state)
        rows.append({
          "state": state,
          "state_fips": state_to_fips[state],
          "year": 2025,
          "female_lfpr_2025": rate / 100,
          "female_lfpr_source": "BLS Geographic Profile 2025 Table 14, Women",
          "female_lfpr_source_url": BLS_FEMALE_LFPR_SOURCE,
          "female_lfpr_workbook": source.name,
        })
  finally:
    wb.close()

  missing = set(state_to_fips) - found
  if len(rows) != 51 or missing:
    raise ValueError(
      f"Expected 51 published 2025 BLS Women rows in {source.name}; "
      f"found {len(rows)}. Missing: {sorted(missing)}. "
      "Check the source workbook's Table 14 layout."
    )

  output = PROJECT_DIR / "control_variables/state_female_lfpr_2025_bls.csv"
  pd.DataFrame(rows).sort_values("state").to_csv(output, index=False)
  print(f"Created 2025 BLS Table 14 CSV from local workbook: {output}")
  return output


def find_published_bls_female_file(explicit=None):
  """Locate a verified 2025 BLS Table 14 CSV, or derive it from source XLSX."""
  candidate = explicit or os.getenv("ND_BLS_FEMALE_LFPR_CSV")
  if candidate:
    path = Path(candidate).expanduser()
    if not path.is_file():
      raise FileNotFoundError(f"BLS Table 14 CSV does not exist: {path}")
    if not _is_published_bls_female_file(path):
      raise ValueError(
        f"Not a published BLS 2025 Table 14 CSV (metadata missing): {path}"
      )
    return path

  # Discover a CSV with *published BLS Table 14 provenance*. Old files built
  # from Census CPS microdata deliberately fail the provenance check.
  candidates = []
  visited = set()
  for folder in (PROJECT_DIR / "control_variables", PROJECT_DIR):
    if not folder.exists():
      continue
    # Scan informative filenames before broader 2025 candidates.
    for pattern in ("*female*2025*.csv", "*lfpr*.csv", "*bls*2025*.csv", "*2025*.csv"):
      for path in folder.rglob(pattern):
        if path in visited or OUTPUT_DIR in path.parents:
          continue
        visited.add(path)
        if _is_published_bls_female_file(path):
          candidates.append(path.resolve())
    if candidates:
      break

  matches = sorted(set(candidates))
  if not matches:
    # A clean reproducibility checkout retains source workbooks rather than
    # generated CSVs. Reconstruct this auxiliary 2025 map input on demand.
    return build_published_bls_2025_from_xlsx()
  if len(matches) != 1:
    listed = "\n".join(str(p) for p in matches)
    raise ValueError(
      "Found several published BLS Table 14 CSV files. "
      "Set BLS_FEMALE_LFPR_CSV explicitly to choose one:\n" + listed
    )
  return matches[0]


def demographic_female_lfpr(csv_path=None):
  """Load published 2025 BLS Table 14 rates (NOT a CPS microdata estimate)."""
  path = find_published_bls_female_file(csv_path)
  data = pd.read_csv(path, dtype={"state_fips": "string"})
  data.columns = data.columns.astype(str).str.strip()
  data["state"] = data["state"].astype("string").str.strip()
  if "year" in data:
    data["year"] = pd.to_numeric(data["year"], errors="raise")
    data = data.loc[data.year.eq(2025)].copy()

  needed = {"state", "female_lfpr_2025", "female_lfpr_source"}
  if missing := needed - set(data.columns):
    raise ValueError(f"Published BLS Table 14 CSV missing {sorted(missing)}: {path}")
  if len(data) != 51 or data.state.nunique() != 51:
    raise ValueError(f"Expected 50 states and DC, 51 unique rows in {path}")
  if set(data.state) != set(DEMOGRAPHIC_STATE_FIPS.values()):
    raise ValueError(f"2025 BLS CSV does not have all 50 states plus DC: {path}")
  sources = data["female_lfpr_source"].fillna("").astype(str)
  if not (
    sources.str.contains("BLS", case=False, regex=False).all()
    and sources.str.contains("Table 14", case=False, regex=False).all()
  ):
    raise ValueError(f"Published BLS provenance not present for all rows: {path}")

  data["female_lfpr_2025"] = pd.to_numeric(data.female_lfpr_2025, errors="raise")
  if not data.female_lfpr_2025.between(0, 1).all():
    raise ValueError("BLS female_lfpr_2025 must be a proportion between 0 and 1")

  data["female_lfpr_2025_pct"] = 100 * data.female_lfpr_2025
  data["female_lfpr_2025_rank_51"] = (
    data.female_lfpr_2025.rank(method="min", ascending=False).astype(int)
  )
  data["state_fips"] = data.state.map({
    name: code for code, name in DEMOGRAPHIC_STATE_FIPS.items()
  })
  data["female_lfpr_source_file"] = str(path)
  if "female_lfpr_source_url" not in data.columns:
    data["female_lfpr_source_url"] = BLS_FEMALE_LFPR_SOURCE

  nd_rate = float(data.loc[data.state.eq("North Dakota"), "female_lfpr_2025"].iloc[0])
  if not np.isclose(nd_rate, 0.652, atol=1e-10):
    print(f"WARNING: Published BLS input gives ND {100 * nd_rate:.1f}% rather than "
          "the published 65.1%; review the source file.")
  print(f"Published BLS Table 14 female LFPR: {path}")
  return data.sort_values("female_lfpr_2025_rank_51").reset_index(drop=True)


# %% FIGURE 6: CUMULATIVE AND YEAR-OVER-YEAR POPULATION CHANGE -----------
def plot_figure_6_population(changes):
  data = changes.loc[changes.state.ne("District of Columbia")]
  fig = plt.figure(figsize=(6.5, 6.5))
  grid = fig.add_gridspec(2, 1, height_ratios=(4, 1.5), hspace=.08)
  ax_top = fig.add_subplot(grid[0])
  ax_bottom = fig.add_subplot(grid[1], sharex=ax_top)
  fig.subplots_adjust(left=.16, right=.97, bottom=.10, top=.985)

  for state, group in data.groupby("state"):
    is_nd = state == DEMOGRAPHIC_STATE
    color = DEMOGRAPHIC_COLORS["north_dakota" if is_nd else "other_states"]
    width = LINE_LW if is_nd else 1.0
    zorder = Z["curve"] + 1 if is_nd else Z["band"]
    ax_top.plot(group.year, group.population_change_from_2005_pct,
                color=color, lw=width, zorder=zorder)
    ax_bottom.plot(group.year, group.population_yoy_pct,
                   color=color, lw=width, zorder=zorder)
    if is_nd:
      ax_top.scatter(group.year, group.population_change_from_2005_pct,
                     s=POINT_SIZE, color=color, clip_on=False, zorder=Z["point"])
      available = group.dropna(subset=["population_yoy_pct"])
      ax_bottom.scatter(available.year, available.population_yoy_pct,
                        s=POINT_SIZE, color=color, clip_on=False, zorder=Z["point"])

  ax_top.set_ylabel("Percent Change From 2005 (2005 = 0%)", labelpad=10)
  ax_bottom.set_ylabel("Year-Over-Year Change (%)", labelpad=12)
  ax_bottom.set_xlabel("Year")
  for axis in (ax_top, ax_bottom):
    axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))

  ax_top.set_xlim(2005, 2025)
  upper_low = data.population_change_from_2005_pct.min()
  upper_high = data.population_change_from_2005_pct.max()
  ax_top.set_ylim(
    5 * np.floor((upper_low - 5) / 5),
    5 * np.ceil((upper_high + 5) / 5)
  )
  yoy = data.population_yoy_pct.dropna()
  yoy_lim = max(2, 2 * np.ceil((yoy.abs().max() + .5) / 2))
  ax_bottom.set_ylim(-yoy_lim, yoy_lim)
  ax_bottom.set_xticks(DEMOGRAPHIC_YEAR_TICKS)
  ax_top.tick_params(axis="x", bottom=False, labelbottom=False)

  demographic_style_axis(ax_top, baseline=0)
  demographic_style_axis(ax_bottom, baseline=0)
  for axis in (ax_top, ax_bottom):
    axis.axhline(0, color=DEMOGRAPHIC_COLORS["baseline"],
                 ls=GRID_LS, dashes=GRID_DASHES[1], lw=.95,
                 alpha=1, zorder=Z["curve"] - .2)
  demographic_legend(ax_top, loc="upper left", ncol=2)
  demographic_save_plot(fig, "figure_6_population_change_yoy_2005_2025")


# %% FIGURE 7: BIRTHS PER YEAR ---------------------------------------------
def plot_figure_7_births(births):
  low = 500 * np.floor((births.births.min() - 500) / 500)
  high = 500 * np.ceil((births.births.max() + 500) / 500)
  fig, ax, ax_zero = demographic_broken_axes(
    low, high, min(700, max(100, .08 * low)), "Births"
  )
  ax.plot(births.year, births.births, color=DEMOGRAPHIC_COLORS["births"],
          lw=LINE_LW, zorder=Z["curve"])
  ax.scatter(births.year, births.births, s=POINT_SIZE,
             color=DEMOGRAPHIC_COLORS["births"],
             clip_on=False, zorder=Z["point"])
  ax.set_yticks(np.arange(low + 500, high + 1, 500))
  ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
  scr_grid_and_frame(fig, ax, ax_zero)
  demographic_save_plot(fig, "figure_7_nd_births_2005_2025")


def build_demographic_figures(figures=DEMOGRAPHIC_FIGURES, refresh=False, lfpr_csv=None):
  """Generate PEP Figures 1/6, births Figure 7, and BLS Figure 5 CSVs."""
  selected = set(figures)
  if not selected or not selected.issubset({1, 5, 6, 7}):
    raise ValueError("Choose one or more figures from (1, 5, 6, 7)")
  demographic_ensure_paths()

  if selected.intersection({1, 6}):
    pep = demographic_pep_panel(refresh=refresh)
    pep.to_csv(DEMOGRAPHIC_DIR / "census_pep_state_year_2005_2025.csv", index=False)
    if 1 in selected:
      pep.loc[pep.state.ne("District of Columbia")].to_csv(
        DEMOGRAPHIC_DIR / "figure_1_under5_state_year.csv", index=False
      )
      plot_figure_1_under5(pep)
    if 6 in selected:
      changes = demographic_population_changes(pep)
      changes.to_csv(DEMOGRAPHIC_DIR / "figure_6_population_state_year.csv", index=False)
      changes.loc[changes.state.eq(DEMOGRAPHIC_STATE), [
        "year", "population", "population_2005",
        "population_index_2005_100", "population_change_from_2005_pct",
        "population_yoy_pct",
      ]].to_csv(DEMOGRAPHIC_DIR / "nd_population_2005_2025.csv", index=False)
      plot_figure_6_population(changes)

    last = pep.loc[pep.state.eq(DEMOGRAPHIC_STATE) & pep.year.eq(2025)].iloc[0]
    print(
      f"ND under-five share, 2025: {last.under5_pct:.2f}% "
      f"(rank {int(last.under5_rank_51)} of 51, including DC)"
    )

  if 5 in selected:
    lfpr = demographic_female_lfpr(csv_path=lfpr_csv)
    lfpr.to_csv(DEMOGRAPHIC_DIR / "figure_5_female_lfpr_state_2025.csv", index=False)
    lfpr[["state", "female_lfpr_2025"]].rename(columns={
      "female_lfpr_2025": "female_lfpr"
    }).sort_values("state").to_csv(
      DEMOGRAPHIC_DIR / "state_female_lfpr_2025_map.csv", index=False
    )
    nd_lfpr = lfpr.loc[lfpr.state.eq(DEMOGRAPHIC_STATE)].iloc[0]
    print(
      f"ND published BLS female LFPR, 2025: {nd_lfpr.female_lfpr_2025_pct:.1f}% "
      f"(rank {int(nd_lfpr.female_lfpr_2025_rank_51)} of 51 including DC)"
    )

  if 7 in selected:
    births = pd.DataFrame({
      "year": list(DEMOGRAPHIC_YEARS),
      "births": ND_BIRTH_COUNTS,
      "source": "Births counts carried from prior script",
    })
    births.to_csv(DEMOGRAPHIC_DIR / "figure_7_nd_births_2005_2025.csv", index=False)
    plot_figure_7_births(births)
    print("Note: verify that the historical births series has not been revised.")

  notes = [
    "Figure 1: Census PEP resident population by age/state, 2005-2025.",
    "Figure 5: published BLS Geographic Profile 2025 Table 14 (women); map data only.",
    f"BLS Table 14 source: {BLS_FEMALE_LFPR_SOURCE}",
    "Published 2025 BLS annual profile uses 11 CPS survey months; October excluded.",
    "No CPS microdata were downloaded or reaggregated by this script.",
    "Figure 6: Census PEP, cumulative change from 2005 (2005=0%) and annual percent change.",
    "Figure 7: births counts carried forward from the original user script.",
    "Figures 1 and 6 compare 50 states; the Figure 5 data include DC (51 jurisdictions).",
    "PEP source years 2005-2009: " + DEMOGRAPHIC_PEP_SOURCES[0][1],
    "PEP source years 2010-2019: " + DEMOGRAPHIC_PEP_SOURCES[1][1],
    "PEP source years 2020-2025: " + DEMOGRAPHIC_PEP_SOURCES[2][1],
    "PEP combines three vintages; check the 2010 and 2020 seams.",
  ]
  (DEMOGRAPHIC_DIR / "data_source_notes.txt").write_text(
    "\n".join(notes) + "\n", encoding="utf-8"
  )
  print(f"\nSaved demographic figures and source CSVs to: {DEMOGRAPHIC_DIR}")

# %% WORKFORCE INDICATORS AND REPRODUCIBLE HISTORICAL SUMMARIES -----------
# Latest complete 2026 state LFPR/unemployment, 2025 state JOLTS,
# 2025 annual average job openings, and published 2025 BLS Table 14
# female LFPR for the map (not a separately calculated CPS series).
# Retain 2021-2025 state averages and full 2026 monthly LAUS as audit data.
# All BLS measures are NOT seasonally adjusted; use common-month comparisons.
# Cache downloads; WORKFORCE_REFRESH=True updates underlying BLS API data.

LABOR_MARKET_DIR = OUTPUT_DIR / "labor_market_summary"
WORKFORCE_YEAR = 2026
WORKFORCE_JOLTS_YEAR = 2025
WORKFORCE_HISTORY_YEARS = range(2021, 2026)
WORKFORCE_REFRESH = False


def workforce_series(year):
  """State LAUS series; include state JOLTS for 2021-2025."""
  rows = []
  for fips, state in DEMOGRAPHIC_STATE_FIPS.items():
    prefix = "LAUST" + fips + "00000000000"
    for code, measure in (
      ("06", "labor_force"),
      ("09", "civilian_population"),
      ("04", "unemployed_persons"),
    ):
      rows.append({
        "series_id": prefix + code,
        "state_fips": fips, "state": state, "measure": measure,
      })
    if year in WORKFORCE_HISTORY_YEARS:
      rows.append({
        "series_id": "JTU000000" + fips + "0000000JOL",
        "state_fips": fips, "state": state,
        "measure": "job_openings_thousands",
      })
  return pd.DataFrame(rows)


def workforce_bls_download(year, refresh=False):
  """Download/cached BLS observations for one year, in batches of 25."""
  cache = LABOR_MARKET_DIR / f"bls_state_series_{year}.csv"
  if cache.exists() and not refresh:
    data = pd.read_csv(cache, dtype={"state_fips": "string"}, parse_dates=["Date"])
    if {"state_fips", "state", "Date", "measure", "value"}.issubset(data.columns):
      return data

  demographic_load_env()
  key = os.getenv("BLS_API_KEY", "")
  if not key:
    raise RuntimeError(
      "BLS_API_KEY not found. Check the environment or ~/openscience/.env."
    )

  metadata = workforce_series(year)
  rows = []
  series_ids = metadata.series_id.tolist()
  endpoint = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
  for start in range(0, len(series_ids), 25):
    batch = series_ids[start:start + 25]
    payload = {
      "seriesid": batch,
      "startyear": str(year), "endyear": str(year),
      "registrationkey": key,
    }
    for attempt in range(5):
      try:
        response = requests.post(endpoint, json=payload, timeout=(20, 180))
        response.raise_for_status()
        result = response.json()
        if result.get("status") != "REQUEST_SUCCEEDED":
          raise RuntimeError(str(result.get("message", result)))
        break
      except (requests.RequestException, RuntimeError) as exc:
        if attempt == 4:
          raise RuntimeError(f"BLS {year} request failed: {exc}") from exc
        time.sleep(min(2 ** (attempt + 1), 16))

    for series in result.get("Results", {}).get("series", []):
      for observation in series.get("data", []):
        period = observation.get("period", "")
        if not re.fullmatch(r"M(0[1-9]|1[0-2])", period):
          continue
        value = pd.to_numeric(
          str(observation.get("value", "")).replace(",", ""), errors="coerce"
        )
        if np.isfinite(value):
          rows.append({
            "series_id": series.get("seriesID"),
            "Date": pd.Timestamp(
              year=int(observation["year"]),
              month=int(period[1:]), day=1,
            ),
            "value": float(value),
          })
    time.sleep(.25)

  if not rows:
    raise RuntimeError(f"No monthly BLS data returned for {year}.")
  data = pd.DataFrame(rows).merge(
    metadata, on="series_id", how="left", validate="many_to_one"
  )
  if data[["state", "measure"]].isna().any().any():
    raise RuntimeError("Unrecognized BLS state/measure IDs.")
  if data.duplicated(["state", "measure", "Date"]).any():
    raise RuntimeError("Duplicate BLS state/month/measure values.")
  data.to_csv(cache, index=False)
  return data


def workforce_monthly(year, refresh=False):
  """Prefer existing 2025/2026 caches; otherwise request one year."""
  old_2025 = LABOR_MARKET_DIR / "bls_labor_monthly_2021_2025.csv"
  old_2026 = LABOR_MARKET_DIR / "bls_laus_2026_long.csv"

  if year in WORKFORCE_HISTORY_YEARS and old_2025.exists() and not refresh:
    monthly = pd.read_csv(
      old_2025, parse_dates=["Date"], dtype={"state_fips": "string"}
    )
    monthly = monthly.loc[monthly.Date.dt.year.eq(year)].copy()
    if monthly.empty:
      raise RuntimeError(f"Historical BLS cache lacks data for {year}.")
  else:
    if year == 2026 and old_2026.exists() and not refresh:
      long = pd.read_csv(
        old_2026, dtype={"state_fips": "string"}, parse_dates=["Date"]
      )
    else:
      long = workforce_bls_download(year, refresh=refresh)
    monthly = (
      long.pivot_table(
        index=["state_fips", "state", "Date"],
        columns="measure", values="value", aggfunc="first"
      )
      .reset_index()
    )
    monthly.columns.name = None

  for col in ("labor_force", "civilian_population", "unemployed_persons"):
    if col not in monthly.columns:
      raise RuntimeError(f"Missing {year} BLS series: {col}")
    monthly[col] = pd.to_numeric(monthly[col], errors="coerce")

  monthly["lfpr_pct"] = 100 * monthly.labor_force / monthly.civilian_population
  monthly["unemployment_rate_pct"] = (
    100 * monthly.unemployed_persons / monthly.labor_force
  )
  if year in WORKFORCE_HISTORY_YEARS:
    if "job_openings" not in monthly:
      if "job_openings_thousands" not in monthly:
        raise RuntimeError("Missing state JOLTS job openings for 2025.")
      monthly["job_openings"] = (
        1000 * pd.to_numeric(monthly.job_openings_thousands, errors="coerce")
      )
    monthly["job_openings"] = pd.to_numeric(monthly.job_openings, errors="coerce")
    monthly["unemployed_per_opening"] = (
      monthly.unemployed_persons / monthly.job_openings
    )
  return monthly


def workforce_latest_common_month(monthly, columns):
  """Select most recent month with complete valid data for all 51 areas."""
  complete = monthly.dropna(subset=columns).copy()
  complete = complete.loc[
    complete.civilian_population.gt(0)
    & complete.labor_force.gt(0)
    & complete.unemployed_persons.ge(0)
  ]
  if "job_openings" in columns:
    complete = complete.loc[complete.job_openings.gt(0)]
  dates = complete.groupby("Date").state.nunique()
  valid = dates.loc[dates.eq(len(DEMOGRAPHIC_STATE_FIPS))]
  if valid.empty:
    raise RuntimeError("No month contains complete observations for all 51 areas.")
  latest_date = valid.index.max()
  return complete.loc[complete.Date.eq(latest_date)].copy(), latest_date


def workforce_rank_label(rank, direction="highest"):
  """Human-readable ordinal ranking for manuscript text."""
  rank = int(rank)
  if rank == 1:
    return direction
  names = {
    2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
  }
  return f"{names.get(rank, str(rank) + 'th')}-{direction}"


def workforce_historical_summary(refresh=False, current_year_monthly=None):
  """Build 2021-2025 state averages using valid monthly BLS observations."""
  output = LABOR_MARKET_DIR / "bls_labor_monthly_2021_2025.csv"
  if output.exists() and not refresh:
    historic = pd.read_csv(
      output, dtype={"state_fips": "string"}, parse_dates=["Date"]
    )
  else:
    historic = pd.concat([
      current_year_monthly.copy()
      if year == WORKFORCE_JOLTS_YEAR and current_year_monthly is not None
      else workforce_monthly(year, refresh=refresh)
      for year in WORKFORCE_HISTORY_YEARS
    ], ignore_index=True)

  required_columns = [
    "state", "Date", "labor_force", "civilian_population",
    "unemployed_persons", "job_openings",
  ]
  if "job_openings" not in historic and "job_openings_thousands" in historic:
    historic["job_openings"] = (
      1000 * pd.to_numeric(historic.job_openings_thousands, errors="coerce")
    )
  missing = sorted(set(required_columns) - set(historic.columns))
  if missing:
    raise RuntimeError(f"Incomplete 2021-2025 BLS cache: {missing}")

  historic = historic.loc[historic.Date.dt.year.isin(WORKFORCE_HISTORY_YEARS)].copy()
  if historic.state.nunique() != len(DEMOGRAPHIC_STATE_FIPS):
    raise RuntimeError("Historical BLS data must contain 50 states and DC.")
  for column in required_columns[2:]:
    historic[column] = pd.to_numeric(historic[column], errors="coerce")
  historic = historic.sort_values(["state", "Date"]).reset_index(drop=True)
  historic.to_csv(output, index=False)

  summary_rows = []
  for state, group in historic.groupby("state"):
    lfpr = group.dropna(subset=["labor_force", "civilian_population"])
    unemployment = group.dropna(subset=["unemployed_persons", "labor_force"])
    openings = group.dropna(subset=["job_openings", "unemployed_persons"])
    lfpr_value = (
      lfpr.labor_force.sum() / lfpr.civilian_population.sum()
      if not lfpr.empty else np.nan
    )
    unemployment_value = (
      unemployment.unemployed_persons.sum() / unemployment.labor_force.sum()
      if not unemployment.empty else np.nan
    )
    openings_value = (
      openings.job_openings.sum() / openings.unemployed_persons.sum()
      if not openings.empty else np.nan
    )
    summary_rows.append({
      "state": state,
      "average_lfpr_2021_2025": lfpr_value,
      "average_lfpr_2021_2025_pct": 100 * lfpr_value,
      "average_unemployment_rate_2021_2025": unemployment_value,
      "average_unemployment_rate_2021_2025_pct": 100 * unemployment_value,
      "job_openings_per_unemployed_2021_2025": openings_value,
      "months_with_paired_openings_unemployment": len(openings),
    })

  summary = pd.DataFrame(summary_rows)
  for measure, rank_column, ascending in (
    ("average_lfpr_2021_2025", "average_lfpr_rank", False),
    ("average_unemployment_rate_2021_2025", "average_unemployment_rate_rank", True),
    ("job_openings_per_unemployed_2021_2025", "job_openings_per_unemployed_rank", False),
  ):
    summary[rank_column] = (
      summary[measure].rank(method="min", ascending=ascending).astype("Int64")
    )
  summary = summary.sort_values("state").reset_index(drop=True)
  summary.to_csv(
    LABOR_MARKET_DIR / "state_labor_market_summary_2021_2025.csv", index=False
  )
  summary.to_excel(
    LABOR_MARKET_DIR / "state_labor_market_summary_2021_2025.xlsx", index=False
  )
  return historic, summary


def workforce_2025_average_openings(openings):
  """Average the available 2025 monthly state job-opening levels."""
  data = openings.loc[openings.Date.dt.year.eq(2025)].copy()
  summary = (
    data.groupby("state", as_index=False)
    .agg(
      average_monthly_job_openings_2025=("job_openings", "mean"),
      months_with_job_openings_2025=("job_openings", "count"),
    )
    .sort_values("state").reset_index(drop=True)
  )
  if summary.state.nunique() != len(DEMOGRAPHIC_STATE_FIPS):
    raise RuntimeError("Expected job-opening observations for 50 states and DC.")
  if not summary.months_with_job_openings_2025.eq(12).all():
    print("WARNING: Not all states have 12 monthly 2025 JOLTS values.")
  summary.to_csv(
    LABOR_MARKET_DIR / "state_average_monthly_job_openings_2025.csv", index=False
  )
  data.loc[data.state.eq("North Dakota"), [
    "Date", "job_openings", "unemployed_persons"
  ]].to_csv(LABOR_MARKET_DIR / "nd_monthly_job_openings_2025.csv", index=False)
  return summary


def build_workforce_manuscript_stats(refresh=WORKFORCE_REFRESH):
  """Export manuscript statistics and historical/2026 audit series."""
  LABOR_MARKET_DIR.mkdir(parents=True, exist_ok=True)

  # Figure 5's state map CSV is NOW sourced exclusively from published BLS
  # Geographic Profile 2025, Table 14. Reject any stale CPS-created CSV.
  female = pd.read_csv(DEMOGRAPHIC_DIR / "figure_5_female_lfpr_state_2025.csv")
  if "female_lfpr_source" not in female.columns or not female[
    "female_lfpr_source"
  ].fillna("").astype(str).str.contains("Table 14", case=False, regex=False).all():
    raise RuntimeError("Figure 5 female LFPR file is not sourced from published BLS Table 14")
  female["female_lfpr_2025_pct"] = pd.to_numeric(
    female.female_lfpr_2025_pct, errors="raise"
  )
  female["rank_51"] = (
    female.female_lfpr_2025_pct.rank(method="min", ascending=False).astype(int)
  )
  female["rank_50"] = pd.NA
  states = female.state.ne("District of Columbia")
  female.loc[states, "rank_50"] = (
    female.loc[states, "female_lfpr_2025_pct"]
    .rank(method="min", ascending=False).astype(int)
  )
  female.to_csv(LABOR_MARKET_DIR / "female_lfpr_states_2025.csv", index=False)

  laus = workforce_monthly(WORKFORCE_YEAR, refresh=refresh)
  laus.to_csv(LABOR_MARKET_DIR / "bls_laus_2026_monthly.csv", index=False)
  latest_laus, laus_date = workforce_latest_common_month(
    laus, ["labor_force", "civilian_population", "unemployed_persons"]
  )
  latest_laus["lfpr_rank_51"] = (
    latest_laus.lfpr_pct.rank(method="min", ascending=False).astype(int)
  )
  latest_laus["unemployment_rank_51"] = (
    latest_laus.unemployment_rate_pct.rank(method="min", ascending=True).astype(int)
  )
  latest_laus["lfpr_rank_50"] = pd.NA
  state_mask = latest_laus.state.ne("District of Columbia")
  latest_laus.loc[state_mask, "lfpr_rank_50"] = (
    latest_laus.loc[state_mask, "lfpr_pct"]
    .rank(method="min", ascending=False).astype(int)
  )
  latest_laus.to_csv(LABOR_MARKET_DIR / "state_laus_latest_2026.csv", index=False)

  complete_laus = laus.dropna(subset=[
    "labor_force", "civilian_population", "unemployed_persons"
  ]).copy()
  complete_laus = complete_laus.loc[
    complete_laus.civilian_population.gt(0)
    & complete_laus.labor_force.gt(0)
    & complete_laus.unemployed_persons.ge(0)
  ]
  month_counts = complete_laus.groupby("Date").state.nunique()
  common_months = month_counts.loc[
    month_counts.eq(len(DEMOGRAPHIC_STATE_FIPS))
  ].index
  ytd = (
    complete_laus.loc[complete_laus.Date.isin(common_months)]
    .groupby("state", as_index=False)
    .agg(
      labor_force_sum=("labor_force", "sum"),
      civilian_population_sum=("civilian_population", "sum"),
      unemployed_persons_sum=("unemployed_persons", "sum"),
      months=("Date", "nunique"),
    )
  )
  ytd["lfpr_pct"] = 100 * ytd.labor_force_sum / ytd.civilian_population_sum
  ytd["unemployment_rate_pct"] = (
    100 * ytd.unemployed_persons_sum / ytd.labor_force_sum
  )
  ytd["lfpr_rank_51"] = (
    ytd.lfpr_pct.rank(method="min", ascending=False).astype(int)
  )
  ytd["unemployment_rank_51"] = (
    ytd.unemployment_rate_pct.rank(method="min", ascending=True).astype(int)
  )
  ytd.to_csv(LABOR_MARKET_DIR / "state_laus_2026_year_to_date.csv", index=False)

  openings = workforce_monthly(WORKFORCE_JOLTS_YEAR, refresh=refresh)
  annual_openings = workforce_2025_average_openings(openings)
  historic, historical_summary = workforce_historical_summary(
    refresh=refresh, current_year_monthly=openings
  )
  latest_openings, openings_date = workforce_latest_common_month(
    openings, [
      "civilian_population", "labor_force", "unemployed_persons", "job_openings"
    ]
  )
  latest_openings["unemployed_per_opening_rank_51"] = (
    latest_openings.unemployed_per_opening
    .rank(method="min", ascending=True).astype(int)
  )
  latest_openings.to_csv(
    LABOR_MARKET_DIR / "state_openings_latest_2025.csv", index=False
  )

  nd_laus = latest_laus.loc[latest_laus.state.eq("North Dakota")].iloc[0]
  nd_female = female.loc[female.state.eq("North Dakota")].iloc[0]
  ne_female = female.loc[female.state.eq("Nebraska")].iloc[0]
  dc_female = female.loc[female.state.eq("District of Columbia")].iloc[0]
  sd_female = female.loc[female.state.eq("South Dakota")].iloc[0]
  mn_female = female.loc[female.state.eq("Minnesota")].iloc[0]
  nd_openings = latest_openings.loc[latest_openings.state.eq("North Dakota")].iloc[0]
  nd_average_openings = annual_openings.loc[annual_openings.state.eq("North Dakota")].iloc[0]
  nd_historical = historical_summary.loc[historical_summary.state.eq("North Dakota")].iloc[0]
  nd_ytd = ytd.loc[ytd.state.eq("North Dakota")].iloc[0]
  best_openings = latest_openings.sort_values(["unemployed_per_opening", "state"]).iloc[0]

  dc_laus = latest_laus.loc[latest_laus.state.eq("District of Columbia")].iloc[0]
  dc_note = (
    " Washington, D.C. reported a higher rate."
    if dc_laus.lfpr_pct > nd_laus.lfpr_pct else ""
  )
  lines = [
    "# Updated workforce paragraphs (nonseasonally adjusted BLS data)", "",
    (
      "North Dakota has one of the highest labor force participation rates "
      "in the United States. In "
      f"{laus_date:%B %Y}, the state's labor force participation rate was "
      f"{nd_laus.lfpr_pct:.1f}%, "
      f"the {workforce_rank_label(nd_laus.lfpr_rank_50)} among the 50 states."
      f"{dc_note} Its unemployment rate was {nd_laus.unemployment_rate_pct:.1f}%, "
      f"the {workforce_rank_label(nd_laus.unemployment_rank_51, 'lowest')} "
      "among the 50 states "
      "and Washington, D.C. Despite these favorable figures, North Dakota "
      "continues to face challenges filling job openings. In 2025, employers "
      f"reported an average of approximately {nd_average_openings.average_monthly_job_openings_2025:,.0f} "
      "job openings per month. "
      f"In {openings_date:%B %Y}, North Dakota had "
      f"{nd_openings.unemployed_per_opening:.2f} unemployed persons per "
      f"job opening, ranking {int(nd_openings.unemployed_per_opening_rank_51)} "
      "lowest among the 50 states and Washington, D.C. "
      "In his 2024 State of the State Address, former North Dakota Governor "
      "Doug Burgum described workforce as 'our biggest challenge' "
      "([source](https://www.governor.nd.gov/sites/www/files/documents/"
      "2024%20SOTS%20transcript%20for%20website.pdf))."
    ), "",
    (
      f"In 2025, North Dakota had the {workforce_rank_label(nd_female.rank_50)} "
      "female labor force participation rate among U.S. states at "
      f"{nd_female.female_lfpr_2025_pct:.1f}%, behind Nebraska "
      f"({ne_female.female_lfpr_2025_pct:.1f}%). Including Washington, D.C. "
      f"({dc_female.female_lfpr_2025_pct:.1f}%), North Dakota ranked "
      f"{workforce_rank_label(nd_female.rank_51).replace('-highest', '')} "
      "overall, as shown in Figure 5. "
      "North Dakota's rate also exceeded neighboring South Dakota "
      f"({sd_female.female_lfpr_2025_pct:.1f}%) and Minnesota "
      f"({mn_female.female_lfpr_2025_pct:.1f}%)."
    ), "",
    "Sources: BLS LAUS (2026), state JOLTS and LAUS (2025),",
    "and published BLS Geographic Profile 2025, Table 14 (women).",
    f"BLS female LFPR: {BLS_FEMALE_LFPR_SOURCE}",
    "Published BLS 2025 Table 14 excludes October 2025; no CPS microdata were recalculated.",
    "2025 job openings are not combined with 2026 unemployment counts.", "",
    "## Historical context (2021–2025 averages)",
    f"North Dakota LFPR: {nd_historical.average_lfpr_2021_2025_pct:.2f}% "
    f"(rank {int(nd_historical.average_lfpr_rank)} of 51).",
    f"North Dakota unemployment: {nd_historical.average_unemployment_rate_2021_2025_pct:.2f}% "
    f"(rank {int(nd_historical.average_unemployment_rate_rank)} lowest of 51).",
    f"North Dakota job openings per unemployed: "
    f"{nd_historical.job_openings_per_unemployed_2021_2025:.3f} "
    f"(rank {int(nd_historical.job_openings_per_unemployed_rank)} of 51).", "",
    "## 2026 year to date (common complete months)",
    f"North Dakota LFPR: {nd_ytd.lfpr_pct:.2f}%; "
    f"unemployment: {nd_ytd.unemployment_rate_pct:.2f}% "
    f"(based on {int(nd_ytd.months)} months).",
  ]
  text_file = LABOR_MARKET_DIR / "updated_workforce_paragraphs.md"
  text_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

  print("\nWORKFORCE STATISTICS FOR MANUSCRIPT")
  print(
    f"2025 ND average monthly job openings: "
    f"{nd_average_openings.average_monthly_job_openings_2025:,.0f} "
    f"({int(nd_average_openings.months_with_job_openings_2025)} months)"
  )
  print(
    f"2021-2025 averages: LFPR {nd_historical.average_lfpr_2021_2025_pct:.2f}%, "
    f"unemployment {nd_historical.average_unemployment_rate_2021_2025_pct:.2f}%, "
    f"job openings per unemployed "
    f"{nd_historical.job_openings_per_unemployed_2021_2025:.3f}"
  )
  print(
    f"2026 YTD ({int(nd_ytd.months)} common months): "
    f"LFPR {nd_ytd.lfpr_pct:.2f}%, "
    f"unemployment {nd_ytd.unemployment_rate_pct:.2f}%"
  )
  print(
    f"{laus_date:%B %Y}: ND LFPR {nd_laus.lfpr_pct:.2f}% "
    f"(rank {int(nd_laus.lfpr_rank_50)} of 50 states; "
    f"{int(nd_laus.lfpr_rank_51)} of 51 including DC); "
    f"unemployment {nd_laus.unemployment_rate_pct:.2f}% "
    f"(rank {int(nd_laus.unemployment_rank_51)} lowest of 51)."
  )
  print(
    f"2025 published BLS ND female LFPR {nd_female.female_lfpr_2025_pct:.2f}% "
    f"(rank {int(nd_female.rank_50)} of 50 states; "
    f"{int(nd_female.rank_51)} of 51 including DC)."
  )
  print(
    f"{openings_date:%B %Y} unemployed per opening: "
    f"ND {nd_openings.unemployed_per_opening:.3f} "
    f"(rank {int(nd_openings.unemployed_per_opening_rank_51)} of 51), "
    f"{best_openings.state} {best_openings.unemployed_per_opening:.3f}."
  )
  print(f"Saved manuscript outputs to: {LABOR_MARKET_DIR}")

  return {
    "female_lfpr_2025": female,
    "latest_2026_laus": latest_laus,
    "monthly_2026_laus": laus,
    "year_to_date_2026_laus": ytd,
    "monthly_2025_openings": openings,
    "annual_2025_openings": annual_openings,
    "latest_2025_openings": latest_openings,
    "historical_monthly_2021_2025": historic,
    "historical_summary_2021_2025": historical_summary,
  }


build_demographic_figures(
  figures=DEMOGRAPHIC_FIGURES,
  refresh=DEMOGRAPHIC_REFRESH,
  lfpr_csv=BLS_FEMALE_LFPR_CSV,
)

workforce_outputs = build_workforce_manuscript_stats(
  refresh=WORKFORCE_REFRESH
)


# %% LATEST MEDIAN HOUSEHOLD INCOME: NORTH DAKOTA VS SOUTH DAKOTA ----------
# Census ACS 1-year estimates, not BLS. The 2025 one-year ACS was released
# October 29, 2026; table B19013 reports income in 2025 dollars.
# This ACS query concerns household income only, NOT the number of children.

ACS_INCOME_YEAR = 2025
ACS_INCOME_URL = f"https://api.census.gov/data/{ACS_INCOME_YEAR}/acs/acs1"
ACS_INCOME_CSV = LABOR_MARKET_DIR / f"acs_median_household_income_{ACS_INCOME_YEAR}.csv"
ACS_INCOME_COMPARISON_CSV = (
  LABOR_MARKET_DIR / f"nd_sd_median_household_income_{ACS_INCOME_YEAR}.csv"
)


def build_household_income_comparison(refresh=False):
  """Download same-year ACS median household incomes, compare ND and SD."""
  LABOR_MARKET_DIR.mkdir(parents=True, exist_ok=True)

  if ACS_INCOME_CSV.exists() and not refresh:
    income = pd.read_csv(ACS_INCOME_CSV, dtype={"state_fips": "string"})
    needed = {"state", "median_household_income", "median_household_income_moe"}
    if not needed.issubset(income.columns):
      raise ValueError("Cached ACS income file has unexpected columns.")
  else:
    demographic_load_env()
    params = {
      "get": "NAME,B19013_001E,B19013_001M",
      "for": "state:*",
    }
    api_key = os.environ.get("CENSUS_API_KEY", "").strip()
    if api_key:
      params["key"] = api_key

    with demographic_request(ACS_INCOME_URL, params=params) as response:
      payload = response.json()
    if not isinstance(payload, list) or len(payload) < 3:
      raise ValueError("Census ACS median household income response was empty.")

    income = pd.DataFrame(payload[1:], columns=payload[0]).rename(columns={
      "NAME": "state",
      "state": "state_fips",
      "B19013_001E": "median_household_income",
      "B19013_001M": "median_household_income_moe",
    })
    income["state_fips"] = income.state_fips.astype("string").str.zfill(2)
    for col in ("median_household_income", "median_household_income_moe"):
      income[col] = pd.to_numeric(income[col], errors="coerce")

    income = income.loc[
      income.state.isin(("North Dakota", "South Dakota"))
    ].copy()
    if len(income) != 2 or income.state.nunique() != 2:
      raise ValueError("Expected one 2025 ACS row each for North and South Dakota.")
    if income.median_household_income.isna().any():
      raise ValueError("Missing ACS median household income estimate.")
    if (income.median_household_income <= 0).any():
      raise ValueError("ACS median household incomes must be positive.")

    income["year"] = ACS_INCOME_YEAR
    income["income_unit"] = f"{ACS_INCOME_YEAR} inflation-adjusted dollars"
    income["source"] = "Census ACS 1-year, B19013"
    income = income.sort_values("state").reset_index(drop=True)
    income.to_csv(ACS_INCOME_CSV, index=False)

  nd_value = float(income.loc[
    income.state.eq("North Dakota"), "median_household_income"
  ].iloc[0])
  sd_value = float(income.loc[
    income.state.eq("South Dakota"), "median_household_income"
  ].iloc[0])

  difference = sd_value - nd_value
  difference_pct = 100 * difference / nd_value
  if difference_pct < 0:
    comparison_text = f"{abs(difference_pct):.1f}% lower than North Dakota"
  elif difference_pct > 0:
    comparison_text = f"{difference_pct:.1f}% higher than North Dakota"
  else:
    comparison_text = "equal to North Dakota"

  summary = pd.DataFrame([{
    "year": ACS_INCOME_YEAR,
    "source": "Census ACS 1-year B19013",
    "north_dakota_median_income": nd_value,
    "south_dakota_median_income": sd_value,
    "south_dakota_minus_north_dakota_dollars": difference,
    "south_dakota_vs_north_dakota_percent": difference_pct,
    "difference_denominator": "North Dakota",
    "comparison": comparison_text,
  }])
  summary.to_csv(ACS_INCOME_COMPARISON_CSV, index=False)

  sentence = (
    f"In {ACS_INCOME_YEAR}, South Dakota's median household income "
    f"(${sd_value:,.0f}) was {comparison_text} "
    f"(${nd_value:,.0f}) (U.S. Census Bureau, {ACS_INCOME_YEAR} ACS 1-year)."
  )
  notes = [
    "# North Dakota vs. South Dakota median household income", "",
    sentence, "", f"Source: {ACS_INCOME_URL}",
    "ACS detailed table B19013, estimate and 90% margin of error.",
    "The percent difference is computed relative to North Dakota.",
    "The ACS estimates and their sampling errors should not be conflated",
    "with BLS wage data or CPS ASEC median-income estimates.",
    "A point-estimate difference is not necessarily statistically significant.",
  ]
  (LABOR_MARKET_DIR / "nd_sd_household_income_manuscript.md").write_text(
    "\n".join(notes) + "\n", encoding="utf-8"
  )

  print("\n2025 ACS MEDIAN HOUSEHOLD INCOME — NORTH DAKOTA VS SOUTH DAKOTA")
  print(income[[
    "state", "median_household_income", "median_household_income_moe"
  ]].to_string(index=False))
  print(f"\n{sentence}")
  print(f"Saved ACS median income comparison to: {LABOR_MARKET_DIR}")
  return income, summary


acs_household_income, acs_household_income_comparison = (
  build_household_income_comparison()
)


# %% CHILD CARE AWARE PRICES ------------------------------------------------
# https://info.childcareaware.org/price-and-supply-2025
# year state inf_center tod_center 4yo_center
# 2025 ND    12_281     11_561     10_855
# 2025 SD     8_859                  8_282


# %% ND VS. SD: CHILDCARE PRICES AND TEACHER EDUCATION ----------------------
# 2023 regulations, 2025 prices. Original OLS models.
SD_REFERENCE_COST = {"infant": 8859, "four_year_old": 8282}

nd = panel.loc[
  panel.state.eq("North Dakota") & panel.year.eq(2023)
].iloc[0]
sd = panel.loc[
  panel.state.eq("South Dakota") & panel.year.eq(2023)
].iloc[0]

rows = []
for age in AGES:
  model = models[age, "original"]

  nd_price = ND_REFERENCE_COST[age]
  sd_price = SD_REFERENCE_COST[age]

  delta_ed = float(sd.ed_t - nd.ed_t)
  log_change = model.params["ed_t"] * delta_ed

  se = abs(delta_ed) * model.bse["ed_t"]
  critical = t.ppf(.975, df=model_states[age, "original"] - 1)

  simulated = nd_price * np.exp(log_change)
  ci_low = nd_price * (1 - np.exp(log_change + critical * se))
  ci_high = nd_price * (1 - np.exp(log_change - critical * se))

  rows.append({
    "Age": age,
    "ND price": nd_price,
    "SD price": sd_price,
    "Observed gap ($)": nd_price - sd_price,
    "SD cheaper (%)": 100 * (nd_price - sd_price) / nd_price,
    "ND education": nd.ed_t,
    "SD education": sd.ed_t,
    "Simulated price": simulated,
    "Education savings ($)": nd_price - simulated,
    "Education savings (%)": 100 * (1 - np.exp(log_change)),
    "95% CI low ($)": ci_low,
    "95% CI high ($)": ci_high,
    "Education p-value": model.pvalues["ed_t"],
  })

comparison = pd.DataFrame(rows)
print("\nND VS. SD — PRICE AND TEACHER EDUCATION COMPARISON")
print(comparison.round(2).to_string(index=False))
comparison.to_csv(OUTPUT_DIR / "nd_sd_education_comparison.csv", index=False)

