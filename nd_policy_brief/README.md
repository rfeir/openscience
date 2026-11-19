# North Dakota childcare regulations and prices: reproducibility materials

This folder contains the datasets, regulatory coding documentation, source records, Python scripts, and outputs used for the North Dakota childcare policy brief *Expanding Access and Affordability*. The primary analysis examines the relationship between state childcare regulations and annual **center-based** childcare prices using state-year observations for **2013, 2018, and 2023**.

The published regression analysis fits **two ordinary least squares (OLS) models**: one for infant childcare prices and one for four-year-old childcare prices. Both use the **original prices reported by Child Care Aware of America**. The research does not perform additional interpolation or imputation; published source prices may already incorporate Child Care Aware's own estimation practices. Standard errors are clustered by state. Regression prices and monetary controls are expressed in **2025 dollars**.

## Repository layout

All paths below are relative to this `nd_policy_brief/` directory.

```text
nd_policy_brief/
├── README.md
├── requirements.txt
├── center_regulations/
│   ├── center_reg_vars.csv
│   ├── legal_sources.xlsx
│   ├── methodology/
│   │   ├── education_rules.md
│   │   ├── scr_mgs_rules.md
│   │   └── ai_assistance_manual_editing_note.md
│   ├── historical_sources/
│   └── revision_references/
├── control_variables/
│   ├── control_generator.py
│   ├── control_variables_state_year_2013_2018_2023.csv
│   ├── state_female_lfpr_2025_bls.csv
│   └── sources/
│       ├── bls_female_lfpr/
│       └── census_pep/
├── cost_left_hand_variable/
│   ├── scraper.py
│   ├── center_cost_python.csv
│   └── sources/
│       ├── cca_pdfs/
│       └── cpi_u_monthly.csv
├── center_ols/
│   ├── nd_center_ols.py
│   ├── .cache/                 # generated; not committed
│   └── output/
│       ├── figures/           # generated PNG plots
│       ├── results/           # regression tables, simulations, map data
│       └── summaries/         # ND–SD comparison, manuscript notes
└── manually_produced_fig/
    └── female_labor_force.png
```

The `center_regulations/` data are **curated research inputs** supported by historical legal records and coding rules; there is no automated regulations-generator script in the execution sequence.

## Requirements

- **Python 3.10 or newer**, with packages from [`requirements.txt`](requirements.txt). Python 3.11 or 3.12 is recommended; exact-version reproduction has not been independently tested.
- **Poppler's `pdftotext` executable** for the Child Care Aware PDF extraction script. This is a system-level program, not a Python package.
- Internet access to request Census ACS, FRED, and BLS data when corresponding cached/source data are not available.
- API credentials as described below to rebuild controls and run the full workforce analysis.

From `nd_policy_brief/`, create a virtual environment and install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Poppler separately, for example on Arch Linux:

```bash
sudo pacman -S poppler
pdftotext -v
```

On Debian/Ubuntu use `sudo apt install poppler-utils`; on macOS with Homebrew use `brew install poppler`.

### API keys and environment variables

The scripts use these environment variables:

| Variable | Provider | When needed |
| --- | --- | --- |
| `FRED_API_KEY` | Federal Reserve Bank of St. Louis (FRED) | **Required** to rebuild minimum-wage and CPI controls using `control_generator.py`. |
| `BLS_API_KEY` | U.S. Bureau of Labor Statistics | **Required** for fresh BLS time-series requests in the complete `nd_center_ols.py` workforce analysis. Previously cached valid responses may avoid some requests. |
| `CENSUS_API_KEY` | U.S. Census Bureau | **Recommended** for ACS 1-year household-income API requests by both the controls generator and the OLS script. Some Census endpoints permit unkeyed requests. |

Obtain credentials directly from the relevant providers: [FRED API](https://fred.stlouisfed.org/docs/api/fred/), [BLS API](https://www.bls.gov/developers/), and [Census API](https://www.census.gov/data/developers.html).

For local use, put them in an **untracked** file at `nd_policy_brief/.env`:

```dotenv
FRED_API_KEY=your_fred_api_key
BLS_API_KEY=your_bls_api_key
CENSUS_API_KEY=your_census_api_key
```

Alternatively, export those environment variables in the shell before running the scripts. The controls generator reads simple `.env` entries itself; the OLS script loads them using `python-dotenv`. **Never commit actual `.env` files, API keys, or API response URLs containing keys.** Confirm the repository's `.gitignore` excludes `.env`, `.venv/`, and `.cache/`. No API keys should be necessary merely to read the committed output tables.

## Reproducing the research

Run all commands **from the `nd_policy_brief/` directory** after activating the virtual environment. The scraper and controls generator may be run in either order; both must finish before running the OLS analysis.

### 1. Extract reported center-based childcare prices

```bash
python cost_left_hand_variable/scraper.py
```

- Reads archived Child Care Aware reports for **2010–2024** from `cost_left_hand_variable/sources/cca_pdfs/` (downloaded as needed).
- Uses `pdftotext` to extract Table I, the annual **center-based** childcare price table.
- Produces `cost_left_hand_variable/center_cost_python.csv` with reported nominal prices, source URLs/page references, and published-source flags.
- Does not extract family-childcare prices or create cleaned/interpolated price series.

### 2. Rebuild the state-year control variables

```bash
python control_variables/control_generator.py
```

This combines:

- Census Population Estimates Program (PEP) population totals and population ages 0–4 from `control_variables/sources/census_pep/`.
- BLS *Geographic Profile*, Table 14 (Women), for female labor-force participation in 2013, 2018, and 2023, using workbooks in `control_variables/sources/bls_female_lfpr/`.
- Census ACS 1-year median household income (B19013) via the Census API.
- Federal/state minimum wages and CPI-U inflation adjustments from FRED.

Expected output:

```text
control_variables/control_variables_state_year_2013_2018_2023.csv
```

The resulting file should contain **153 state-year observations** (50 states plus Washington, D.C., over three years). Female labor-force participation and the under-five population share are fractions in the 0–1 range. Household income and minimum wage controls are inflation-adjusted to 2025 dollars.

**Version check before publishing:** In `control_variables/control_generator.py`, verify that `OUTPUT_FILE` points to the exact filename above. Some prior script versions used a different suffix (`_bls_rebuilt.csv`), whereas `nd_center_ols.py` expects this shorter filename. Make sure the two scripts agree before claiming that a clean run is reproducible.

### 3. Use the manually verified regulatory panel

The third required input is already supplied:

```text
center_regulations/center_reg_vars.csv
```

Its staff-child ratio (SCR), maximum group size (MGS), and education requirement codes are documented in:

- [`center_regulations/methodology/scr_mgs_rules.md`](center_regulations/methodology/scr_mgs_rules.md)
- [`center_regulations/methodology/education_rules.md`](center_regulations/methodology/education_rules.md)
- [`center_regulations/methodology/ai_assistance_manual_editing_note.md`](center_regulations/methodology/ai_assistance_manual_editing_note.md)

Historical source documents and revision records are in `center_regulations/historical_sources/` and `center_regulations/revision_references/`; corresponding source references are cataloged in `center_regulations/legal_sources.xlsx`. These files support auditing the curated coding rather than executing an additional automated generation step.

### 4. Fit the two models and generate results

```bash
python center_ols/nd_center_ols.py
```

The analysis script:

1. Loads the reported childcare prices, regulatory codes, and generated state-year controls; joins by `state` and `year`.
2. Converts nominal childcare prices to **2025 dollars** using CPI-U (matching the controls generator's 2025-dollar convention).
3. Fits **two** OLS models: infant prices and four-year-old prices, each with state-clustered standard errors and 2013 as the omitted year.
4. Produces model diagnostics, correlations, North Dakota regulatory counterfactuals, demographic plots, 2025 BLS female labor-force map data, workforce summaries, and an ND–SD comparison.

Its figures and tables appear in:

```text
center_ols/output/figures/       # PNGs
center_ols/output/results/       # CSVs, including regression tables
center_ols/output/summaries/     # manuscript_updates.md and comparisons
```

Downloaded data and API caches are kept outside those output directories, under `center_ols/.cache/`. They are not part of the publication outputs. Some manuscript statistics use **2025–2026** sources in addition to the **2013/2018/2023** regression panel; the manuscript section should be checked against the date and vintage of the public data when reproducing the original findings.

## Manually produced figure: female labor-force participation map

The map image at [`manually_produced_fig/female_labor_force.png`](manually_produced_fig/female_labor_force.png) was **assembled manually**, rather than generated by `center_ols/nd_center_ols.py`. Its numerical data are provided in the repository:

- **Map input CSV:** [`center_ols/output/results/figure_5_female_lfpr_state_2025.csv`](center_ols/output/results/figure_5_female_lfpr_state_2025.csv). The OLS script generates this from the published BLS state-level *Women* series.
- **Derived 2025 BLS data file:** [`control_variables/state_female_lfpr_2025_bls.csv`](control_variables/state_female_lfpr_2025_bls.csv), which the OLS script can reuse if present.
- **Underlying published source:** [`control_variables/sources/bls_female_lfpr/ptable14full2025.xlsx`](control_variables/sources/bls_female_lfpr/ptable14full2025.xlsx), BLS *Geographic Profile of Employment and Unemployment*, 2025, Table 14, Women (https://www.bls.gov/opub/geographic-profile/2025/home.htm).

To reproduce the **values**, run Step 4 and use the generated Figure 5 CSV. Recreating the **exact map artwork and layout** requires manually reproducing the design in the original figure-authoring workflow; there is no automated map-rendering step in the repository. This is distinct from the PNG charts generated automatically in `center_ols/output/figures/`.

## Interpretation and provenance notes

- Reported annual childcare prices represent prices paid by families for center-based care, **not** provider operating costs.
- Original reported prices are used as supplied by Child Care Aware; this workflow does not replace reported values using researcher-generated interpolation or multiple imputation.
- SCR is coded as children permitted per qualifying staff member (for example, a 1:7 ratio is represented by 7). The MGS/SCR covariate is maximum permitted group size divided by that coded SCR.
- The demographic, regulatory, and cost sources are independent upstream data streams. The regression merge occurs in `center_ols/nd_center_ols.py`.
- The manuscript's **2025 North Dakota price benchmarks** used for simulations are defined separately from the 2013/2018/2023 regression observations. They are not estimated by the OLS regression.
- The 2025 BLS Table 14 rates and the 2025-dollar CPI convention reflect the published/source-period treatment described in the scripts; API revisions and later source vintages may affect reproducibility.

## Citation and reuse

When using these materials, cite the associated North Dakota childcare policy brief, Child Care Aware of America price data, the BLS/Census/FRED series, and the relevant historical childcare regulations. The archived laws, reports, figures, and other third-party data retain their original authorship and any applicable terms of reuse; the code license should not be interpreted as relicensing third-party source materials.
