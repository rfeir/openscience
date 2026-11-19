"""Extract Child Care Aware's center-based childcare costs for 2010–2024.

Requires pandas and Poppler (pdftotext).
"""

from pathlib import Path
from urllib.request import Request, urlopen
from html import unescape
import re
import subprocess

import pandas as pd


# %% SETTINGS ---------------------------------------------------------------
# In a script, use the script directory. In an IPython cell, use the
# current working directory (start IPython from cost_left_hand_variable/).
DATA_DIR = (Path(__file__).resolve().parent
            if "__file__" in globals() else Path.cwd())
PDF_DIR = DATA_DIR / "sources/cca_pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
  2010: "https://info.childcareaware.org/hubfs/COC_Appendices_2010_2011.pdf",
  2011: "https://info.childcareaware.org/hubfs/COC_Appendices_2011_2012.pdf",
  2012: "https://info.childcareaware.org/hubfs/COC_Appendices_2012_2013.pdf",
  2013: "https://info.childcareaware.org/hubfs/COC_Appendices_2013_2014.pdf",
  2014: "https://info.childcareaware.org/hubfs/COC_Appendices_2014_2015.pdf",
  2015: "https://info.childcareaware.org/hubfs/COC_Appendices_2015_2016.pdf",
  2016: "https://info.childcareaware.org/hubfs/COC_Appendices_2016_2017.pdf",
  2017: "https://info.childcareaware.org/hubfs/COC_Appendices_2017_2018.pdf",
  2018: (
    "https://cdn2.hubspot.net/hubfs/3957809/2019%20Price%20of%20Care%20"
    "State%20Sheets/Final-TheUSandtheHighPriceofChildCare-Appendices.pdf"
  ),
  2019: "https://www.childcareaware.org/wp-content/uploads/2020/09/PUTP-Appendices-FINAL-9-18-20.pdf",
  2020: "https://info.childcareaware.org/hubfs/Demanding%20Change%20Appendices.pdf",
  2021: "https://info.childcareaware.org/hubfs/Child%20Care%20Affordability%20Analysis%202021.pdf",
  2022: "https://info.childcareaware.org/hubfs/2022_CC_Afford_Analysis.pdf",
  2023: "https://info.childcareaware.org/hubfs/2023_Affordability_Analysis.pdf",
  2024: "https://info.childcareaware.org/hubfs/Affordability_Analysis_Updated_2024.pdf",
}

COLUMNS = [
  "infant", "toddler", "four_year_old", "before_after_school",
  "full_time_summer", "part_time_summer", "school_age_12_months",
  "school_age_9_months_plus_summer",
]

SCHEMA = {year: ["infant", "four_year_old"] for year in range(2010, 2015)}
SCHEMA[2015] = [
  "infant", "four_year_old", "before_after_school",
  "school_age_12_months", "school_age_9_months_plus_summer"
]
SCHEMA.update({
  year: [
    "infant", "toddler", "four_year_old", "before_after_school",
    "full_time_summer", "part_time_summer"
  ]
  for year in range(2016, 2019)
})
SCHEMA.update({
  year: [
    "infant", "toddler", "four_year_old", "before_after_school",
    "full_time_summer"
  ]
  for year in range(2019, 2025)
})

STATES = """Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|
District of Columbia|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|
Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|
Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|
North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Puerto Rico|
Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|
Washington|West Virginia|Wisconsin|Wyoming""".replace("\n", "").split("|")

STATES = [s.strip() for s in STATES]
REQUIRED_STATES = set(STATES) - {"Puerto Rico"}

STATE_RE = re.compile(
  r"^\s*("
  + "|".join(map(re.escape, sorted(STATES, key=len, reverse=True)))
  + r")(?![A-Za-z])([\s*†‡+^#]*)(.*)$"
)

PRICE_RE = re.compile(
  r"\$\s*\d[\d,]*(?:\.\d+)?|\b(?:NR|N/A|NA)\b|(?<![A-Za-z$])\d[\d,]*\*?",
  re.I
)

CENTER_TITLE = r"Average Annual (?:Cost|Price) of Full-Time Center-Based Child Care by State"
# Used only as the boundary at which Table I ends; no family data are extracted.
NEXT_TABLE_TITLE = r"Average Annual (?:Cost|Price) of Full-Time Family(?:-Based)? Child Care(?: \(FCC\))? by State"


# %% PARSE VALUES -----------------------------------------------------------
def to_number(token):
  token = token.replace("$", "").replace(",", "").replace("*", "").strip()
  return None if token.upper() in ("NR", "NA", "N/A") else round(float(token))


def parse_line(line, year):
  match = STATE_RE.match(line)
  if not match:
    return None

  state, marker, remainder = match.groups()
  prices = PRICE_RE.findall(remainder)
  n = len(SCHEMA[year])

  if len(prices) < n:
    return None

  return {
    "state": state,
    "cca_imp_flag": int(year >= 2016 and "*" in marker),
    **{
      col: to_number(price)
      for col, price in zip(SCHEMA[year], prices[:n])
    },
  }


# %% DOWNLOAD / EXTRACT PDF -------------------------------------------------
def pdf_pages(year):
  path = PDF_DIR / f"cca_{year}.pdf"

  if not path.exists():
    print(f"  Downloading {year} report")
    request = Request(SOURCES[year], headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(request, timeout=90) as response:
      content = response.read()

    if not content.startswith(b"%PDF-"):
      raise ValueError(f"{year}: downloaded file is not a PDF")

    path.write_bytes(content)

  result = subprocess.run(
    ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
    capture_output=True, text=True, errors="replace", check=True
  )

  return path, result.stdout.split("\f")


# %% RECOVER SPLIT PDF ROW --------------------------------------------------
def recover_position(path, page_number, state, year):
  """Recover a state row split across lines by Poppler, notably 2017 D.C."""

  result = subprocess.run(
    [
      "pdftotext", "-f", str(page_number + 1),
      "-l", str(page_number + 1),
      "-bbox-layout", str(path), "-"
    ],
    capture_output=True, text=True, errors="replace", check=True
  )

  words = []

  for match in re.finditer(r"<word\b([^>]*)>(.*?)</word>", result.stdout, re.S):
    attr = dict(
      re.findall(r'(xMin|xMax|yMin|yMax)="([^"]+)"', match.group(1))
    )

    if len(attr) == 4:
      words.append((
        unescape(match.group(2)).strip(),
        float(attr["xMin"]),
        float(attr["xMax"]),
        (float(attr["yMin"]) + float(attr["yMax"])) / 2
      ))

  for text, x1, x2, y in words:
    if text.strip("*") != state.split()[0]:
      continue

    line = sorted(
      (w for w in words if abs(w[3] - y) <= 4 and w[1] >= x1 - 1),
      key=lambda w: w[1]
    )

    name = state.split()

    if [w[0].strip("*") for w in line[:len(name)]] != name:
      continue

    rest = line[len(name):]

    prices = [
      w[0] for w in rest
      if w[1] > line[len(name) - 1][2] + 1 and PRICE_RE.fullmatch(w[0])
    ]

    if len(prices) != len(SCHEMA[year]):
      continue

    flag = any("*" in w[0] for w in line[:len(name)])
    flag |= any(w[0] == "*" for w in rest[:1])

    return {
      "state": state,
      "cca_imp_flag": int(year >= 2016 and flag),
      **{col: to_number(p) for col, p in zip(SCHEMA[year], prices)},
    }

  return None


# %% SCRAPE TABLE -----------------------------------------------------------
def scrape_table(pages, path, year):
  start = None

  for i, page in enumerate(pages):
    compact = " ".join(page.split())

    if (
      re.search(CENTER_TITLE, compact, re.I)
      and "TABLE OF CONTENTS" not in page.upper()
    ):
      matching = sum(
        parse_line(line, year) is not None
        for line in page.splitlines()
      )

      if matching >= 5:
        start = i
        break

  if start is None:
    raise ValueError(f"{year} center: table not found")

  rows = {}
  table_pages = []

  for i in range(start, min(start + 4, len(pages))):
    if (
      i > start
      and re.search(NEXT_TABLE_TITLE, " ".join(pages[i].split()), re.I)
    ):
      break

    table_pages.append(i)

    for line in pages[i].splitlines():
      row = parse_line(line, year)

      if row:
        state = row.pop("state")

        if state in rows:
          raise ValueError(f"{year} center: duplicate {state}")

        rows[state] = {**row, "source_page": i + 1}

    if "Wyoming" in rows:
      break

  # Specific fallback for the problematic 2017 center row.
  if year == 2017 and "District of Columbia" not in rows:
    for i in table_pages:
      row = recover_position(path, i, "District of Columbia", year)

      if row:
        rows[row.pop("state")] = {**row, "source_page": i + 1}
        print("  Recovered 2017 center District of Columbia")
        break

  # Missing source or unextractable row: retain state with null costs.
  for state in sorted(REQUIRED_STATES - rows.keys()):
    print(f"  WARNING: {year} center: {state} not extracted; costs set to null")
    rows[state] = {"cca_imp_flag": None, "source_page": None}

  present = int(
    any(row.get("cca_imp_flag") == 1 for row in rows.values())
  )

  return [
    {
      "year": year,
      "state": state,
      "cca_imp_pres": present,
      **{col: None for col in COLUMNS},
      **data,
      "source_url": SOURCES[year],
    }
    for state, data in rows.items()
  ]


# %% SCRAPE ALL YEARS -------------------------------------------------------
center_rows = []

for year in SOURCES:
  print(f"{year}: extracting center-based childcare prices (Table I)")
  path, pages = pdf_pages(year)
  rows = scrape_table(pages, path, year)
  center_rows.extend(rows)
  print(f"  center: {len(rows)} jurisdictions")


# %% SAVE EXTRACTED DATA ----------------------------------------------------
data = pd.DataFrame(center_rows).sort_values(["year", "state"])
output = DATA_DIR / "center_cost_python.csv"
data.to_csv(output, index=False)
print(f"Saved {output} ({len(data)} rows)")

