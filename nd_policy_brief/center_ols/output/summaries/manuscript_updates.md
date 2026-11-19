# Original-price regression numbers for Expanding Access and Affordability

All amounts are in 2025 dollars. The regression uses 2013, 2018,
and 2023 state-year observations and North Dakota's 2023 policy anchors.
North Dakota reference costs are treated as fixed benchmarks.
The reported 95% intervals reflect state-clustered coefficient uncertainty.

## Regression-based figures

### Original model

**Infant SCR 4 to 5; MGS/SCR fixed (Figure 4)**
  Starting $12,281; scenario $11,620; change $-661 (-5.38%).
  95% CI for percent change: [-10.90%, +0.48%].
  95% CI for scenario price from observed 4y/infant anchor: [$10,942, $12,340].

**Four-year-old SCR 10 to 12; MGS/SCR fixed (Figure 3)**
  Starting $10,855; scenario $10,387; change $-468 (-4.31%).
  95% CI for percent change: [-7.25%, -1.28%].
  95% CI for scenario price from observed 4y/infant anchor: [$10,068, $10,716].

**Four-year-old benchmark under three-year-old rules (7, 20)**
  Starting $10,855; scenario $11,667; change $812 (+7.48%).
  95% CI for percent change: [+1.08%, +14.28%].
  95% CI for scenario price from observed 4y/infant anchor: [$10,972, $12,405].

**Three-year-old proxy SCR 7 to 10; MGS/SCR fixed (orange curve)**
  Starting $11,667; scenario $10,703; change $-964 (-8.26%).
  95% CI for percent change: [-13.70%, -2.49%].
  95% CI for scenario price from observed 4y/infant anchor: [$10,345, $11,072].
  $1,000/month illustration: $917/month; savings $83/month.

**Three-year-old proxy SCR 7 to 10; MGS increases 20 to 25 (draft scenario)**
  Starting $11,667; scenario $10,855; change $-812 (-6.96%).
  95% CI for percent change: [-12.50%, -1.07%].
  95% CI for scenario price from observed 4y/infant anchor: [$10,855, $10,855].
  $1,000/month illustration: $930/month; savings $70/month.

**Three-year-old proxy SCR 7 to 10; MGS stays at 20 (sensitivity)**
  Starting $11,667; scenario $11,072; change $-594 (-5.10%).
  95% CI for percent change: [-13.07%, +3.61%].
  95% CI for scenario price from observed 4y/infant anchor: [$10,558, $11,611].
  $1,000/month illustration: $949/month; savings $51/month.

**Orange curve at SCR 10, using the 3y MGS/SCR ratio**
  Starting $10,855; scenario $10,703; change $-152 (-1.40%).
  95% CI for percent change: [-4.70%, +2.00%].
  95% CI for scenario price from observed 4y/infant anchor: [$10,345, $11,072].

## Figures requiring separate source data (not identified by this regression)

- Table 3: 2025 age-specific national ratio/MGS medians and state counts
  require the full 51-jurisdiction regulatory data at all seven ages.
- The Common Schools Trust Fund $2,508 per pupil requires updated
  state appropriation/distribution figures.
- Labor-market numbers (LFPR, unemployment, job openings, rankings)
  require current BLS Table 14 / LAUS / JOLTS data and consistent reference periods.
- Population under age five, its ranking, and births require updated PEP data.
- ND vs. SD median household income and childcare price differences
  require matching-year, matching-age source observations.
- The anchors of $12,281/$10,855 come from ND_REFERENCE_COST,
  not from a new estimate of 2025 observed annual prices.

IMPORTANT: the draft's 3-year-old scenario uses MGS 25 at SCR 10,
but the orange publication curve holds MGS/SCR fixed and therefore
uses MGS 28.5714 at SCR 10.
Both results are reported separately; don't interchange them.


## Demographic and BLS source notes

- Figure 1: Census PEP resident population by age/state, 2005-2025.
- Figure 5: published BLS Geographic Profile 2025 Table 14 (women); map data only.
- BLS Table 14 source: https://www.bls.gov/opub/geographic-profile/2025/home.htm
- Published 2025 BLS annual profile uses 11 CPS survey months; October excluded.
- No CPS microdata were downloaded or reaggregated by this script.
- Figure 6: Census PEP, cumulative change from 2005 (2005=0%) and annual percent change.
- Figure 7: births counts carried forward from the original user script.
- Figures 1 and 6 compare 50 states; the Figure 5 data include DC (51 jurisdictions).
- PEP source years 2005-2009: https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/state/st-est00int-agesex.csv
- PEP source years 2010-2019: https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/intercensal/state/asrh/sc-est2020int-alldata6.csv
- PEP source years 2020-2025: https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/state/asrh/sc-est2025-alldata6.csv
- PEP combines three vintages; check the 2010 and 2020 seams.

# Updated workforce paragraphs (nonseasonally adjusted BLS data)

North Dakota has one of the highest labor force participation rates in the United States. In September 2026, the state's labor force participation rate was 69.3%, the highest among the 50 states. Washington, D.C. reported a higher rate. Its unemployment rate was 1.8%, the second-lowest among the 50 states and Washington, D.C. Despite these favorable figures, North Dakota continues to face challenges filling job openings. In 2025, employers reported an average of approximately 21,750 job openings per month. In December 2025, North Dakota had 0.63 unemployed persons per job opening, ranking 2 lowest among the 50 states and Washington, D.C. In his 2024 State of the State Address, former North Dakota Governor Doug Burgum described workforce as 'our biggest challenge' ([source](https://www.governor.nd.gov/sites/www/files/documents/2024%20SOTS%20transcript%20for%20website.pdf)).

In 2025, North Dakota had the second-highest female labor force participation rate among U.S. states at 65.1%, behind Nebraska (67.1%). Including Washington, D.C. (70.1%), North Dakota ranked third overall, as shown in Figure 5. North Dakota's rate also exceeded neighboring South Dakota (64.3%) and Minnesota (63.9%).

Sources: BLS LAUS (2026), state JOLTS and LAUS (2025),
and published BLS Geographic Profile 2025, Table 14 (women).
BLS female LFPR: https://www.bls.gov/opub/geographic-profile/2025/home.htm
Published BLS 2025 Table 14 excludes October 2025; no CPS microdata were recalculated.
2025 job openings are not combined with 2026 unemployment counts.

## Historical context (2021–2025 averages)
North Dakota LFPR: 69.04% (rank 3 of 51).
North Dakota unemployment: 2.46% (rank 2 lowest of 51).
North Dakota job openings per unemployed: 2.568 (rank 2 of 51).

## 2026 year to date (common complete months)
North Dakota LFPR: 69.53%; unemployment: 2.41% (based on 9 months).

# North Dakota vs. South Dakota median household income

In 2025, South Dakota's median household income ($80,506) was 1.3% lower than North Dakota ($81,545) (U.S. Census Bureau, 2025 ACS 1-year).

Source: https://api.census.gov/data/2025/acs/acs1
ACS detailed table B19013, estimate and 90% margin of error.
The percent difference is computed relative to North Dakota.
The ACS estimates and their sampling errors should not be conflated
with BLS wage data or CPS ASEC median-income estimates.
A point-estimate difference is not necessarily statistically significant.

## North Dakota versus South Dakota: education comparison

          Age  ND price  SD price  Observed gap ($)  SD cheaper (%)  ND education  SD education  Simulated price  Education savings ($)  Education savings (%)  95% CI low ($)  95% CI high ($)  Education p-value
       infant   12281.0      8859            3422.0          27.864             3             1        11652.782                628.218                  5.115         120.636         1114.613              0.017
four_year_old   10855.0      8282            2573.0          23.703             3             1        10029.755                825.245                  7.602         352.227         1276.959              0.001
