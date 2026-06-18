#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
plt.style.use("ricky_mocha")

#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Load and prepare data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# created by csv_maker.py file, which contains data sources and loading.
df = pd.read_csv("data/public/reg_data_2005_2023.csv")


#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. Construct linear time trend (t = 0, 1, 2, …)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df["year"] = df["year"].astype(int)

unique_years = sorted(df["year"].unique())
year_to_t = {year: i for i, year in enumerate(unique_years)}
df["t"] = df["year"].map(year_to_t) # type: ignore[arg-type]

# Sort before lagging
df = df.sort_values(["st", "year"])


#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. Normalized Log Variables
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# NOT centered
df["ln_r_priv_gdp"] = np.log(df["r_priv_gdp"])

# All centered logs (log of var divided by mean)
df["ln_domestics_c"]    = np.log(df["domestics"]    / df["domestics"].mean())
df["ln_immigrants_c"]   = np.log(df["immigrants"]   / df["immigrants"].mean())
df["ln_efi_c"]          = np.log(df["efi_score"]    / df["efi_score"].mean())
df["ln_bach_c"]         = np.log(df["bach_share"]   / df["bach_share"].mean())
df["ln_cmshare_c"]      = np.log(df["cm_share"]     / df["cm_share"].mean())
df["ln_ehsshare_c"]     = np.log(df["ehs_share"]    / df["ehs_share"].mean())
df["ln_ifpbsshare_c"]   = np.log(df["ifpbs_share"]  / df["ifpbs_share"].mean())
df["ln_nrmshare_c"]     = np.log(df["nrm_share"]    / df["nrm_share"].mean())



#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. Translog Interactions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ----- Halved Quadratic Terms (0.5 * x^2)
df["h_ln_domestics_c_2"]    = 0.5 * df["ln_domestics_c"]**2
df["h_ln_immigrants_c_2"]   = 0.5 * df["ln_immigrants_c"]**2
df["h_t_2"]                 = 0.5 * df["t"]**2
df["h_ln_efi_c_2"]          = 0.5 * df["ln_efi_c"]**2
df["h_ln_bach_c_2"]         = 0.5 * df["ln_bach_c"]**2
df["h_ln_cmshare_c_2"]      = 0.5 * df["ln_cmshare_c"]**2
df["h_ln_ehsshare_c_2"]     = 0.5 * df["ln_ehsshare_c"]**2
df["h_ln_ifpbsshare_c_2"]   = 0.5 * df["ln_ifpbsshare_c"]**2
df["h_ln_nrmshare_c_2"]     = 0.5 * df["ln_nrmshare_c"]**2

# ----- Time Interactions
df["ln_domestics_c_t"]      = df["ln_domestics_c"]  * df["t"]
df["ln_immigrants_c_t"]     = df["ln_immigrants_c"] * df["t"]

df["ln_efi_c_t"]            = df["ln_efi_c"]        * df["t"]
df["ln_bach_c_t"]           = df["ln_bach_c"]       * df["t"]

df["ln_cmshare_c_t"]        = df["ln_cmshare_c"]    * df["t"]
df["ln_ehsshare_c_t"]       = df["ln_ehsshare_c"]   * df["t"]
df["ln_ifpbsshare_c_t"]     = df["ln_ifpbsshare_c"] * df["t"]
df["ln_nrmshare_c_t"]       = df["ln_nrmshare_c"]   * df["t"]

# ----- Economic Freedom Interactions
df["ln_efi_c_ln_bach_c"]         = df["ln_efi_c"]  * df["ln_bach_c"]

df["ln_efi_c_ln_cmshare_c"]      = df["ln_efi_c"] * df["ln_cmshare_c"]
df["ln_efi_c_ln_ehsshare_c"]     = df["ln_efi_c"] * df["ln_ehsshare_c"]
df["ln_efi_c_ln_ifpbsshare_c"]   = df["ln_efi_c"] * df["ln_ifpbsshare_c"]
df["ln_efi_c_ln_nrmshare_c"]     = df["ln_efi_c"] * df["ln_nrmshare_c"]

# ----- Education Interactions
df["ln_bach_c_ln_cmshare_c"]     = df["ln_bach_c"] * df["ln_cmshare_c"]
df["ln_bach_c_ln_ehsshare_c"]    = df["ln_bach_c"] * df["ln_ehsshare_c"]
df["ln_bach_c_ln_ifpbsshare_c"]  = df["ln_bach_c"] * df["ln_ifpbsshare_c"]
df["ln_bach_c_ln_nrmshare_c"]    = df["ln_bach_c"] * df["ln_nrmshare_c"]

# ----- Industry Interactions
df["ln_cmshare_c_ln_ehsshare_c"]   = df["ln_cmshare_c"] *df["ln_ehsshare_c"]
df["ln_cmshare_c_ln_ifpbsshare_c"] = df["ln_cmshare_c"] *df["ln_ifpbsshare_c"]
df["ln_cmshare_c_ln_nrmshare_c"]   = df["ln_cmshare_c"] *df["ln_nrmshare_c"]

df["ln_ehsshare_c_ln_ifpbsshare_c"] =df["ln_ehsshare_c"]*df["ln_ifpbsshare_c"]
df["ln_ehsshare_c_ln_nrmshare_c"]   =df["ln_ehsshare_c"]*df["ln_nrmshare_c"]

df["ln_ifpbsshare_c_ln_nrmshare_c"] =df["ln_ifpbsshare_c"]*df["ln_nrmshare_c"]


# ----- Other Interactions
df["ln_domestics_immigrant_c"]        =df["ln_domestics_c"] * df["ln_immigrants_c"]

df["ln_domestics_c_ln_efi_c"]         =df["ln_domestics_c"] * df["ln_efi_c"]
df["ln_domestics_c_ln_bach_c"]        =df["ln_domestics_c"] * df["ln_bach_c"]

df["ln_immigrants_c_ln_efi_c"]        =df["ln_immigrants_c"] *df["ln_efi_c"]
df["ln_immigrants_c_ln_bach_c"]       =df["ln_immigrants_c"] *df["ln_bach_c"]

df["ln_domestics_c_ln_cmshare_c"]     =df["ln_domestics_c"] *df["ln_cmshare_c"]
df["ln_domestics_c_ln_ehsshare_c"]    =df["ln_domestics_c"] *df["ln_ehsshare_c"]
df["ln_domestics_c_ln_ifpbsshare_c"] =df["ln_domestics_c"] *df["ln_ifpbsshare_c"]
df["ln_domestics_c_ln_nrmshare_c"]    =df["ln_domestics_c"] *df["ln_nrmshare_c"]

df["ln_immigrants_c_ln_cmshare_c"]    =df["ln_immigrants_c"]*df["ln_cmshare_c"]
df["ln_immigrants_c_ln_ehsshare_c"]   =df["ln_immigrants_c"]*df["ln_ehsshare_c"]
df["ln_immigrants_c_ln_ifpbsshare_c"]=df["ln_immigrants_c"]*df["ln_ifpbsshare_c"]
df["ln_immigrants_c_ln_nrmshare_c"]   =df["ln_immigrants_c"]*df["ln_nrmshare_c"]


#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5. Regression
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

formula = """
ln_r_priv_gdp ~ ln_domestics_c + h_ln_domestics_c_2 + ln_domestics_c_t

              + ln_immigrants_c + h_ln_immigrants_c_2 + ln_immigrants_c_t 

              + t + h_t_2 

              + ln_efi_c + h_ln_efi_c_2 + ln_efi_c_t 

              + ln_bach_c + h_ln_bach_c_2 + ln_bach_c_t 

              + ln_domestics_immigrant_c 

              + ln_domestics_c_ln_efi_c
              + ln_domestics_c_ln_bach_c 

              + ln_immigrants_c_ln_efi_c
              + ln_immigrants_c_ln_bach_c               

              + ln_efi_c_ln_bach_c

              + ln_cmshare_c + ln_ehsshare_c + ln_ifpbsshare_c 
              + ln_nrmshare_c 
 
              + ln_domestics_c_ln_cmshare_c + ln_domestics_c_ln_ehsshare_c 
              + ln_domestics_c_ln_ifpbsshare_c + ln_domestics_c_ln_nrmshare_c 
 
              + ln_immigrants_c_ln_cmshare_c + ln_immigrants_c_ln_ehsshare_c 
              + ln_immigrants_c_ln_ifpbsshare_c + ln_immigrants_c_ln_nrmshare_c 

              + h_ln_cmshare_c_2 + h_ln_ehsshare_c_2
              + h_ln_ifpbsshare_c_2 + h_ln_nrmshare_c_2

              + ln_cmshare_c_t + ln_ehsshare_c_t 
              + ln_ifpbsshare_c_t + ln_nrmshare_c_t 
 
              + ln_cmshare_c_ln_ehsshare_c + ln_cmshare_c_ln_ifpbsshare_c 
              + ln_cmshare_c_ln_nrmshare_c 
 
              + ln_ehsshare_c_ln_ifpbsshare_c + ln_ehsshare_c_ln_nrmshare_c

              + ln_ifpbsshare_c_ln_nrmshare_c 
 
              + ln_efi_c_ln_cmshare_c + ln_efi_c_ln_ehsshare_c
              + ln_efi_c_ln_ifpbsshare_c + ln_efi_c_ln_nrmshare_c
 
              + ln_bach_c_ln_cmshare_c + ln_bach_c_ln_ehsshare_c
              + ln_bach_c_ln_ifpbsshare_c + ln_bach_c_ln_nrmshare_c 
 
              + C(st)
"""

ols = smf.ols(formula=formula, data=df)

groups = (
    df.loc[ols.data.row_labels, ["st", "year"]]
      .apply(lambda x: pd.Categorical(x).codes)
      .to_numpy()
)

model_gdp_cse = ols.fit(
    cov_type="cluster",
    cov_kwds={"groups": groups}
)

print(model_gdp_cse.summary())
model_gdp = smf.ols(formula=formula, data=df).fit()

# Coefficient table
results_df = pd.DataFrame({
    "term": model_gdp_cse.params.index,
    "coef": model_gdp_cse.params.values,
    "std_err": model_gdp_cse.bse.values,
    "t": model_gdp_cse.tvalues.values,
    "p_value": model_gdp_cse.pvalues.values,
    "ci_lower": model_gdp_cse.conf_int()[0].values,
    "ci_upper": model_gdp_cse.conf_int()[1].values
})

# Model-level statistics
model_stats = pd.DataFrame({
    "term": [
        "N",
        "R_squared",
        "Adj_R_squared",
        "F_statistic",
        "F_p_value",
        "AIC",
        "BIC",
        "df_model",
        "df_resid"
    ],
    "coef": [
        int(model_gdp_cse.nobs),
        model_gdp_cse.rsquared,
        model_gdp_cse.rsquared_adj,
        model_gdp_cse.fvalue,
        model_gdp_cse.f_pvalue,
        model_gdp_cse.aic,
        model_gdp_cse.bic,
        model_gdp_cse.df_model,
        model_gdp_cse.df_resid
    ],
    "std_err": "",
    "t": "",
    "p_value": "",
    "ci_lower": "",
    "ci_upper": ""
})

# Combine and export
export_df = pd.concat([results_df, model_stats], ignore_index=True)

export_df.to_csv("Reg_6_12_24.csv", index=False)


#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Graph Inspection
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model_dict = {
    "model_gdp_cse": {
        "model": model_gdp_cse,
        "resid": "ln_r_priv_gdp_resid"
    },
    "model_gdpfe": {
        "model": model_gdpfe,
        "resid": "ln_r_priv_gdp_resid"
    },
}

#%% --- Select Model and Run  

model_name = "model_gdpfe"   # choose: "model_dom", "model_imm",

model = model_dict[model_name]["model"]
var = model_dict[model_name]["resid"]

#%% --- Graph Inspection

plt.scatter(model.fittedvalues, model.resid, alpha=0.5)
plt.axhline(0, color='white', linestyle='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted")
plt.show()
plt.close('all')

#%%
sm.qqplot(model.resid, line='45', fit=True, alpha=0.1)
plt.title("Q–Q Plot of Regression Residuals")
plt.show()
plt.close('all')

#%%
plt.plot(df["t"], model.resid, 'o', alpha=0.2)
plt.axhline(0, color='white', linestyle='--')
plt.xlabel("Time (t)")
plt.ylabel("Residuals")
plt.title("Residuals Over Time")
plt.show()
plt.close('all')

#%%
state_means = df.groupby("st")[var].mean()
state_means.plot(kind='bar', fontsize=8)
plt.title("Mean Residual by State")
plt.axhline(0, color='white', linestyle='--')
plt.ylabel("Residual Mean")
plt.show()
plt.close()

#%% Cook's Distance
influence = model.get_influence()
inf_df = influence.summary_frame()

plt.figure()
plt.stem(inf_df["cooks_d"], markerfmt=",")
plt.title("Cook's Distance for Each Observation")
plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.show()
plt.close()


#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Time Fixed-Effects
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

formula = """
ln_r_priv_gdp ~ ln_domestics_c + h_ln_domestics_c_2

              + ln_immigrants_c + h_ln_immigrants_c_2 

              + ln_efi_c + h_ln_efi_c_2 

              + ln_bach_c + h_ln_bach_c_2 

              + ln_domestics_immigrant_c 

              + ln_domestics_c_ln_efi_c
              + ln_domestics_c_ln_bach_c 

              + ln_immigrants_c_ln_efi_c
              + ln_immigrants_c_ln_bach_c               

              + ln_efi_c_ln_bach_c

              + ln_cmshare_c + ln_ehsshare_c + ln_ifpbsshare_c 
              + ln_nrmshare_c 
 
              + ln_domestics_c_ln_cmshare_c + ln_domestics_c_ln_ehsshare_c 
              + ln_domestics_c_ln_ifpbsshare_c + ln_domestics_c_ln_nrmshare_c 
 
              + ln_immigrants_c_ln_cmshare_c + ln_immigrants_c_ln_ehsshare_c 
              + ln_immigrants_c_ln_ifpbsshare_c + ln_immigrants_c_ln_nrmshare_c 

              + h_ln_cmshare_c_2 + h_ln_ehsshare_c_2
              + h_ln_ifpbsshare_c_2 + h_ln_nrmshare_c_2
 
              + ln_cmshare_c_ln_ehsshare_c + ln_cmshare_c_ln_ifpbsshare_c 
              + ln_cmshare_c_ln_nrmshare_c 
 
              + ln_ehsshare_c_ln_ifpbsshare_c + ln_ehsshare_c_ln_nrmshare_c

              + ln_ifpbsshare_c_ln_nrmshare_c 
 
              + ln_efi_c_ln_cmshare_c + ln_efi_c_ln_ehsshare_c
              + ln_efi_c_ln_ifpbsshare_c + ln_efi_c_ln_nrmshare_c
 
              + ln_bach_c_ln_cmshare_c + ln_bach_c_ln_ehsshare_c
              + ln_bach_c_ln_ifpbsshare_c + ln_bach_c_ln_nrmshare_c 
 
              + C(st) + C(year)
"""

state_codes = df["st"].astype("category").cat.codes
year_codes  = df["year"].astype("category").cat.codes

model_gdpfe = smf.ols(formula=formula, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": [state_codes, year_codes]}
)

print(model_gdpfe.summary())


