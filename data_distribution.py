#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import distplot
from scipy.stats import norm, expon, lognorm
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, choose_grid, multiple_line_chart, multiple_bar_chart, bar_chart, HEIGHT

AIR_QUALITY_FILE = "data/air_quality_tabular.csv"
COLLISIONS_FILE = "data/NYC_collisions_tabular.csv"

register_matplotlib_converters()

air_data = pd.read_csv(AIR_QUALITY_FILE, index_col="FID", parse_dates=True, infer_datetime_format=True)
collisions_data = pd.read_csv(COLLISIONS_FILE, index_col="COLLISION_ID", parse_dates=True, infer_datetime_format=True).drop(["UNIQUE_ID", "VEHICLE_ID"], axis=1)
# %%
"""
---------- FIVE-NUMBER SUMMARY ----------
"""
summary5_air = air_data.describe()
summary5_air

#%%
summary5_colls = collisions_data.describe()
summary5_colls
# %%
"""
---------- BOXPLOTS ----------
These have the intent of helping us visualize what the
five-number summaries mean.
"""

def boxplot(data, path):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = plt.subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
        axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
# %%
boxplot(air_data, "images/lab1/distribution/air_quality_boxplot.png")
# %%
boxplot(collisions_data, "images/lab1/distribution/nyc_crashes_boxplot.png")
# %%
"""
---------- OUTLIERS ----------
Here, we want to compare between the outliers caught by the
IQR outliers and the n-std outliers.
"""
def outliers(data, path):
    #? Devemos explorar diferentes NR_STDEV?
    NR_STDEV: int = 2

    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    outliers_iqr = []
    outliers_stdev = []
    summary5 = data.describe(include='number')

    for var in numeric_vars:
        iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
        outliers_iqr += [
            data[data[var] > summary5[var]['75%']  + iqr].count()[var] +
            data[data[var] < summary5[var]['25%']  - iqr].count()[var]]
        std = NR_STDEV * summary5[var]['std']
        outliers_stdev += [
            data[data[var] > summary5[var]['mean'] + std].count()[var] +
            data[data[var] < summary5[var]['mean'] - std].count()[var]]

    outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
    plt.figure(figsize=(12, HEIGHT))
    multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

#%%
outliers(air_data, "images/lab1/distribution/air_quality_outliers.png")
# %%
outliers(collisions_data, "images/lab1/distribution/nyc_crashes_outliers.png")

# %%
"""
---------- HISTOGRAMS ----------
The previous diagrams don't give us a sense of the
distribution of all of the individual variables. This is
why we will use histograms next.
"""
def histogram(data, path):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = plt.subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0

    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
        axs[i, j].set_xlabel(numeric_vars[n])
        axs[i, j].set_ylabel("nr records")
        axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

# %%
histogram(air_data, "images/lab1/distribution/air_quality_histograms.png")
# %%
histogram(collisions_data, "images/lab1/distribution/nyc_crashes_histograms.png")
# %%
def histogram_with_trend(data, path):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = plt.subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
        distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
#%%
histogram_with_trend(air_data, "images/lab1/distribution/air_quality_histograms_trend.png")
# %%
histogram_with_trend(collisions_data, "images/lab1/distribution/nyc_crashes_histograms_trend.png")
#%%
def compute_known_distributions(x_values: list) -> dict:
    #? Devemos experimentar mais distribuicoes?
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(np.log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
    return distributions

def histogram_with_distributions(ax: plt.Axes, series: pd.Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    distributions = compute_known_distributions(values)
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')

def fit_distributions(data, path):

    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = plt.subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

# %%
fit_distributions(air_data, "images/lab1/distribution/air_quality_histograms_fits.png")
# %%
fit_distributions(collisions_data, "images/lab1/distribution/nyc_crashes_histograms_fits.png")
#%%
"""
---------- SYMBOLIC VARIABLES ----------
Symbolic variables can't be evaluated with the previous exploration
techniques. Hence, we will use bar graphs.
"""
def sym_bars(data, path):
    symbolic_vars = get_variable_types(data)['Symbolic']
    if [] == symbolic_vars:
        raise ValueError('There are no symbolic variables.')

    rows, cols = choose_grid(len(symbolic_vars))
    fig, axs = plt.subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(symbolic_vars)):
        counts = data[symbolic_vars[n]].value_counts()
        bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s'%symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
#%%
sym_bars(collisions_data, "images/lab1/distribution/nyc_crashes_syms.png")
#%%
#! Tive que apagar uma entrada com um "s" no GbCity para essa coluna nao contar como symbolic || Scratch that, a coluna fica sempre como object..?
sym_bars(air_data.drop("GbCity", axis=1), "images/lab1/distribution/air_quality_syms.png")