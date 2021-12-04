#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import distplot
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, choose_grid, multiple_bar_chart, bar_chart, HEIGHT

AIR_QUALITY_FILE = "data/air_quality_tabular.csv"
COLLISIONS_FILE = "data/NYC_collisions_tabular.csv"

register_matplotlib_converters()

air_data = pd.read_csv(AIR_QUALITY_FILE, index_col="FID", na_values="", parse_dates=True, infer_datetime_format=True)
collisions_data = pd.read_csv(COLLISIONS_FILE, index_col="COLLISION_ID", na_values="", parse_dates=True, infer_datetime_format=True)

"""
---------- FIVE-NUMBER SUMMARY ----------
"""
# %%
summary5_air = air_data.describe()
summary5_air

#%%
summary5_colls = collisions_data.describe()
summary5_colls

"""
---------- BOXPLOTS ----------
These have the intent of helping us visualize what the
five-number summaries mean.
"""

# %%
def boxplot(data, path):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = plt.subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT))
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        if rows == 1:
            axs[j].set_title('Boxplot for %s'%numeric_vars[n])
            axs[j].boxplot(data[numeric_vars[n]].dropna().values)
        else:
            axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
            axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.savefig(path)
    plt.show()
# %%
boxplot(air_data, "images/lab1/distribution/air_quality_boxplot.png")
# %%
boxplot(collisions_data, "images/lab1/distribution/nyc_crashes_boxplot.png")

#TODO: retirar variaveis parvos tipo os IDs

"""
---------- OUTLIERS ----------
Here, we want to compare between the outliers caught by the
IQR outliers and the n-std outliers.
"""
# %%
def outliers(data, path):
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
    plt.savefig(path)
    plt.show()

#TODO: retirar variaveis parvas tipo os IDs

# %%
outliers(collisions_data, "images/lab1/distribution/nyc_crashes_outliers.png")
#%%
outliers(air_data, "images/lab1/distribution/air_quality_outliers.png")

# %%
"""
---------- HISTOGRAMS ----------
The previous diagrams don't give us a sense of the
distribution of all of the individual variables. This is
why we will use histograms next.
"""

#TODO: retirar variaveis parvas tipo os IDs
#TODO: Meter escala boa para a idade das pessoas.

def histogram(data, path):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = plt.subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT))
    i, j = 0, 0

    for n in range(len(numeric_vars)):
        if rows == 1:
            axs[j].set_title('Histogram for %s'%numeric_vars[n])
            axs[j].set_xlabel(numeric_vars[n])
            axs[j].set_ylabel("nr records")
            axs[j].hist(data[numeric_vars[n]].dropna().values, 'auto')
        else:
            axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
            axs[i, j].set_xlabel(numeric_vars[n])
            axs[i, j].set_ylabel("nr records")
            axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.savefig(path)
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
    fig, axs = plt.subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT))
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        if rows == 1:
            axs[j].set_title('Histogram with trend for %s'%numeric_vars[n])
            distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[j], axlabel=numeric_vars[n])
        else:
            axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
            distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.savefig(path)
    plt.show()

# %%
histogram_with_trend(air_data, "images/lab1/distribution/air_quality_histograms_trend.png")
# %%
histogram_with_trend(collisions_data, "images/lab1/distribution/nyc_crashes_histograms_trend.png")
#%%
#TODO: Criar histogramas com o fit de varias distribuicoes diferentes.
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
    plt.savefig(path)
    plt.show()
#%%
sym_bars(collisions_data.drop(["PERSON_ID"], axis=1), "images/lab1/distribution/nyc_crashes_syms.png")
#%%
#! Tive que apagar uma entrada com um "s" no GbCity para essa coluna nao contar como symbolic || Scratch that, a coluna fica sempre como object..?
sym_bars(air_data.drop("GbCity", axis=1), "images/lab1/distribution/air_quality_syms.png")
# %%
air_data.dtypes
# %%
