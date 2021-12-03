#%%

from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show,  figure, title
from ds_charts import get_variable_types, HEIGHT
from seaborn import heatmap

def scatter_sparcity(data, output_filename):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(len(numeric_vars)):
        var1 = numeric_vars[i]
        for j in range(i+1, len(numeric_vars)):
            var2 = numeric_vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])
    savefig(output_filename)
    show()

# Prepare data

register_matplotlib_converters()
tabulars = {"air":'data/air_quality_tabular.csv',"nyc":'data/NYC_collisions_tabular.csv'}
timeseries = {"air":'data/air_quality_timeseries.csv',"nyc":'data/NYC_collisions_timeseries.csv'}

data_tab = {}
data_time = {}

data_tab["air"] = read_csv(tabulars["air"], index_col='FID', na_values='', parse_dates=False, infer_datetime_format=False)
data_tab["nyc"] = read_csv(tabulars["nyc"], index_col='COLLISION_ID', na_values='', parse_dates=False, infer_datetime_format=False)
data_time["air"] = read_csv(timeseries["air"], index_col='DATE', na_values='', parse_dates=True, infer_datetime_format=True)
data_time["nyc"] = read_csv(timeseries["nyc"], index_col='timestamp', na_values='', parse_dates=True, infer_datetime_format=True)

#Make scatter plots

#scatter_sparcity(data_tab["air"], "./images/lab1/sparcity/air_tabular_sparcity_scatter.png")
#scatter_sparcity(data_tab["nyc"], "./images/lab1/sparcity/nyc_tabular_sparcity_scatter.png")
scatter_sparcity(data_time["air"], "./images/lab1/sparcity/air_time_sparcity_scatter.png")
scatter_sparcity(data_time["nyc"], "./images/lab1/sparcity/nyc_time_sparcity_scatter.png")

#%%
# make plots for symbolic data

def sparcity_for_symbols(data, output_filename):
    symbolic_vars = get_variable_types(data)['Symbolic']
    if [] == symbolic_vars:
        raise ValueError('There are no symbolic variables.')

    rows, cols = len(symbolic_vars)-1, len(symbolic_vars)-1
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(len(symbolic_vars)):
        var1 = symbolic_vars[i]
        for j in range(i+1, len(symbolic_vars)):
            var2 = symbolic_vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])
    savefig(output_filename)
    show()

sparcity_for_symbols(data_tab["air"], "./images/lab1/sparcity/air_tabular_sparcity_symbols.png")
sparcity_for_symbols(data_tab["nyc"], "./images/lab1/sparcity/nyc_tabular_sparcity_symbols.png")
sparcity_for_symbols(data_time["air"], "./images/lab1/sparcity/air_time_sparcity_symbols.png")
sparcity_for_symbols(data_time["nyc"], "./images/lab1/sparcity/nyc_time_sparcity_symbols.png")


# print correlation matrixes

print(data_tab["air"].corr())
print(data_tab["nyc"].corr())
print(data_time["air"].corr())
print(data_time["nyc"].corr())

# plot heatmaps

def heatmap(data, output_filename):
    corr_mtx = data.corr()
    fig = figure(figsize=[12, 12])

    heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    title('Correlation analysis')
    savefig(output_filename)
    show()
    
heatmap(data_tab["air"], "./images/lab1/sparcity/air_tabular_corr_heatmap.png")
heatmap(data_tab["nyc"], "./images/lab1/sparcity/nyc_tabular_corr_heatmap.png")
heatmap(data_time["air"], "./images/lab1/sparcity/air_time_corr_heatmap.png")
heatmap(data_time["nyc"], "./images/lab1/sparcity/nyc_time_corr_heatmap.png")
#%%