
#%%
from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import register_matplotlib_converters

from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types

# Extract data from files to dataframes

register_matplotlib_converters()
tabulars = {"air":'data/air_quality_tabular.csv',"nyc":'data/NYC_collisions_tabular.csv'}
timeseries = {"air":'data/air_quality_timeseries.csv',"nyc":'data/NYC_collisions_timeseries.csv'}

data_tab = {}
data_time = {}

data_tab["air"] = read_csv(tabulars["air"], index_col='FID', na_values='', parse_dates=False, infer_datetime_format=False)
data_tab["nyc"] = read_csv(tabulars["nyc"], index_col='COLLISION_ID', na_values='', parse_dates=False, infer_datetime_format=False)
data_time["air"] = read_csv(timeseries["air"], index_col='DATE', na_values='', parse_dates=True, infer_datetime_format=True)
data_time["nyc"] = read_csv(timeseries["nyc"], index_col='timestamp', na_values='', parse_dates=True, infer_datetime_format=True)
print(data_tab["air"].shape)
print(data_tab["nyc"].shape)
print(data_time["air"].shape)
print(data_time["nyc"].shape)


# Make barcharts of number of records vs vars
def bar_chart_records_variables(data, filename):
    figure(figsize=(4,2))
    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
    bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
    savefig(filename)
    show()

bar_chart_records_variables(data_tab["air"], "./images/lab1/dimensionality/air_tabular_records_variables.png")
bar_chart_records_variables(data_tab["nyc"], "./images/lab1/dimensionality/nyc_tabular_records_variables.png")
bar_chart_records_variables(data_time["air"], "./images/lab1/dimensionality/air_time_records_variables.png")
bar_chart_records_variables(data_time["nyc"], "./images/lab1/dimensionality/nyc_time_records_variables.png")

# Print data types

for var in ["air","nyc"]:
    print("------------Data types for "+var+":------------\n---timeseries:---")
    print(data_time[var].dtypes)
    print("---tabular:---")
    print(data_tab[var].dtypes)

# Turn object into category

def object_to_category(data):
    cat_vars = data.select_dtypes(include='object')
    data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))

for var in ["air","nyc"]:
    print("------------Data types for "+var+":------------\n---timeseries:---")
    print(data_time[var].dtypes)
    print("---tabular:---")
    print(data_tab[var].dtypes)


# Get counts of data types


def get_variable_types(df: DataFrame) -> dict:
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for c in df.columns:
        uniques = df[c].dropna(inplace=False).unique()
        if len(uniques) == 2:
            variable_types['Binary'].append(c)
            df[c].astype('bool')
        elif df[c].dtype == 'datetime64':
            variable_types['Date'].append(c)
        elif df[c].dtype == 'int':
            variable_types['Numeric'].append(c)
        elif df[c].dtype == 'float':
            variable_types['Numeric'].append(c)
        else:
            df[c].astype('category')
            variable_types['Symbolic'].append(c)
    return variable_types


def get_data_type_counts(data, filename):
    variable_types = get_variable_types(data)
    counts = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])
    figure(figsize=(4,2))
    bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
    savefig(filename)
    show()


for var in ["air","nyc"]:
    get_data_type_counts(data_tab[var], "./images/lab1/dimensionality/" + var + "_tabular_var_types.png")
    get_data_type_counts(data_time[var], "./images/lab1/dimensionality/" + var + "_time_var_types.png")


# count missing values per variable

def count_missing_values(data, filename):
    mv = {}
    for var in data:
        nr = data[var].isna().sum()
        if nr > 0:
            mv[var] = nr

    figure()
    print(len(list(mv.keys())))
    print(len(list(mv.values())))
    bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
                xlabel='variables', ylabel='nr missing values', rotation=True)
    savefig(filename)
    show()

for var in ["air","nyc"]:
    count_missing_values(data_tab[var], "./images/lab1/dimensionality/" + var + "_tabular_missing_values.png")
    #count_missing_values(data_time[var], "./images/lab1/dimensionality/" + var + "_time_var_missing_values.png")
# %%
