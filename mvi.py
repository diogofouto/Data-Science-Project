#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from ds_charts import bar_chart, get_variable_types
from pandas.plotting import register_matplotlib_converters

AQ_FNAME = "air_quality_tabular"
AIR_QUALITY_FILE = "data/air_quality_tabular.csv"
COL_FNAME = "NYC_collisions_tabular"
COLLISIONS_FILE = "data/NYC_collisions_tabular.csv"

register_matplotlib_converters()

air_data = pd.read_csv(AIR_QUALITY_FILE, index_col="FID", parse_dates=True, infer_datetime_format=True)
collisions_data = pd.read_csv(COLLISIONS_FILE, index_col="COLLISION_ID", parse_dates=True, infer_datetime_format=True)
#%%
def na_count_chart(data, path):
    mv = {}
    plt.figure()
    for var in data:
        nr = data[var].isna().sum()
        if nr > 0:
            mv[var] = nr

    bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable', xlabel='variables', ylabel='nr missing values', rotation=True)
    plt.savefig(path)

    return mv
#%%
air_mvs = na_count_chart(air_data, "images/lab2/mvi/air_quality_mv.png")
#%%
crashes_mvs = na_count_chart(collisions_data, "images/lab2/mvi/nyc_crashes_mv.png")
# %%
"""
---------- DROP OF MISSING VALUES ----------
It's a dangerous thing to drop missing values. We need to make sure
that, when dropping a column, there are still enough values to work
with.
"""

def mv_drop_vars(data, mv, file):
    # defines the number of records to discard entire columns
    threshold = data.shape[0] * 0.90

    missings = [c for c in mv.keys() if mv[c] > threshold]
    df = data.drop(columns=missings, inplace=False)
    df.to_csv(f'data/{file}_drop_columns_mv.csv', index=True)
    print('Dropped variables', missings)
#%%
mv_drop_vars(air_data, air_mvs, AQ_FNAME)
#%%
mv_drop_vars(collisions_data, crashes_mvs, COL_FNAME)
# %%
"""
As seen by the prints, no variables were dropped. This is a good sign.
Now, we will try to drop records that have a lot of missing values.
"""
def mv_drop_recs(data, file):
    print("Before: " + str(data.shape))
    # defines the number of variables to discard entire records
    threshold = data.shape[1] * 0.50

    df = data.dropna(thresh=threshold, inplace=False)
    df.to_csv(f'data/{file}_drop_records_mv.csv', index=True)
    print("After: " + str(df.shape))
#%%
mv_drop_recs(air_data, AQ_FNAME)
"""
Quite a few records are dropped in this dataset: 7642 records, to be exact.
We end up with only 161631 records.
"""
#%%
mv_drop_recs(collisions_data, COL_FNAME)
"""
In this dataset, no records are dropped.
"""
#%%
"""
---------- FILLING OF MISSING VALUES ----------
Filling missing values needs to be done in a careful manner as well.
We want to modify as little as possible the original distribution of the
data.
"""

#! Only filling values with mean or most_frequent. Filling with constants
#! didn't make sense to me, but I might be wrong.

def fill_missing_values(data, file):
    tmp_nr, tmp_sb, tmp_bool = None, None, None
    variables = get_variable_types(data)
    numeric_vars = variables['Numeric']
    symbolic_vars = variables['Symbolic']
    binary_vars = variables['Binary']

    tmp_nr, tmp_sb, tmp_bool = None, None, None
    if len(numeric_vars) > 0:
        imp = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
        tmp_nr = pd.DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
    if len(symbolic_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=np.nan, copy=True)
        tmp_sb = pd.DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
    if len(binary_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=np.nan, copy=True)
        tmp_bool = pd.DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

    df = pd.concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
    df.to_csv(f'data/{file}_mv_most_frequent.csv', index=True)
    return df

#%%
air_data_filled = fill_missing_values(air_data, AQ_FNAME)
air_data.describe(include='all')
#%%
air_data_filled.describe(include='all')
#%%
coll_data_filled = fill_missing_values(collisions_data, COL_FNAME)
collisions_data.describe(include='all')
#%%
coll_data_filled.describe(include='all')