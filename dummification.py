#%%
from pandas import read_csv, DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types
from sklearn.preprocessing import OneHotEncoder
from numpy import number


register_matplotlib_converters()

AIR_QUALITY_FILE = "data/air_quality_tabular_mv_most_frequent.csv"
COLLISIONS_FILE = "data/NYC_collisions_tabular_mv_most_frequent.csv"


air_data = read_csv(AIR_QUALITY_FILE, index_col="FID", na_values="", parse_dates=True, infer_datetime_format=True)
collisions_data = read_csv(COLLISIONS_FILE, index_col="COLLISION_ID", na_values="", parse_dates=True, infer_datetime_format=True)

# Drop out all records with missing values
air_data.dropna(inplace=True)
collisions_data.dropna(inplace=True)

def dummify(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
    X = df[vars_to_dummify]
    print(X)
    encoder.fit(X)
    new_vars = encoder.get_feature_names(vars_to_dummify)
    trans_X = encoder.transform(X)
    dummy = DataFrame(trans_X, columns=new_vars, index=X.index)
    dummy = dummy.convert_dtypes(convert_boolean=True)

    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df

def dummy(data, filename):
    variables = get_variable_types(data)
    symbolic_vars = variables['Symbolic']
    data = data.astype(str)
    df = dummify(data, symbolic_vars)
    df.to_csv(f'data/{filename}_dummified.csv', index=True)

    return df

#%%
dummified_collision_data = dummy(collisions_data, "NYC_collisions_tabular")
dummified_collision_data.describe(include=[bool])

#%%
dummified_air_data = dummy(air_data, "air_quality_tabular")
dummified_air_data.describe(include=[bool])