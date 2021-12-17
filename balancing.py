#%%
from pandas import read_csv, concat, DataFrame,Series
from imblearn.over_sampling import SMOTE
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, multiple_bar_chart


def class_balance(data, class_var, output_file):
    print("BALANCE FOR ", class_var)
    target_count = data[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()
    #ind_positive_class = target_count.index.get_loc(positive_class)
    print('Minority class=', positive_class, ':', target_count[positive_class])
    print('Majority class=', negative_class, ':', target_count[negative_class])
    print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
    values = {'Original': [target_count[positive_class], target_count[negative_class]]}

    figure()
    bar_chart(target_count.index, target_count.values, title='Class balance')
    savefig(f'images/lab4/balancing/{output_file}_balance.png')
    show()

    return values, positive_class, negative_class

#%%
AIR_QUALITY_FILE = "data/air_quality_scaled_zscore.csv"
AIR_CLASS = "ALARM"
COLLISIONS_FILE = "data/NYC_collisions_scaled_minmax.csv"
COLLISIONS_CLASS = "PERSON_INJURY"

air_data = read_csv(AIR_QUALITY_FILE, na_values="", parse_dates=True, infer_datetime_format=True)
collisions_data = read_csv(COLLISIONS_FILE, na_values="", parse_dates=True, infer_datetime_format=True)

# %%
values_air, positive_class_air, negative_class_air = class_balance(air_data, AIR_CLASS, "air_quality_tabular")
#%%
values_col, positive_class_col, negative_class_col = class_balance(collisions_data, COLLISIONS_CLASS, "NYC_collisions_tabular")
#%% COUNT POSITIVE AND NEGATIVE CASES

df_positives_air = air_data[air_data[AIR_CLASS] == positive_class_air]
df_negatives_air = air_data[air_data[AIR_CLASS] == negative_class_air]

#%%
df_positives_col = collisions_data[collisions_data[COLLISIONS_CLASS] == positive_class_col]
df_negatives_col = collisions_data[collisions_data[COLLISIONS_CLASS] == negative_class_col]
# %%

def undersample(values, positive_class, negative_class, df_negatives, df_positives, output_file):
    df_neg_sample = DataFrame(df_negatives.sample(len(df_positives)))
    df_under = concat([df_positives, df_neg_sample], axis=0)
    df_under.to_csv(f'data/{output_file}_under.csv', index=False)
    values['UnderSample'] = [len(df_positives), len(df_neg_sample)]
    print('Minority class=', positive_class, ':', len(df_positives))
    print('Majority class=', negative_class, ':', len(df_neg_sample))
    print('Proportion:', round(len(df_positives) / len(df_neg_sample), 2), ': 1')
    return df_under

def oversample(values, positive_class, negative_class, df_negatives, df_positives, output_file):
    df_pos_sample = DataFrame(df_positives.sample(len(df_negatives), replace=True))
    df_over = concat([df_pos_sample, df_negatives], axis=0)
    df_over.to_csv(f'data/{output_file}_over.csv', index=False)
    values['OverSample'] = [len(df_pos_sample), len(df_negatives)]
    print('Minority class=', positive_class, ':', len(df_pos_sample))
    print('Majority class=', negative_class, ':', len(df_negatives))
    print('Proportion:', round(len(df_pos_sample) / len(df_negatives), 2), ': 1')
    return df_over

def smote(data, class_var, values, positive_class, negative_class,output_file):
    RANDOM_STATE = 42
    smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
    y = data.pop(class_var).values
    X = data.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(data.columns) + [class_var]
    df_smote.to_csv(f'data/{output_file}_smote.csv', index=False)

    smote_target_count = Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count[positive_class], smote_target_count[negative_class]]
    print('Minority class=', positive_class, ':', smote_target_count[positive_class])
    print('Majority class=', negative_class, ':', smote_target_count[negative_class])
    print('Proportion:', round(smote_target_count[positive_class] / smote_target_count[negative_class], 2), ': 1')  

    return df_smote

#%% USE EACH STRATEGY TO COUNTER BALANCE
undersample(values_air, positive_class_air, negative_class_air, df_negatives_air, df_positives_air, "air_quality_tabular")
oversample(values_air, positive_class_air, negative_class_air, df_negatives_air, df_positives_air, "air_quality_tabular")
smote(air_data,AIR_CLASS, values_air, positive_class_air, negative_class_air, "air_quality_tabular")
figure()
multiple_bar_chart([positive_class_air, negative_class_air], values_air, title='Target', xlabel='frequency', ylabel='Class balance')
savefig(f'images/lab4/balancing/air_quality_balancing_techniques.png')
show()
# %%
undersample(values_col, positive_class_col, negative_class_col, df_negatives_col, df_positives_col, "NYC_collisions_tabular")
oversample(values_col, positive_class_col, negative_class_col, df_negatives_col, df_positives_col, "NYC_collisions_tabular")
smote(collisions_data,COLLISIONS_CLASS, values_col, positive_class_col, negative_class_col, "NYC_collisions_tabular")
figure()
multiple_bar_chart([positive_class_col, negative_class_col], values_col, title='Target', xlabel='frequency', ylabel='Class balance')
savefig(f'images/lab4/balancing/NYC_collisions_balancing_techniques.png')
show()
# %%
