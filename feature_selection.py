import pandas as pd
from seaborn import heatmap
import numpy as np
import config as cfg
from matplotlib.pyplot import title, savefig, clf, figure, xticks, subplots, tight_layout
from pandas.plotting import register_matplotlib_converters
from ds_charts import bar_chart, get_variable_types, HEIGHT


THRESHOLD = 0.9


def get_redundant_variables(corr_mtx, threshold: float) -> tuple[dict, pd.DataFrame]:
    if corr_mtx.empty:
        return {}

    corr_mtx = abs(corr_mtx)
    vars_to_drop = {}
    for el in corr_mtx.columns:
        el_corr = (corr_mtx[el]).loc[corr_mtx[el] >= threshold]
        if len(el_corr) == 1:
            corr_mtx.drop(labels=el, axis=1, inplace=True)
            corr_mtx.drop(labels=el, axis=0, inplace=True)
        else:
            vars_to_drop[el] = el_corr.index
    return vars_to_drop, corr_mtx


def get_variable_correlation(corr_mtx, title):
	if corr_mtx.empty:
    	raise ValueError('Matrix is empty.')

	figure(figsize=[10, 10])
	heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, linecolor=cfg.LINE_COLOR, cmap='Blues')
	title('Filtered Correlation Analysis')
	savefig(f'images/lab6/feature_selection/' + title +'_filtered_correlation_analysis_{THRESHOLD}.png', dpi=300, bbox_inches="tight")
	clf() # cleanup


def drop_redundant(data: pd.DataFrame, vars_2drop: dict) -> pd.DataFrame:
    sel_2drop = []
    print(vars_2drop.keys())
    for key in vars_2drop.keys():
        if key not in sel_2drop:
            for r in vars_2drop[key]:
                if r != key and r not in sel_2drop:
                    sel_2drop.append(r)
    print('Variables to drop', sel_2drop)
    df = data.copy()
    for var in sel_2drop:
        df.drop(labels=var, axis=1, inplace=True)
    return df


def select_low_variance(data: pd.DataFrame, threshold: float) -> list:
    lst_variables = []
    lst_variances = []
    for el in data.columns:
        value = data[el].var()
        if value >= threshold:
            lst_variables.append(el)
            lst_variances.append(value)

    print(len(lst_variables), lst_variables)
    figure(figsize=[10, 4])
    bar_chart(lst_variables, lst_variances, title='Variance analysis', xlabel='variables', ylabel='variance')
    savefig('images/lab6/feature_selection/' + title + '_filtered_variance_analysis.png', dpi=300, bbox_inches="tight")
    clf() # cleanup
    return lst_variables


def main():
	register_matplotlib_converters()


	# ------- CORRELATION FOR ALL VARIABLES ------- #

	# 'Air Quality'
	data = pd.read_csv('data/air_quality_tabular.csv', index_col='FID', parse_dates=True, infer_datetime_format=True)
	
	drop, corr_mtx = get_redundant_variables(data.corr(), THRESHOLD)
	print(drop.keys())
	get_variable_correlation(corr_mtx, 'air_quality')
	df = drop_redundant(data, drop)

	# A partir de aqui usa-se df ou data? 

	numeric = get_variable_types(data)['Numeric']
	vars_2drop = select_low_variance(data[numeric], 0.1)
	print(vars_2drop)

	# E quais são as verdadeiras vars_to_drop?


	# 'NYC Collisions'
	data = pd.read_csv('data/NYC_collisions_tabular.csv', index_col='COLLISION_ID', parse_dates=True, infer_datetime_format=True)

	drop, corr_mtx = get_redundant_variables(data.corr(), THRESHOLD)
	print(drop.keys())
	get_variable_correlation(corr_mtx, 'air_quality')
	df = drop_redundant(data, drop)

	numeric = get_variable_types(data)['Numeric']
	vars_2drop = select_low_variance(data[numeric], 0.1)
	print(vars_2drop)


if __name__ == '__main__':
	main()