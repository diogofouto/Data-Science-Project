import pandas as pd
from seaborn import heatmap
import numpy as np
import config as cfg
from matplotlib.pyplot import title, savefig, clf, figure, xticks, subplots, tight_layout
from pandas.plotting import register_matplotlib_converters
from ds_charts import bar_chart, get_variable_types, HEIGHT


AQ_VAR_THRESHOLD = 1.5
AQ_CORR_THRESHOLD = "TODO"

NYC_VAR_THRESHOLD = 0.05
NYC_CORR_THRESHOLD = 0.9


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


def get_variable_correlation(corr_mtx, titl):
	if corr_mtx.empty:
		raise ValueError('Matrix is empty.')

	figure(figsize=[10, 10])
	heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, linecolor=cfg.LINE_COLOR, cmap='Blues')
	title('Filtered Correlation Analysis')
	savefig('images/lab6/feature_selection/' + titl +'_correlation_analysis.png', dpi=300, bbox_inches="tight")
	clf() # cleanup


def drop_redundant(data: pd.DataFrame, vars_2drop: dict) -> pd.DataFrame:
	sel_2drop = []
	#print(vars_2drop.keys())
	for key in vars_2drop.keys():
		if key not in sel_2drop:
			for r in vars_2drop[key]:
				if r != key and r not in sel_2drop:
					sel_2drop.append(r)
	#print('Variables to drop', sel_2drop)
	df = data.copy()
	for var in sel_2drop:
		df.drop(labels=var, axis=1, inplace=True)
	return df


def select_low_variance(data: pd.DataFrame, threshold: float, titl) -> list:
	lst_variables = []
	lst_variances = []
	for el in data.columns:
		value = data[el].var()
		if value < threshold:
			lst_variables.append(el)
			lst_variances.append(value)

	#print(len(lst_variables), lst_variables)
	#figure(figsize=[10, 4])
	#bar_chart(lst_variables, lst_variances, title='Variance analysis', xlabel='variables', ylabel='variance', rotation=45)
	#savefig('images/lab6/feature_selection/' + titl + '_variance_analysis.png', dpi=300, bbox_inches="tight")
	#clf() # cleanup
	return lst_variables

def drop_redundant_list(data: pd.DataFrame, vars_2drop) -> pd.DataFrame:
	df = data.copy()
	for var in vars_2drop:
		df.drop(labels=var, axis=1, inplace=True)
	return df


def main():
	register_matplotlib_converters()

	# ------- VARIANCE ANALYSIS ------- #

	# AQ
	# train
	data = pd.read_csv('data/air_quality_train_smote.csv', na_values='?')
	numeric = get_variable_types(data)['Numeric']
	vars_2drop = select_low_variance(data[numeric], 1.5, 'air_quality')
	#print(vars_2drop)
	data = drop_redundant_list(data, vars_2drop)
	data.to_csv(f'data/air_quality_train_smote_no_low_variance_vars.csv', index=False)
	# test
	data = pd.read_csv('data/air_quality_test.csv', na_values='?')
	data = drop_redundant_list(data, vars_2drop)
	data.to_csv(f'data/air_quality_test_no_low_variance_vars.csv', index=False)


	# NYC
	# train
	data = pd.read_csv('data/NYC_collisions_train_smote.csv', na_values='?')
	numeric = get_variable_types(data)['Numeric']
	vars_2drop = select_low_variance(data[numeric], 0.05, 'nyc_collisions')
	#print(vars_2drop)
	data = drop_redundant_list(data, vars_2drop)
	data.to_csv(f'data/NYC_collisions_train_smote_no_low_variance_vars.csv', index=False)
	# test
	data = pd.read_csv('data/NYC_collisions_test.csv', na_values='?')
	data = drop_redundant_list(data, vars_2drop)
	data.to_csv(f'data/NYC_collisions_test_no_low_variance_vars.csv', index=False)

	
	# ------- CORRELATION ANALYSIS ------- #

	# datasets
	aq_data_train = pd.read_csv('data/air_quality_train_smote_no_low_variance_vars.csv', na_values='?')
	aq_data_test = pd.read_csv('data/air_quality_test_no_low_variance_vars.csv', na_values='?')

	nyc_data_train = pd.read_csv('data/NYC_collisions_train_smote_no_low_variance_vars.csv', na_values='?')
	nyc_data_test = pd.read_csv('data/NYC_collisions_test_no_low_variance_vars.csv', na_values='?')

	for corr_thres in [ .1*x for x in range(0, 10)]:

		# AQ
		# train
		drop, corr_mtx = get_redundant_variables(aq_data_train.corr(), corr_thres)
		get_variable_correlation(corr_mtx, 'air_quality_' + str(corr_thres))
		df = drop_redundant(aq_data_train, drop)
		df.to_csv(f'data/air_quality_train_smote_no_low_variance_vars_no_correlated_vars_{corr_thres}.csv', index=False)
		# test
		df = drop_redundant(aq_data_test, drop)
		df.to_csv(f'data/air_quality_test_no_low_variance_vars_no_correlated_vars_{corr_thres}.csv', index=False)


		# NYC
		# train
		drop, corr_mtx = get_redundant_variables(nyc_data_train.corr(), corr_thres)
		get_variable_correlation(corr_mtx, 'nyc_collisions_' + str(corr_thres))
		df = drop_redundant(nyc_data_train, drop)
		df.to_csv(f'data/NYC_collisions_train_smote_no_low_variance_vars_no_correlated_vars_{corr_thres}.csv', index=False)
		# test
		df = drop_redundant(nyc_data_test, drop)
		df.to_csv(f'data/NYC_collisions_test_no_low_variance_vars_no_correlated_vars_{corr_thres}.csv', index=False)


	# ------- CHI-SQUARE ANALYSIS ------- #
	# TODO
	

if __name__ == '__main__':
	main()
