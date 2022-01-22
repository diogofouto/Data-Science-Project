import pandas as pd
import numpy as np
import config as cfg

from seaborn import heatmap
from matplotlib.pyplot import title, savefig, clf, figure, xticks, subplots, tight_layout
from pandas.plotting import register_matplotlib_converters
from ds_charts import bar_chart, get_variable_types, HEIGHT
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


AQ_VAR_THRESHOLD = 2.0 # best
AQ_CORR_THRESHOLD = 0.4
AQ_K_BEST = 4

NYC_VAR_THRESHOLD = 0.00
NYC_CORR_THRESHOLD = 0.8 # best
NYC_K_BEST = 4

AQ_TARGET = 'ALARM'
NYC_TARGET = 'PERSON_INJURY'


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

	#figure(figsize=[10, 10])
	#heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, linecolor=cfg.LINE_COLOR, cmap='Blues')
	#title('Filtered Correlation Analysis')
	#savefig('images/lab6/feature_selection/' + titl +'_correlation_analysis.png', dpi=300, bbox_inches="tight")
	#clf() # cleanup


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

	print(len(lst_variables), lst_variables)
	#figure(figsize=[10, 4])
	#bar_chart(lst_variables, lst_variances, title='Variance analysis', xlabel='variables', ylabel='variance', rotation=45)
	#savefig(f'images/lab6/feature_selection/{titl}_variance_analysis_{threshold}.png', dpi=300, bbox_inches="tight")
	#clf() # cleanup
	return lst_variables

def drop_redundant_list(data: pd.DataFrame, vars_2drop) -> pd.DataFrame:
	df = data.copy()
	for var in vars_2drop:
		df.drop(labels=var, axis=1, inplace=True)
	return df


def chi2_feature_selection(data, target, k_value=5):
	df = data.copy()
	test = SelectKBest(f_classif, k=k_value)
	fit = test.fit_transform(df, df[target])
	relevant_cols = [column[0]  for column in zip(df.columns,test.get_support()) if column[1]]
	cols_to_drop = [c for c in df.columns.values if c not in relevant_cols]
	return cols_to_drop


def main():
	register_matplotlib_converters()

	# ------- VARIANCE ANALYSIS ------- #
	"""

	# datasets
	aq_data_train = pd.read_csv('data/air_quality_scaled_zscore.csv', na_values='?')
	aq_data_train_numeric = get_variable_types(aq_data_train)['Numeric']
	aq_data_test = pd.read_csv('data/air_quality_test.csv', na_values='?')

	nyc_data_train = pd.read_csv('data/NYC_collisions_scaled_minmax.csv', na_values='?')
	nyc_data_train_numeric = get_variable_types(nyc_data_train)['Numeric']
	nyc_data_test = pd.read_csv('data/NYC_collisions_test.csv', na_values='?')

	# AQ
	# train
	vars_2drop = select_low_variance(aq_data_train[aq_data_train_numeric], 2, 'air_quality')
	df = drop_redundant_list(aq_data_train, vars_2drop)
	df.to_csv(f'data/air_quality_scaled_zscore_fs.csv', index=False)
	# test
	#df = drop_redundant_list(aq_data_test, vars_2drop)
	#df.to_csv(f'data/NYC_collisions_scaled_minmax_fs.csv', index=False)


	# NYC
	# train
	#vars_2drop = select_low_variance(nyc_data_train[nyc_data_train_numeric], var_thres, 'NYC_collisions')
	#df = drop_redundant_list(nyc_data_train, vars_2drop)
	#df.to_csv(f'data/NYC_collisions_scaled_minmax_fs.csv', index=False)
	# test
	#df = drop_redundant_list(nyc_data_test, vars_2drop)
	#df.to_csv(f'data/NYC_collisions_test_fs.csv', index=False)


	"""
	# ------- CORRELATION ANALYSIS ------- #

	# datasets
	aq_data_train = pd.read_csv('data/air_quality_train_smote_no_low_variance_vars_2.0.csv', na_values='?')
	aq_data_test = pd.read_csv('data/air_quality_test_no_low_variance_vars_2.0.csv', na_values='?')

	nyc_data_train = pd.read_csv('data/NYC_collisions_scaled_minmax.csv', na_values='?')
	nyc_data_test = pd.read_csv('data/NYC_collisions_test.csv', na_values='?')


	# NYC
	# train
	drop, corr_mtx = get_redundant_variables(nyc_data_train.corr(), 0.8)
	get_variable_correlation(corr_mtx, 'nyc_collisions_' + str(0.8))
	df = drop_redundant(nyc_data_train, drop)
	df.to_csv(f'data/NYC_collisions_scaled_minmax_fs.csv', index=False)
	# test


	# ------- F-CLASSIFICATION ANALYSIS ------- #
	"""

	aq_target_to_binary = {'Safe': 1,'Danger': 0}
	nyc_target_to_binary = {'Injured': 1,'Killed': 0}

	# datasets
	aq_data_train = pd.read_csv('data/air_quality_train_smote.csv', na_values='?')
	aq_data_test = pd.read_csv('data/air_quality_test.csv', na_values='?')
	aq_data_train[AQ_TARGET] = [aq_target_to_binary[item] for item in aq_data_train[AQ_TARGET]]
	aq_data_test[AQ_TARGET] = [aq_target_to_binary[item] for item in aq_data_test[AQ_TARGET]]
	aq_data_train.to_csv('data/air_quality_train_smote.csv', index=False)
	aq_data_test.to_csv('data/air_quality_test.csv', index=False)

	nyc_data_train = pd.read_csv('data/NYC_collisions_train_smote.csv', na_values='?')
	nyc_data_test = pd.read_csv('data/NYC_collisions_test.csv', na_values='?')
	nyc_data_train[NYC_TARGET] = [nyc_target_to_binary[item] for item in nyc_data_train[NYC_TARGET]]
	nyc_data_test[NYC_TARGET] = [nyc_target_to_binary[item] for item in nyc_data_test[NYC_TARGET]]
	nyc_data_train.to_csv('data/NYC_collisions_train_smote.csv', index=False)
	nyc_data_test.to_csv('data/NYC_collisions_test.csv', index=False)
	
	
	for k in range(3, 20):
		# AQ
		# train
		cols_to_drop = chi2_feature_selection(aq_data_train, AQ_TARGET, k)
		df = drop_redundant_list(aq_data_train, cols_to_drop)
		df.to_csv(f'data/air_quality_train_smote_f_classif_k={k}.csv', index=False)
		# test
		df = drop_redundant_list(aq_data_test, cols_to_drop)
		df.to_csv(f'data/air_quality_test_f_classif_k={k}.csv', index=False)

		# NYC
		# train
		cols_to_drop = chi2_feature_selection(nyc_data_train, NYC_TARGET, k)
		df = drop_redundant_list(nyc_data_train, cols_to_drop)
		df.to_csv(f'data/NYC_collisions_train_smote_f_classif_k={k}.csv', index=False)
		# test
		df = drop_redundant_list(nyc_data_test, cols_to_drop)
		df.to_csv(f'data/NYC_collisions_test_f_classif_k={k}.csv', index=False)
	"""
	

if __name__ == '__main__':
	main()