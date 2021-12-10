import pandas as pd
import seaborn as sns
import numpy as np
import config as cfg
from matplotlib.pyplot import title, savefig, clf, figure, xticks, subplots, tight_layout
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, HEIGHT


def get_correlations(data, name, path=None, ax=None):
	if ax is None:
		figure(figsize=[14, 14])

	corr_mtx = data.corr()
	sns.heatmap(abs(corr_mtx),  ax=ax, linecolor=cfg.LINE_COLOR, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, cmap='Blues', fmt='.2f', annot=True, annot_kws={"size": 6})
	
	if ax is None:
		title(name)
		xticks(rotation=45)
		savefig(path, dpi=300, bbox_inches="tight")
		clf() # cleanup
	else:
		ax.set_title(name)
		ax.tick_params(axis='x', rotation=45)


def get_class_correlations(data, clas, path):
	variables = get_variable_types(data)['Numeric']
	variables = [item for item in variables if 'ID' not in item]
	if [] == variables:
	    raise ValueError('There are no numeric variables.')

	rows = len(variables)

	fig, axs = subplots(rows, 1, figsize=(1*HEIGHT, rows*HEIGHT), squeeze=False)
	for i in range(rows):
		if 'ID' in variables[i]:
			continue
		axs[i, 0].set_title(clas + ' x ' + variables[i])
		corr_mtx = data[clas].corr(data[variables[i]])
		sns.heatmap([[abs(corr_mtx),],], ax=axs[i,0], xticklabels=False, yticklabels=False, linecolor=cfg.LINE_COLOR, cmap='Blues', fmt='.2f', annot=True, annot_kws={"size": 6})
		axs[i, 0].set_xlabel(variables[i])
		axs[i, 0].set_ylabel(clas)

	tight_layout(h_pad=2)
	savefig(path, dpi=300, bbox_inches="tight")
	clf() # cleanup


def main():
	register_matplotlib_converters()


	# ------- CORRELATION FOR ALL VARIABLES ------- #

	# 'Air Quality'
	temp = pd.read_csv('data/air_quality_tabular.csv', index_col='FID', parse_dates=True, infer_datetime_format=True)
	data = temp.apply(lambda x: pd.factorize(x)[0]) # Convert all var types to numerical
	get_correlations(data, 'Correlation Analysis for Air Quality\n', 'images/lab1/correlation/air_quality_correlation_analysis.png')


	# 'NYC Collisions'
	temp = pd.read_csv('data/NYC_collisions_tabular.csv', index_col='COLLISION_ID', parse_dates=True, infer_datetime_format=True)
	data = temp.apply(lambda x: pd.factorize(x)[0]) # Convert all var types to numerical
	get_correlations(data, 'Correlation Analysis for NYC Collisions\n', 'images/lab1/correlation/NYC_collisions_correlation_analysis.png')


	# ------- CORRELATION WITHIN CLASSES ------- #

	# 'Air Quality'
	temp = pd.read_csv('data/air_quality_tabular.csv', index_col='FID', parse_dates=True, infer_datetime_format=True)
	data = temp.apply(lambda x: pd.factorize(x)[0]) # Convert all var types to numerical
	get_class_correlations(data, 'ALARM', 'images/lab1/correlation/air_quality_correlation_analysis_for_class.png')

	# 'NYC Collisions'
	temp = pd.read_csv('data/NYC_collisions_tabular.csv', index_col='COLLISION_ID', parse_dates=True, infer_datetime_format=True)
	data = temp.apply(lambda x: pd.factorize(x)[0]) # Convert all var types to numerical
	get_class_correlations(data, 'PERSON_INJURY', 'images/lab1/correlation/NYC_collisions_correlation_analysis_for_class.png')



if __name__ == '__main__':
	main()