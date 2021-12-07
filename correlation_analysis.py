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
	sns.heatmap(abs(corr_mtx),  ax=ax, linecolor=cfg.LINE_COLOR, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, cmap='Blues', fmt='.1f', annot=True)
	
	if ax is None:
		title(name)
		xticks(rotation=45)
		savefig(path, dpi=300, bbox_inches="tight")
		clf() # cleanup
	else:
		ax.set_title(name)
		ax.tick_params(axis='x', rotation=45)


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
	air_quality_classes = {"EN": ["City_EN", "Prov_EN", "GbCity", "GbProv"], "CO": ["CO_Mean","CO_Min","CO_Max","CO_Std"],
							"NO2": ["NO2_Mean","NO2_Min","NO2_Max","NO2_Std"], "O3": ["O3_Mean","O3_Min","O3_Max","O3_Std"],
							"PM2.5": ["PM2.5_Mean","PM2.5_Min","PM2.5_Max","PM2.5_Std"], "PM10": ["PM10_Mean","PM10_Min","PM10_Max","PM10_Std"],
							"SO2": ["SO2_Mean","SO2_Min","SO2_Max","SO2_Std"]}

	# create dataframe and subplots
	temp = pd.read_csv('data/air_quality_tabular.csv', index_col='FID', parse_dates=True, infer_datetime_format=True)
	data = temp.apply(lambda x: pd.factorize(x)[0]) # Convert all var types to numerical
	fig, axs = subplots(len(air_quality_classes), 1, figsize=(HEIGHT, len(air_quality_classes)*HEIGHT), squeeze=False)
	
	# get correlations
	i = 0
	for c in air_quality_classes:
		temp = data.drop(columns=[col for col in data if col not in air_quality_classes[c]])
		get_correlations(temp, 'Correlation Analysis for ' + c + '\n', ax=axs[i,0])
		i=i+1

	# save
	tight_layout(h_pad=4)
	savefig('images/lab1/correlation/air_quality_correlation_analysis_for_classes.png', dpi=300, bbox_inches="tight")
	clf() # cleanup


	# 'NYC Collisions'
	# todo


if __name__ == '__main__':
	main()