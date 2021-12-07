import pandas as pd
import seaborn as sns
import numpy as np
import config as cfg
from matplotlib.pyplot import title, savefig, clf, figure, xticks
from pandas.plotting import register_matplotlib_converters


def get_correlations(data, name, path):
	figure(figsize=[14, 14])
	corr_mtx = data.corr()
	sns.heatmap(abs(corr_mtx),  linecolor=cfg.LINE_COLOR, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, cmap='Blues', fmt='.1f', annot=True)
	title(name)
	xticks(rotation=45)
	savefig(path, dpi=300, bbox_inches="tight")
	clf() # cleanup


def main():
	register_matplotlib_converters()

	# Get correlations for 'Air Quality'
	temp = pd.read_csv('data/air_quality_tabular.csv', index_col='FID', parse_dates=True, infer_datetime_format=True)
	data = temp.apply(lambda x: pd.factorize(x)[0]) # Convert all var types to numerical
	get_correlations(data, 'Correlation Analysis for Air Quality\n', 'images/lab1/correlation/air_quality_correlation_analysis.png')


	# Get correlations for 'NYC Collisions'
	temp = pd.read_csv('data/NYC_collisions_tabular.csv', index_col='COLLISION_ID', parse_dates=True, infer_datetime_format=True)
	data = temp.apply(lambda x: pd.factorize(x)[0]) # Convert all var types to numerical
	get_correlations(data, 'Correlation Analysis for NYC Collisions\n', 'images/lab1/correlation/NYC_collisions_correlation_analysis.png')


if __name__ == '__main__':
	main()