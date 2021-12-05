import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.pyplot import title, savefig, clf
from pandas.plotting import register_matplotlib_converters


def get_correlations(data, name, path):
	correlation_matrix = data.corr()
	sns.heatmap(correlation_matrix,  xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns, cmap='Blues', fmt='.3f')
	title(name)
	savefig(path, dpi=300, bbox_inches="tight")
	clf()


def main():
	register_matplotlib_converters()

	# Get correlations for 'Air Quality'
	temp = pd.read_csv('data/air_quality_tabular.csv', index_col='FID', na_values='', parse_dates=True, infer_datetime_format=True)
	data = temp.apply(lambda x: pd.factorize(x)[0]) # Convert all var types to numerical
	get_correlations(data, 'Correlation Analysis for Air Quality\n', 'images/lab1/correlation/air_quality_correlation_analysis.png')


	# Get correlations for 'NYC Collisions'
	temp = pd.read_csv('data/NYC_collisions_tabular.csv', index_col='COLLISION_ID', na_values='', parse_dates=True, infer_datetime_format=True)
	data = temp.apply(lambda x: pd.factorize(x)[0]) # Convert all var types to numerical
	get_correlations(data, 'Correlation Analysis for NYC Collisions\n', 'images/lab1/correlation/NYC_collisions_correlation_analysis.png')


if __name__ == '__main__':
	main()