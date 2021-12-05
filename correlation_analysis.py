import pandas as pd
import seaborn as sns
from matplotlib.pyplot import show, savefig

def main():
	# Get correlations for 'Air Quality'
	temp = pd.read_csv('data/air_quality_tabular.csv', index_col='FID', na_values='', parse_dates=False, infer_datetime_format=False)
	data = temp.apply(lambda x: pd.factorize(x)[0]) # Convert all var types to numerical
	sns.heatmap(data.corr()).set(title="Correlation matrix for Air Quality\n")
	savefig('images/lab1/correlation/air_quality_correlation_analysis.png', dpi=300, bbox_inches = "tight")
	show()


	# Get correlations for 'NYC Collisions'
	temp = pd.read_csv('data/NYC_collisions_tabular.csv', index_col='COLLISION_ID', na_values='', parse_dates=False, infer_datetime_format=False)
	data = temp.apply(lambda x: pd.factorize(x)[0]) # Convert all var types to numerical
	sns.heatmap(data.corr()).set(title="Correlation matrix for NYC Collisions\n")
	savefig('images/lab1/correlation/NYC_collisions_correlation_analysis.png', dpi=300, bbox_inches = "tight")
	show()


if __name__ == '__main__':
	main()