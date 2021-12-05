from pandas import read_csv
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import subplots, savefig


def plot_histograms(data, bins):
	rows = len(data.columns)
	cols = len(bins)

	fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT))
	for i in range(rows):
	    for j in range(cols):
	        axs[i, j].set_title('Histogram for %s - %d bins'%(data.columns[i], bins[j]))
	        axs[i, j].set_xlabel(data.columns[i])
	        axs[i, j].set_ylabel('Nr records')
	        axs[i, j].hist(data[data.columns[i]].values.tolist(), bins=bins[j])


def main():
	# Get granularity for 'Air Quality'
	data = read_csv('data/air_quality_tabular.csv', index_col='FID', na_values='', parse_dates=False, infer_datetime_format=False)
	bins = (10, 100)
	plot_histograms(data, bins)
	savefig('images/lab1/granularity/air_quality_granularity_study.png', dpi=300, bbox_inches = "tight")

	# Get granularity for 'NYC Collisions'
	data = read_csv('data/NYC_collisions_tabular.csv', index_col='COLLISION_ID', na_values='', parse_dates=False, infer_datetime_format=False)
	bins = (10, 100, 1000)
	plot_histograms(data, bins)
	savefig('images/lab1/granularity/NYC_collisions_granularity_study.png', dpi=300, bbox_inches = "tight")


if __name__ == '__main__':
	main()