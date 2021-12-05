from pandas import read_csv
from ds_charts import HEIGHT
from matplotlib.pyplot import subplots, savefig, clf
from pandas.plotting import register_matplotlib_converters
import config as cfg



def plot_histograms(data, bins, path):
	rows = len(data.columns)
	cols = len(bins)

	fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
	for i in range(rows):
	    for j in range(cols):
	        axs[i, j].set_title('Histogram for %s - %d bins'%(data.columns[i], bins[j]))
	        axs[i, j].set_xlabel(data.columns[i])
	        axs[i, j].set_ylabel('Nr records')
	        axs[i, j].hist(data[data.columns[i]].values.tolist(), bins=bins[j], edgecolor=cfg.LINE_COLOR, color=cfg.FILL_COLOR)

	savefig(path, dpi=300, bbox_inches="tight")
	clf()


def main():
	register_matplotlib_converters()
	
	# Get granularity for 'Air Quality'
	data = read_csv('data/air_quality_tabular.csv', index_col='FID', na_values='', parse_dates=True, infer_datetime_format=True)
	bins = (10, 100)
	path = 'images/lab1/granularity/air_quality_granularity_study.png'
	plot_histograms(data, bins, path)

	# Get granularity for 'NYC Collisions'
	data = read_csv('data/NYC_collisions_tabular.csv', index_col='COLLISION_ID', na_values='', parse_dates=True, infer_datetime_format=True)
	bins = (10, 100, 1000)
	path = 'images/lab1/granularity/NYC_collisions_granularity_study.png'
	plot_histograms(data, bins, path)


if __name__ == '__main__':
	main()