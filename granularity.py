from pandas import read_csv, to_datetime
from ds_charts import HEIGHT
from matplotlib.pyplot import subplots, savefig, clf, figure, xticks
from pandas.plotting import register_matplotlib_converters
from ts_functions import plot_series
import config as cfg



def plot_histograms(data, bins, output_path):
	rows = len(data.columns)
	cols = len(bins)

	fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
	for i in range(rows):
	    for j in range(cols):
	        axs[i, j].set_title('Histogram for %s - %d bins'%(data.columns[i], bins[j]))
	        axs[i, j].set_xlabel(data.columns[i])
	        axs[i, j].set_ylabel('Nr records')
	        axs[i, j].hist(data[data.columns[i]].values.tolist(), bins=bins[j], edgecolor=cfg.LINE_COLOR, color=cfg.FILL_COLOR)

	savefig(output_path, dpi=300, bbox_inches="tight")
	clf() # cleanup


def plot_timeseries(data, output_path, y_label):
	# daily
	day_df = data.copy().groupby(data.index.date).mean()
	figure(figsize=(3*HEIGHT, HEIGHT))
	plot_series(day_df, title='Daily ' + y_label, x_label='timestamp', y_label=y_label)
	xticks(rotation = 45)
	savefig(output_path + 'daily.png', dpi=300, bbox_inches="tight")

	# weekly
	index = data.index.to_period('W')
	week_df = data.copy().groupby(index).mean()
	week_df['timestamp'] = index.drop_duplicates().to_timestamp()
	week_df.set_index('timestamp', drop=True, inplace=True)
	figure(figsize=(3*HEIGHT, HEIGHT))
	plot_series(week_df, title='Weekly ' + y_label, x_label='timestamp', y_label=y_label)
	xticks(rotation = 45)
	savefig(output_path + 'weekly.png', dpi=300, bbox_inches="tight")

	# monthly
	index = data.index.to_period('M')
	month_df = data.copy().groupby(index).mean()
	month_df['timestamp'] = index.drop_duplicates().to_timestamp()
	month_df.set_index('timestamp', drop=True, inplace=True)
	figure(figsize=(3*HEIGHT, HEIGHT))
	plot_series(month_df, title='Monthly ' + y_label, x_label='timestamp', y_label=y_label)
	savefig(output_path + 'monthly.png', dpi=300, bbox_inches="tight")

	clf() # cleanup


def main():
	register_matplotlib_converters()


	# ------- GRANULARITY FOR TABULAR DATA ------- #
	
	# 'Air Quality'
	data = read_csv('data/air_quality_tabular.csv', index_col='FID', na_values='', parse_dates=True, infer_datetime_format=True)
	bins = (10, 100)
	output_path = 'images/lab1/granularity/air_quality_tabular_data_granularity.png'
	#plot_histograms(data, bins, output_path)

	# 'NYC Collisions'
	data = read_csv('data/NYC_collisions_tabular.csv', index_col='COLLISION_ID', na_values='', parse_dates=True, infer_datetime_format=True)
	bins = (10, 100, 1000)
	output_path = 'images/lab1/granularity/NYC_collisions_tabular_data_granularity.png'
	#plot_histograms(data, bins, output_path)


	# ------- GRANULARITY FOR TIME SERIES ------- #

	# 'Air Quality'
	data = read_csv('data/air_quality_timeseries.csv', index_col='DATE', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
	data.index.rename('timestamp', inplace=True)
	data.index = to_datetime(data.index)
	output_path = 'images/lab1/granularity/air_quality_timeseries_granularity_'
	plot_timeseries(data, output_path, 'air quality')

	# 'NYC Collisions'
	data = read_csv('data/NYC_collisions_timeseries.csv', index_col='timestamp', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
	data.drop(data.tail(1).index, inplace=True)
	data.index = to_datetime(data.index)
	output_path = 'images/lab1/granularity/NYC_collisions_tabular_data_granularity_'
	plot_timeseries(data, output_path, 'nr collisions')


if __name__ == '__main__':
	main()