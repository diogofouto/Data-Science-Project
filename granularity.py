from pandas import read_csv, to_datetime
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import subplots, savefig, clf, figure, xticks, tight_layout
from pandas.plotting import register_matplotlib_converters
from ts_functions import plot_series
import config as cfg



def plot_histograms(data, bins, output_path):
	variables = get_variable_types(data)['Numeric']
	if [] == variables:
	    raise ValueError('There are no numeric variables.')

	rows = len(variables)
	#rows = len(data.columns)
	cols = len(bins)

	fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
	for i in range(rows):
		for j in range(cols):
			axs[i, j].set_title('Histogram for %s - %d bins'%(variables[i], bins[j]))
			#axs[i, j].set_title('Histogram for %s - %d bins'%(data.columns[i], bins[j]))
			axs[i, j].set_xlabel(variables[i])
			#axs[i, j].set_xlabel(data.columns[i])
			axs[i, j].set_ylabel('Nr records')
			axs[i, j].hist(data[variables[i]].values, bins=bins[j], edgecolor=cfg.LINE_COLOR, color=cfg.FILL_COLOR)
			#axs[i, j].hist(data[data.columns[i]].values.tolist(), bins=bins[j], edgecolor=cfg.LINE_COLOR, color=cfg.FILL_COLOR)
			axs[i, j].tick_params(axis='x', rotation=45)
			if variables[i] in ['date',]: # 'City_EN', 'GbCity'
				axs[i, j].set_xticks(axs[i, j].get_xticks()[::30])
			"""
			if data.columns[i] in ['Prov_EN',]:
				axs[i, j].set_xticks(axs[i, j].get_xticks()[::5])
			"""

	tight_layout(h_pad=2)
	savefig(output_path, dpi=300, bbox_inches="tight")
	clf() # cleanup


def plot_timeseries(data, output_path, y_label):
	fig, axs = subplots(4, 1, figsize=(2*HEIGHT, 4*HEIGHT), squeeze=False)

	# daily
	day_df = data.copy().groupby(data.index.date).mean()
	plot_series(day_df, ax=axs[0, 0], title='Daily ' + y_label, x_label='timestamp', y_label=y_label)
	axs[0, 0].tick_params(axis='x', rotation=45)

	# weekly
	index = data.index.to_period('W')
	week_df = data.copy().groupby(index).mean()
	week_df['timestamp'] = index.drop_duplicates().to_timestamp()
	week_df.set_index('timestamp', drop=True, inplace=True)
	plot_series(week_df, ax=axs[1, 0], title='Weekly ' + y_label, x_label='timestamp', y_label=y_label)
	axs[1, 0].tick_params(axis='x', rotation=45)

	# monthly
	index = data.index.to_period('M')
	month_df = data.copy().groupby(index).mean()
	month_df['timestamp'] = index.drop_duplicates().to_timestamp()
	month_df.set_index('timestamp', drop=True, inplace=True)
	plot_series(month_df, ax=axs[2, 0], title='Monthly ' + y_label, x_label='timestamp', y_label=y_label)
	axs[2, 0].tick_params(axis='x', rotation=45)

	# quarterly
	index = data.index.to_period('Q')
	quarter_df = data.copy().groupby(index).mean()
	quarter_df['timestamp'] = index.drop_duplicates().to_timestamp()
	quarter_df.set_index('timestamp', drop=True, inplace=True)
	plot_series(quarter_df, ax=axs[3, 0], title='Quarterly ' + y_label, x_label='timestamp', y_label=y_label)
	axs[3, 0].tick_params(axis='x', rotation=45)

	tight_layout(h_pad=4)
	savefig(output_path, dpi=300, bbox_inches="tight")
	clf() # cleanup


def main():
	register_matplotlib_converters()


	# ------- GRANULARITY FOR TABULAR DATA ------- #
	
	# 'Air Quality'
	data = read_csv('data/air_quality_tabular.csv', index_col='FID', na_values='', parse_dates=True, infer_datetime_format=True)
	bins = (10, 100)
	output_path = 'images/lab1/granularity/air_quality_tabular_data_granularity.png'
	plot_histograms(data, bins, output_path)

	# 'NYC Collisions'
	data = read_csv('data/NYC_collisions_tabular.csv', index_col='COLLISION_ID', na_values='', parse_dates=True, infer_datetime_format=True)
	bins = (10, 100, 1000)
	output_path = 'images/lab1/granularity/NYC_collisions_tabular_data_granularity.png'
	plot_histograms(data, bins, output_path)


	# ------- GRANULARITY FOR TIME SERIES ------- #

	# 'Air Quality'
	data = read_csv('data/air_quality_timeseries.csv', index_col='DATE', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
	data.index.rename('timestamp', inplace=True)
	data.index = to_datetime(data.index)
	output_path = 'images/lab1/granularity/air_quality_timeseries_granularity.png'
	plot_timeseries(data, output_path, 'air quality')

	# 'NYC Collisions'
	data = read_csv('data/NYC_collisions_timeseries.csv', index_col='timestamp', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
	data.drop(data.tail(1).index, inplace=True)
	data.index = to_datetime(data.index)
	output_path = 'images/lab1/granularity/NYC_collisions_timeseries_granularity.png'
	plot_timeseries(data, output_path, 'nr of collisions')


if __name__ == '__main__':
	main()