import pandas as pd
import config as cfg
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.pyplot import subplots, savefig, tight_layout, clf


def scale(data, file, output_path):
	# Check variables
	variables = get_variable_types(data)
	numeric_vars = variables['Numeric']
	symbolic_vars = variables['Symbolic']
	boolean_vars = variables['Binary']

	data_sb = pd.DataFrame()
	data_bool = pd.DataFrame()

	if [] != numeric_vars:
		data_nr = data[numeric_vars]
	else:
		raise ValueError('There are no numeric variables.')

	if [] != symbolic_vars:
		data_sb = data[symbolic_vars]

	if [] != boolean_vars:
		data_bool = data[boolean_vars]

	# Scale with StandardScaler
	transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data_nr)
	tmp = pd.DataFrame(transf.transform(data_nr), index=data.index, columns= numeric_vars)
	norm_data_zscore = pd.concat([tmp, data_sb,  data_bool], axis=1)
	# Save file
	norm_data_zscore.to_csv(f'data/{file}_scaled_zscore.csv', index=False)

	# Scale with MinMaxScaler
	transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data_nr)
	tmp = pd.DataFrame(transf.transform(data_nr), index=data.index, columns= numeric_vars)
	norm_data_minmax = pd.concat([tmp, data_sb,  data_bool], axis=1)
	# Save file
	norm_data_minmax.to_csv(f'data/{file}_scaled_minmax.csv', index=False)
	# print(norm_data_minmax.describe())

	# Get boxplot charts
	fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
	axs[0, 0].set_title('Original data')
	data.boxplot(ax=axs[0, 0])
	axs[0, 0].tick_params(axis='x', rotation=45)
	
	axs[0, 1].set_title('Z-score normalization')
	norm_data_zscore.boxplot(ax=axs[0, 1])
	axs[0, 1].tick_params(axis='x', rotation=45)
	
	axs[0, 2].set_title('MinMax normalization')
	norm_data_minmax.boxplot(ax=axs[0, 2])
	axs[0, 2].tick_params(axis='x', rotation=45)

	# Save figs
	tight_layout(h_pad=2)
	savefig(output_path, dpi=300, bbox_inches="tight")
	clf() # cleanup


def main():
	register_matplotlib_converters()

	# 'Air Quality'
	data = pd.read_csv('data/air_quality_tabular_dummified.csv', parse_dates=True, infer_datetime_format=True)
	scale(data, 'air_quality', 'images/lab3/scaling/air_quality_scaling.png')

	# 'NYC Collisions'
	data = pd.read_csv('data/NYC_collisions_tabular_dummified.csv', parse_dates=True, infer_datetime_format=True)
	scale(data, 'NYC_collisions', 'images/lab3/scaling/NYC_collisions_scaling.png')



if __name__ == '__main__':
	main()