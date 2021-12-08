import pandas as pd
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types


def print_unique_symbolic_values(data):
	variables = get_variable_types(data)['Symbolic']
	if [] == variables:
		raise ValueError('There are no symbolic variables.')

	if 'GbCity' in variables:
		variables.remove('GbCity')

	if 'date' in variables:
		variables.remove('date')

	if 'CRASH_DATE' in variables:
		variables.remove('CRASH_DATE')

	if 'CRASH_TIME' in variables:
		variables.remove('CRASH_TIME')

	if 'PERSON_ID' in variables:
		variables.remove('PERSON_ID')

	#print()
	#print(variables)

	data = data.drop(columns=[col for col in data if col not in variables])

	for col in data:
		print('\n' + col)
		print(data[col].unique())

	print()


def main():
	register_matplotlib_converters()

	# 'Air Quality'
	data = pd.read_csv('data/air_quality_tabular.csv', index_col='FID', parse_dates=True, infer_datetime_format=True)
	print_unique_symbolic_values(data)


	# 'NYC Collisions'
	data = pd.read_csv('data/NYC_collisions_tabular.csv', index_col='COLLISION_ID', parse_dates=True, infer_datetime_format=True)
	print_unique_symbolic_values(data)



if __name__ == '__main__':
	main()