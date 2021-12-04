from pandas import read_csv
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import subplots, savefig


def plotHistograms(filename, bins):
	data = read_csv(filename)
	variables = get_variable_types(data)['Numeric']
	if [] == variables:
	    raise ValueError('There are no numeric variables.')

	rows = len(variables)
	cols = len(bins)
	fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT))
	for i in range(rows):
	    for j in range(cols):
	        axs[i, j].set_title('Histogram for %s - %d bins'%(variables[i], bins[j]))
	        axs[i, j].set_xlabel(variables[i])
	        axs[i, j].set_ylabel('Nr records')
	        axs[i, j].hist(data[variables[i]].values, bins=bins[j])


def main():
	# Get granularity for 'Air Quality'
	filename = 'data/air_quality_tabular.csv'
	bins = (10, 100)
	plotHistograms(filename, bins)
	savefig('images/lab1/granularity/air_quality_granularity_study.png')

	# Get granularity for 'NYC Collisions'
	filename = 'data/NYC_collisions_tabular.csv'
	bins = (10, 100, 1000, 10000)
	plotHistograms(filename, bins)
	savefig('images/lab1/granularity/NYC_collisions_granularity_study.png')


if __name__ == '__main__':
	main()