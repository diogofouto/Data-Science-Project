import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ds_charts import plot_evaluation_results, bar_chart, get_variable_types
from pandas.plotting import register_matplotlib_converters


AQ_TARGET = 'ALARM'
NYC_TARGET = 'PERSON_INJURY'


estimators = {'GaussianNB': GaussianNB(),
			  'MultinomialNB': MultinomialNB(),
			  'BernoulliNB': BernoulliNB()
			  }


def make_train_test_sets(filename, target):
	data: pd.DataFrame = pd.read_csv(f'data/{filename}.csv', parse_dates=True, infer_datetime_format=True)

	y: np.ndarray = data.pop(target).values
	X: np.ndarray = data.values
	labels: np.ndarray = pd.unique(y)
	labels.sort()

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)

	return X_train, X_test, y_train, y_test, labels


def normalize_for_multinomialNB(X_train, X_test):
	scaler = MinMaxScaler()
	X_train1 = scaler.fit_transform(X_train)
	X_test1 = scaler.fit_transform(X_test)

	return X_train1, X_test1


def nb_variants(trnX, tstX, trnY, tstY, file_tag):
	xvalues = []
	yvalues = []
	best = 'GaussianNB'
	last_best = 0
	X_train, X_test = trnX, tstX
	
	for clf in estimators:
		if clf == 'MultinomialNB':
			X_train, X_test = normalize_for_multinomialNB(trnX, tstX)
		
		xvalues.append(clf)
		estimators[clf].fit(X_train, trnY)
		prdY = estimators[clf].predict(X_test)
		yvalues.append(accuracy_score(tstY, prdY))
		
		if yvalues[-1] > last_best:
			best = clf
			last_best = yvalues[-1]
		
		# Reset sets
		X_train, X_test = trnX, tstX

	plt.figure()
	bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', xlabel='n', ylabel='accuracy', percentage=True)
	plt.savefig(f'images/lab4/nb/{file_tag}_nb_study.png')
	plt.clf()

	return best


def nb_performance(best, X_train, X_test, y_train, y_test, labels, file_tag):
	if best == 'MultinomialNB':
		X_train, X_test = normalize_for_multinomialNB(X_train, X_test)

	clf = estimators[best]
	clf.fit(X_train, y_train)
	prd_trn = clf.predict(X_train)
	prd_tst = clf.predict(X_test)

	plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)

	plt.savefig(f'images/lab4/nb/{file_tag}_nb_best.png')
	plt.clf()


def main():
	register_matplotlib_converters()

	for (filename, filetag, target) in [('air_quality_tabular_dummified', 'air_quality_noscaling', AQ_TARGET),
								('NYC_collisions_tabular_dummified', 'NYC_collisions_noscaling', NYC_TARGET),
								('air_quality_scaled_zscore', 'air_quality_zscore', AQ_TARGET),
								('NYC_collisions_scaled_zscore', 'NYC_collisions_zscore', NYC_TARGET),
								('air_quality_scaled_minmax', 'air_quality_minmax', AQ_TARGET),
								('NYC_collisions_scaled_minmax', 'NYC_collisions_minmax', NYC_TARGET)]:

		X_train, X_test, y_train, y_test, labels = make_train_test_sets(filename, target)

		best = nb_variants(X_train, X_test, y_train, y_test, filetag)
		nb_performance(best, X_train, X_test, y_train, y_test, labels, filetag)

if __name__ == '__main__':
	main()
