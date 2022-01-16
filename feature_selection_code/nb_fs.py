import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ds_charts import plot_evaluation_results, bar_chart, get_variable_types, get_recall_score, bar_chart
from pandas.plotting import register_matplotlib_converters


AQ_TARGET = 'ALARM'
NYC_TARGET = 'PERSON_INJURY'


estimators = {'GaussianNB': GaussianNB(),
			  'MultinomialNB': MultinomialNB(),
			  'BernoulliNB': BernoulliNB()
			  }


def make_train_test_sets(filename_train, filename_test, target):
	train: pd.DataFrame = pd.read_csv(f'data/{filename_train}.csv')
	y: np.ndarray = train.pop(target).values
	X: np.ndarray = train.values
	labels = np.unique(y)
	labels.sort()

	X_train, X_dev, y_train, y_dev = train_test_split(X, y, train_size=0.7, stratify=y)

	test: pd.DataFrame = pd.read_csv(f'data/{filename_test}.csv')
	y_test: np.ndarray = test.pop(target).values
	X_test: np.ndarray = test.values

	return X_train, X_test, y_train, y_test, labels


def normalize_for_multinomialNB(X_train, X_test):
	scaler = MinMaxScaler()
	X_train1 = scaler.fit_transform(X_train)
	X_test1 = scaler.fit_transform(X_test)

	return X_train1, X_test1


def nb_performance(best, X_train, X_test, y_train, y_test, labels, file_tag):
	if best == 'MultinomialNB':
		X_train, X_test = normalize_for_multinomialNB(X_train, X_test)

	clf = estimators[best]
	clf.fit(X_train, y_train)
	prd_trn = clf.predict(X_train)
	prd_tst = clf.predict(X_test)

	return get_recall_score(labels, y_train, prd_trn, y_test, prd_tst)

	plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)

	plt.savefig(f'images/lab6/feature_selection/{file_tag}_nb_best.png')
	plt.clf()


def main():
	register_matplotlib_converters()

	lst_variables = []
	lst_recall_train = []
	lst_recall_test = []

	for (filename_train, filename_test, filetag, target) in [('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.0', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.0', 'No_correlated_vars_0.0', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.1', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.1', 'No_correlated_vars_0.1', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.2', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.2', 'No_correlated_vars_0.2', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.3', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.3', 'No_correlated_vars_0.3', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.4', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.4', 'No_correlated_vars_0.4', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.5', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.5', 'No_correlated_vars_0.5', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.6', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.6', 'No_correlated_vars_0.6', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.7', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.7', 'No_correlated_vars_0.7', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.8', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.8', 'No_correlated_vars_0.8', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.9', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.9', 'No_correlated_vars_0.9', AQ_TARGET)]:


		X_train, X_test, y_train, y_test, labels = make_train_test_sets(filename_train, filename_test, target)

		if target == NYC_TARGET:
			best = 'MultinomialNB'
		else:
			best = 'GaussianNB'

		recall_score = nb_performance(best, X_train, X_test, y_train, y_test, labels, filetag)
		
		lst_variables.append(filetag)
		lst_recall_train.append(recall_score[0])
		lst_recall_test.append(recall_score[1])

	fig, axs = plt.subplots(1, 2, figsize=(8, 4))
	bar_chart(lst_variables, lst_recall_train, ax=axs[0], title='Air Quality Train set', xlabel='variables', ylabel='recall', rotation=45)
	bar_chart(lst_variables, lst_recall_test, ax=axs[1], title='Air Quality Test set', xlabel='variables', ylabel='recall', rotation=45)
	plt.savefig('images/lab6/feature_selection/air_quality_correlation_nb_study', dpi=300, bbox_inches="tight")
	plt.clf() # cleanup

if __name__ == '__main__':
	main()
