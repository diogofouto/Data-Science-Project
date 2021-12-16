import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from ds_charts import plot_evaluation_results, multiple_line_chart

AQ_FILENAME = 'data/air_quality_tabular_dummified'
AQ_FILETAG = 'air_quality_tabular'
AQ_TARGET = 'ALARM'

AQ_NEIGHS = [10, 20, 100, 200, 500, 1000, 1500]

NYC_FILENAME = 'data/NYC_collisions_tabular_dummified'
NYC_FILETAG = 'NYC_collisions_tabular'
NYC_TARGET = 'PERSON_INJURY'

NYC_NEIGHS = range(1, 20, 2)

def make_train_test_sets(filename, target):
    data: pd.DataFrame = pd.read_csv(f'data/{filename}.csv', parse_dates=True, infer_datetime_format=True)

    y: np.ndarray = data.pop(target).values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)
    labels.sort()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)

    return X_train, X_test, y_train, y_test, labels

def knn_variants(X_train, X_test, y_train, y_test, file_tag, nvalues):
    dist = ['manhattan', 'euclidean', 'seuclidean', 'chebyshev', 'minkowsky', 'wminkowsky', 'mahalanobis']
    values = {}
    best = (0, '')
    last_best = 0
    for d in dist:
        yvalues = []
        for n in nvalues:
            print("-- variant: ({}, {}) --".format(n, d))
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(X_train, y_train)
            prdY = knn.predict(X_test)
            yvalues.append(accuracy_score(y_test, prdY))
            if yvalues[-1] > last_best:
                best = (n, d)
                last_best = yvalues[-1]
        values[d] = yvalues

    plt.figure()
    multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
    print("-- plotted. now saving --")
    plt.savefig(f'images/lab3/knn/{file_tag}_knn_study.png')
    print('Best results with %d neighbors and %s'%(best[0], best[1]))

    return best

def knn_performance(n_neighs, metric, X_train, X_test, y_train, y_test, labels, file_tag):
    clf = KNeighborsClassifier(n_neighbors=n_neighs, metric=metric)
    clf.fit(X_train, y_train)

    prd_trn = clf.predict(X_train)
    prd_tst = clf.predict(X_test)

    print("-- plotting --")
    plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
    print("-- plotted. now saving --")
    plt.savefig(f'images/lab3/knn/{file_tag}_knn_best.png')

def main():

    for (filename, filetag, target, neighs) in [('air_quality_tabular_dummified',       'air_quality_noscaling',    AQ_TARGET,  AQ_NEIGHS),
                                                ('NYC_collisions_tabular_dummified',    'NYC_collisions_noscaling', NYC_TARGET, NYC_NEIGHS),
                                                ('air_quality_scaled_zscore',           'air_quality_zscore',       AQ_TARGET,  AQ_NEIGHS),
                                                ('NYC_collisions_scaled_zscore',        'NYC_collisions_zscore',    NYC_TARGET, NYC_NEIGHS),
                                                ('air_quality_scaled_minmax',           'air_quality_minmax',       AQ_TARGET,  AQ_NEIGHS),
                                                ('NYC_collisions_scaled_minmax',        'NYC_collisions_minmax',    NYC_TARGET, NYC_NEIGHS)]:

        print("current: ", filename, filetag)

        X_train, X_test, y_train, y_test, labels = make_train_test_sets(filename, target)

        print("- knn_variants start -")
        best = knn_variants(X_train, X_test, y_train, y_test, filetag, nvalues=neighs)

        print("- knn_performance start -")
        knn_performance(best[0], best[1], X_train, X_test, y_train, y_test, labels, filetag)

        print("! done !")

if __name__ == '__main__':
    main()
