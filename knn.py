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
    values = {}
    last_best = 0
    best = (0, '')

    def normal_versions(best, last_best):
        dist = ['manhattan', 'euclidean', 'chebyshev']
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

        return last_best, best

    def weighted_versions(best, last_best):
        wdist = ['wmanhattan', 'weuclidean']
        for (ix, d) in zip(range(len(wdist)), wdist):
            yvalues = []
            for n in nvalues:
                print("-- variant: ({}, {}) --".format(n, d))
                print("-- p: ", ix + 1)
                knn = KNeighborsClassifier(weights='distance', n_neighbors=n, p=ix + 1)
                knn.fit(X_train, y_train)
                prdY = knn.predict(X_test)
                yvalues.append(accuracy_score(y_test, prdY))
                if yvalues[-1] > last_best:
                    best = (n, d)
                    last_best = yvalues[-1]
            values[d] = yvalues
        
        yvalues = []
        for n in nvalues:
            print("-- variant: ({}, wchebyshev) --".format(n))
            knn = KNeighborsClassifier(weights='distance', metric='chebyshev', n_neighbors=n)
            knn.fit(X_train, y_train)
            prdY = knn.predict(X_test)
            yvalues.append(accuracy_score(y_test, prdY))
            if yvalues[-1] > last_best:
                best = (n, 'wchebyshev')
                last_best = yvalues[-1]
            values['wchebyshev'] = yvalues

        return last_best, best

    last_best, best = normal_versions(best, last_best)
    last_best, best = weighted_versions(best, last_best)

    plt.figure()
    multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
    print("-- plotted. now saving --")
    plt.savefig(f'images/lab3/knn/{file_tag}_knn_study.png')
    print('Best results with %d neighbors and %s'%(best[0], best[1]))

    return best

def knn_performance(n_neighs, metric, X_train, X_test, y_train, y_test, labels, file_tag):
    clf = None

    if metric == 'wmanhattan':
        clf = KNeighborsClassifier(weights='distance', n_neighbors=n_neighs, p = 1)
    elif metric == 'weuclidean':
        clf = KNeighborsClassifier(weights='distance', n_neighbors=n_neighs, p = 2)
    elif metric == 'wchebyshev':
        clf = KNeighborsClassifier(weights='distance', n_neighbors=n_neighs, metric='chebyshev')
    else:
        clf = KNeighborsClassifier(n_neighbors=n_neighs, metric=metric)

    clf.fit(X_train, y_train)

    prd_trn = clf.predict(X_train)
    prd_tst = clf.predict(X_test)

    print("-- plotting --")
    plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
    print("-- plotted. now saving --")
    plt.savefig(f'images/lab3/knn/{file_tag}_knn_best.png')

def main():

    for (filename, filetag, target, neighs) in [('air_quality_tabular_over',       'air_quality_over',    AQ_TARGET,  AQ_NEIGHS),
                                                ('NYC_collisions_tabular_over',    'NYC_collisions_over', NYC_TARGET, NYC_NEIGHS),
                                                ('air_quality_tabular_smote',           'air_quality_smote',       AQ_TARGET,  AQ_NEIGHS),
                                                ('NYC_collisions_tabular_smote',        'NYC_collisions_smote',    NYC_TARGET, NYC_NEIGHS),
                                                ('air_quality_tabular_under',           'air_quality_under',       AQ_TARGET,  AQ_NEIGHS),
                                                ('NYC_collisions_tabular_under',        'NYC_collisions_under',    NYC_TARGET, NYC_NEIGHS)]:

        print("current: ", filename, filetag)

        X_train, X_test, y_train, y_test, labels = make_train_test_sets(filename, target)

        print("- knn_variants start -")
        best = knn_variants(X_train, X_test, y_train, y_test, filetag, nvalues=neighs)

        print("- knn_performance start -")
        knn_performance(best[0], best[1], X_train, X_test, y_train, y_test, labels, filetag)

        print("! done !")

if __name__ == '__main__':
    main()
