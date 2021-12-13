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

NYC_FILENAME = 'data/NYC_collisions_tabular_dummified'
NYC_FILETAG = 'NYC_collisions_tabular'
NYC_TARGET = 'PERSON_INJURY'

def make_train_test_sets(filename, file_tag, target):
    data: pd.DataFrame = pd.read_csv(f'data/{filename}.csv', parse_dates=True, infer_datetime_format=True)

    X: np.ndarray = data.drop(columns=[target])
    y: np.ndarray = data.pop(target).values
    labels: np.ndarray = pd.unique(y)
    labels.sort()

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    train = pd.concat([pd.DataFrame(trnX, columns=data.columns), pd.DataFrame(trnY,columns=[target])], axis=1)
    train.to_csv(f'data/{file_tag}_train.csv', index=False)

    test = pd.concat([pd.DataFrame(tstX, columns=data.columns), pd.DataFrame(tstY,columns=[target])], axis=1)
    test.to_csv(f'data/{file_tag}_test.csv', index=False)

    return trnX, tstX, trnY, tstY, labels


def knn_variants(trnX, tstX, trnY, tstY):
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    best = (0, '')
    last_best = 0
    for d in dist:
        yvalues = []
        for n in nvalues:
            print("-- variant: ({}, {}) --".format(n, d))
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (n, d)
                last_best = yvalues[-1]
        values[d] = yvalues

    plt.figure()
    multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
    print("-- plotted. now saving --")
    plt.savefig('images/lab3/knn/{file_tag}_knn_study.png')
    print('Best results with %d neighbors and %s'%(best[0], best[1]))

    return best

def knn_performance(n_neighs, metric, trnX, trnY, tstX, tstY, labels):
    clf = KNeighborsClassifier(n_neighbors=n_neighs, metric=metric)

    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    print("-- plotting --")
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    print("-- plotted. now saving --")
    plt.savefig('images/lab3/knn/{file_tag}_knn_best.png')

def main():

    for (filename, filetag) in [('air_quality_tabular_dummified', 'air_quality_noscaling'),
                                ('NYC_collisions_tabular_dummified', 'NYC_collisions_noscaling'),
                                ('air_quality_scaled_zscore', 'air_quality_zscore'),
                                ('NYC_collisions_scaled_zscore', 'NYC_collisions_zscore'),
                                ('air_quality_scaled_minmax', 'air_quality_minmax'),
                                ('NYC_collisions_scaled_minmax', 'NYC_collisions_minmax')]:

        print("current: ", filename, filetag)

        trnX, tstX, trnY, tstY, labels = make_train_test_sets(filename, filetag, AQ_TARGET)

        print("- knn_variants start -")
        best = knn_variants(trnX, tstX, trnY, tstY)

        print("- knn_performance start -")
        knn_performance(best[0], best[1], trnX, tstX, trnY, tstY, labels)

        print("! done !")

if __name__ == '__main__':
    main()
