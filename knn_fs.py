import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from ds_charts import get_recall_score, bar_chart

AQ_TARGET = 'ALARM'
AQ_NEIGHS = [10]

NYC_TARGET = 'PERSON_INJURY'
NYC_NEIGHS = [1]

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
                yvalues.append(recall_score(y_test, prdY, average="micro"))
                if yvalues[-1] > last_best:
                    best = (n, d)
                    last_best = yvalues[-1]
            values[d] = yvalues

        return last_best, best

    def weighted_versions(best, last_best):
        wdist = ['wmanhattan']
        for (ix, d) in zip(range(len(wdist)), wdist):
            yvalues = []
            for n in nvalues:
                print("-- variant: ({}, {}) --".format(n, d))
                print("-- p: ", ix + 1)
                knn = KNeighborsClassifier(weights='distance', n_neighbors=n, p=ix + 1)
                knn.fit(X_train, y_train)
                prdY = knn.predict(X_test)
                yvalues.append(recall_score(y_test, prdY, average="micro"))
                if yvalues[-1] > last_best:
                    best = (n, d)
                    last_best = yvalues[-1]
            values[d] = yvalues

        return last_best, best

    last_best, best = weighted_versions(best, last_best)

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

    return get_recall_score(labels, y_train, prd_trn, y_test, prd_tst)

def main():

    lst_variables = []
    lst_recall_train = []
    lst_recall_test = []

    for (filename_train, filename_test, filetag, target) in [('NYC_collisions_train_smote_no_correlated_vars_0.8', 'NYC_collisions_test_no_correlated_vars_0.8', 'corr_thres=0.8', NYC_TARGET),
    ('NYC_collisions_train_smote_f_classif_k=4', 'NYC_collisions_test_f_classif_k=4', 'k=4', NYC_TARGET)]:


        X_train, X_test, y_train, y_test, labels = make_train_test_sets(filename_train, filename_test, target)

        print("- knn_variants start -")
        best = knn_variants(X_train, X_test, y_train, y_test, filetag, nvalues=[1])

        print("- knn_performance start -")
        recall_score = knn_performance(best[0], best[1], X_train, X_test, y_train, y_test, labels, filetag)
        
        lst_variables.append(filetag)
        lst_recall_train.append(recall_score[0])
        lst_recall_test.append(recall_score[1])

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    bar_chart(lst_variables, lst_recall_train, ax=axs[0], title='NYC Collisions Train set', xlabel='fs', ylabel='recall', rotation=45)
    bar_chart(lst_variables, lst_recall_test, ax=axs[1], title='NYC Collisions Test set', xlabel='fs', ylabel='recall', rotation=45)
    plt.savefig('images/lab6/feature_selection/NYC_collisions_fs_knn_study', dpi=300, bbox_inches="tight")
    plt.clf() # cleanup

if __name__ == '__main__':
    main()
