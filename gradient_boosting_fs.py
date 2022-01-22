import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT, plot_overfitting_study, get_recall_score, bar_chart
from sklearn.metrics import accuracy_score
from ds_charts import plot_overfitting_study
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score


AQ_FILENAME = 'air_quality'
AQ_FILETAG = 'air_quality'
AQ_TARGET = 'ALARM'

NYC_TARGET = 'PERSON_INJURY'


def get_splits_tests_and_labels(filename_train, filename_test, target):
    train: pd.DataFrame = pd.read_csv(f'data/{filename_train}.csv')
    y: np.ndarray = train.pop(target).values
    X: np.ndarray = train.values
    labels = np.unique(y)
    labels.sort()

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, train_size=0.7, stratify=y)

    test: pd.DataFrame = pd.read_csv(f'data/{filename_test}.csv')
    y_test: np.ndarray = test.pop(target).values
    X_test: np.ndarray = test.values

    return train, X_train, y_train, X_dev, y_dev, test, X_test, y_test, labels

def gradient_boosting_study(X_train, y_train, X_test, y_test, max_depths, file_tag):
    n_estimators = [150]
    learning_rate = [.1]
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(max_depths)
    plt.figure()
    fig, axs = plt.subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(max_depths)):
        print(f"-- Depth: {max_depths[k]} --")
        d = max_depths[k]
        values = {}
        for lr in learning_rate:
            print(f"--- Learning rate: {lr} ---")
            yvalues = []
            for n in n_estimators:
                print(f"---- Nr. Estimators: {n}")
                gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
                gb.fit(X_train, y_train)
                prdY = gb.predict(X_test)
                yvalues.append(recall_score(y_test, prdY, average='micro'))
                if yvalues[-1] > last_best:
                    best = (d, lr, n)
                    last_best = yvalues[-1]
                    best_model = gb
            values[lr] = yvalues
        multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Gradient Boosting with max_depth={d}',
                               xlabel='nr estimators', ylabel='accuracy', percentage=True)
    
    return best_model

def get_model_evaluation(model, X_train, y_train, X_test, y_test, labels, file_tag, sufix):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return get_recall_score(labels, y_train, train_pred, y_test, test_pred)


def main():
    lst_variables = []
    lst_recall_train = []
    lst_recall_test = []

    for (filename_train, filename_test, filetag, target) in [('NYC_collisions_train_smote_no_correlated_vars_0.8', 'NYC_collisions_test_no_correlated_vars_0.8', 'corr_thres=0.8', NYC_TARGET),
    ('NYC_collisions_train_smote_f_classif_k=4', 'NYC_collisions_test_f_classif_k=4', 'k=4', NYC_TARGET)]:


        train, X_train, y_train, X_dev, y_dev, test, X_test, y_test, labels = get_splits_tests_and_labels(filename_train, filename_test, target)

        best = gradient_boosting_study(X_train, y_train, X_test, y_test, [2], filetag)

        recall_score = get_model_evaluation(best, X_train, y_train, X_dev, y_dev, labels, filetag, 'gb_best')
        
        lst_variables.append(filetag)
        lst_recall_train.append(recall_score[0])
        lst_recall_test.append(recall_score[1])

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    bar_chart(lst_variables, lst_recall_train, ax=axs[0], title='NYC Collisions Train set', xlabel='fs', ylabel='recall', rotation=45)
    bar_chart(lst_variables, lst_recall_test, ax=axs[1], title='NYC Collisions Test set', xlabel='fs', ylabel='recall', rotation=45)
    plt.savefig('images/lab6/feature_selection/NYC_collisions_fs_gb_study', dpi=300, bbox_inches="tight")
    plt.clf() # cleanup

if __name__ == '__main__':
    main()