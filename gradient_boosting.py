import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT, plot_overfitting_study
from sklearn.metrics import accuracy_score
from ds_charts import plot_overfitting_study
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score


AQ_FILENAME = 'air_quality'
AQ_FILETAG = 'air_quality'
AQ_TARGET = 'ALARM'

NYC_FILENAME = 'NYC_collisions'
NYC_FILETAG = 'NYC_collisions'
NYC_TARGET = 'PERSON_INJURY'


def get_splits_tests_and_labels(filename, target):
    train: pd.DataFrame = pd.read_csv(f'data/{filename}_train.csv')
    y: np.ndarray = train.pop(target).values
    X: np.ndarray = train.values
    labels = np.unique(y)
    labels.sort()

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, train_size=0.7, stratify=y)

    test: pd.DataFrame = pd.read_csv(f'data/{filename}_test.csv')
    y_test: np.ndarray = test.pop(target).values
    X_test: np.ndarray = test.values

    return train, X_train, y_train, X_dev, y_dev, test, X_test, y_test, labels

def gradient_boosting_study(X_train, y_train, X_test, y_test, max_depths, file_tag,):
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
    #plt.savefig(f'images/lab7/gradient_boosting/{file_tag}_gb_study1.png')
    #plt.show()
    #print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

    return best_model

def get_model_evaluation(model, X_train, y_train, X_test, y_test, labels, file_tag, sufix):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    plot_evaluation_results(labels, y_train, train_pred, y_test, test_pred)
    #plt.savefig(f'images/lab7/gradient_boosting/{file_tag}_{sufix}1.png')
    #plt.show()1


def plot_feature_importance(best_model, train, file_tag):
    variables = train.columns
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    stdevs = np.std([tree[0].feature_importances_ for tree in best_model.estimators_], axis=0)
    elems = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

    #plt.figure()
    #horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Gradient Boosting Features importance', xlabel='importance', ylabel='variables')
    #plt.savefig(f'images/lab7/gradient_boosting/{file_tag}_gb_ranking1.png')


def plot_overfitting(trnX, trnY, tstX, tstY, n_estimators):
    lr = 0.1
    d = 2
    eval_metric = recall_score
    y_tst_values = []
    y_trn_values = []
    for n in n_estimators:
        gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
        gb.fit(trnX, trnY)
        prd_tst_Y = gb.predict(tstX)
        prd_trn_Y = gb.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y, average='micro'))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y, average='micro'))
    plot_overfitting_study(n_estimators, y_trn_values, y_tst_values, name=f"GB_depth={d}_lr={lr}", xlabel='nr_estimators', ylabel=str(eval_metric))


def main():
    for filename, file_tag, target, max_depths in [(NYC_FILENAME, NYC_FILETAG, NYC_TARGET, [2])]:
        print(f">> {file_tag}")
        train, X_train, y_train, X_dev, y_dev, test, X_test, y_test, labels = get_splits_tests_and_labels(filename, target)
        print("- Starting Gradient Boosting Study")
        best = gradient_boosting_study(X_train, y_train, X_test, y_test, max_depths, file_tag)
        print("- Getting Model Evaluation (w/ dev)")
        get_model_evaluation(best, X_train, y_train, X_dev, y_dev, labels, file_tag, 'gb_best')
        print("- Plotting Feature Importance")
        plot_feature_importance(best, train, file_tag)
        print("- Getting Model Evaluation (w/ test)")
        get_model_evaluation(best, X_train, y_train, X_test, y_test, labels, file_tag, 'gb_test')
        print("- Plotting overfitting")
        plot_overfitting(X_train, y_train, X_test, y_test, [i for i in range(50, 550, 50)])

if __name__ == '__main__':
    main()