import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score

from utils import *

def random_forests_study(X_train, y_train, X_test, y_test, file_tag):
    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25]
    max_features = [.1, .3, .5, .7, .9, 1]
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(max_depths)
    plt.figure()
    fig, axs = plt.subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(max_depths)):
        d = max_depths[k]
        values = {}
        for f in max_features:
            yvalues = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                yvalues.append(accuracy_score(y_test, y_pred))
                if yvalues[-1] > last_best:
                    best = (d, f, n)
                    last_best = yvalues[-1]
                    best_model = rf

            values[f] = yvalues
        multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',
                            xlabel='nr estimators', ylabel='accuracy', percentage=True)
    plt.savefig(f'images/{file_tag}_rf_study.png')
    plt.show()
    print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

    return best_model

def plot_feature_importance(best_model, train, file_tag):
    variables = train.columns
    importances = best_model.feature_importances_
    stdevs = np.std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    elems = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

    plt.figure()
    horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Random Forest Features importance', xlabel='importance', ylabel='variables')
    plt.savefig(f'images/{file_tag}_rf_ranking.png')

def main():
    for filename, file_tag, target in [(AQ_FILENAME, AQ_FILETAG, AQ_TARGET),
                                    (NYC_FILENAME, NYC_FILETAG, NYC_TARGET)]:

        train, X_train, y_train, test, X_test, y_test, labels = get_splits_and_labels(filename, target)
        best = random_forests_study(X_train, y_train, X_test, y_test, file_tag)
        get_model_evaluation(best, X_train, y_train, X_test, y_test, labels, file_tag, 'rf_best')
        plot_feature_importance(train, best, file_tag)

if __name__ == '__main__':
    main()