import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score

from utils import *

def random_forests_study(X_train, y_train, X_test, y_test, max_depths, file_tag, min_imp):
    n_estimators = [i for i in range(50, 550, 50)]
    max_features = [.1, .3, .5, .7, .9, 1]
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
        for f in max_features:
            print(f"--- Max Features: {f} ---")
            yvalues = []
            for n in n_estimators:
                print(f"---- Nr. Estimators: {n}")
                rf = RandomForestClassifier(criterion='entropy', n_estimators=n, max_depth=d, max_features=f, min_impurity_decrease=min_imp)
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
    plt.savefig(f'images/lab6/{file_tag}_rf_study.png')
    plt.show()
    print('Best results with depth=%d, %1.5f features and %d estimators, with accuracy=%1.5f'%(best[0], best[1], best[2], last_best))

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
    plt.savefig(f'images/lab6/{file_tag}_rf_ranking.png')

def main():
    for filename, file_tag, target, max_depths in [(AQ_FILENAME, AQ_FILETAG, AQ_TARGET, [10]),
                                                    (NYC_FILENAME, NYC_FILETAG, NYC_TARGET, [20])]:
        print(f">> {file_tag}")
        train, X_train, y_train, X_dev, y_dev, test, X_test, y_test, labels = get_splits_tests_and_labels(filename, target)
        print("- Starting Random Forests Study")
        best = random_forests_study(X_train, y_train, X_test, y_test, max_depths, file_tag, min_imp=0.0005)
        print("- Getting Model Evaluation (w/ dev)")
        get_model_evaluation(best, X_train, y_train, X_dev, y_dev, labels, file_tag, 'rf_best')
        print("- Plotting Feature Importance")
        plot_feature_importance(train, best, file_tag)
        print("- Getting Model Evaluation (w/ test)")
        get_model_evaluation(best, X_train, y_train, X_test, y_test, labels, file_tag, 'rf_test')

if __name__ == '__main__':
    main()