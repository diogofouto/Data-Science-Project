import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from ds_charts import multiple_line_chart, plot_evaluation_results, horizontal_bar_chart

from utils import *

def dtree_study(X_train, y_train, X_test, y_test, file_tag):
    min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
    max_depths = [2, 5, 10, 15, 20, 25]
    criteria = ['entropy', 'gini']
    best = ('',  0, 0.0)
    last_best = 0
    best_model = None

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for imp in min_impurity_decrease:
                tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(X_train, y_train)
                y_pred = tree.predict(X_test)
                yvalues.append(accuracy_score(y_test, y_pred))
                if yvalues[-1] > last_best:
                    best = (f, d, imp)
                    last_best = yvalues[-1]
                    best_model = tree

            values[d] = yvalues
        multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                            xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
    plt.savefig(f'images/{file_tag}_dt_study.png')
    plt.show()
    print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

    return best_model

def plot_feature_importance(training_set, best_model, file_tag):
    variables = training_set.columns
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    elems = []
    imp_values = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        imp_values += [importances[indices[f]]]
        print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

    plt.figure()
    horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance', ylabel='variables')
    plt.savefig(f'images/{file_tag}_dt_ranking.png')

def plot_tree(model, train, labels, file_tag):
    lbls = [str(value) for value in labels]
    plot_tree(model, feature_names=train.columns, class_names=lbls)
    plt.savefig(f'images/{file_tag}_dt_best_tree.png')

def main():
    for filename, file_tag, target in [(AQ_FILENAME, AQ_FILETAG, AQ_TARGET),
                                        (NYC_FILENAME, NYC_FILETAG, NYC_TARGET)]:

        train, X_train, y_train, test, X_test, y_test, labels = get_splits_and_labels(filename, target)
        best = dtree_study(X_train, y_train, X_test, y_test, file_tag)
        plot_tree(best, train, labels, file_tag)
        get_model_evaluation(best, X_train, y_train, X_test, y_test, labels, file_tag, 'dt_best')
        plot_feature_importance(train, best, file_tag)

if __name__ == '__main__':
    main()