import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from ds_charts import multiple_line_chart, plot_evaluation_results, horizontal_bar_chart

from utils import *

AQ_DEPTHS = [2, 5, 10, 15, 20, 25, 30]
NYC_DEPTHS = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

def dtree_study(X_train, y_train, X_test, y_test, max_depths, file_tag):
    min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
    criteria = ['entropy', 'gini']
    best = ('',  0, 0.0)
    last_best = 0
    best_model = None

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        print(f"-- {criteria[k]} --")
        f = criteria[k]
        values = {}
        for d in max_depths:
            print(f"--- Depth: {d} ---")
            yvalues = []
            for imp in min_impurity_decrease:
                print(f"---- Impurity Decrease: {imp} ----")
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
    plt.savefig(f'images/lab5/{file_tag}_dt_study.png')
    plt.show()
    print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.5f ==> accuracy=%1.5f'%(best[0], best[1], best[2], last_best))

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
    plt.savefig(f'images/lab5/{file_tag}_dt_ranking.png')

def plot_tree(model, train, labels, file_tag):
    lbls = [str(value) for value in labels]
    plt.figure(figsize=(20, 20), dpi=300)
    tree.plot_tree(model, feature_names=train.columns, class_names=lbls)
    plt.savefig(f'images/lab5/{file_tag}_dt_best_tree.png')

def main():
    for filename, file_tag, target, max_depths in [(AQ_FILENAME, AQ_FILETAG, AQ_TARGET, AQ_DEPTHS),
                                                    (NYC_FILENAME, NYC_FILETAG, NYC_TARGET, NYC_DEPTHS)]:

        print(f">> {file_tag}")
        train, X_train, y_train, X_dev, y_dev, test, X_test, y_test, labels = get_splits_tests_and_labels(filename, target)
        print("- Starting Random Forests Study")
        best = dtree_study(X_train, y_train, X_dev, y_dev, max_depths, file_tag)
        print("- Plotting Tree")
        plot_tree(best, train, labels, file_tag)
        print("- Getting Model Evaluation (w/ dev)")
        get_model_evaluation(best, X_train, y_train, X_dev, y_dev, labels, file_tag, 'dt_best')
        print("- Getting Feature Importance")
        plot_feature_importance(train, best, file_tag)
        print("- Getting Model Evaluation (w/ test)")
        get_model_evaluation(best, X_train, y_train, X_test, y_test, labels, file_tag, 'dt_test')

if __name__ == '__main__':
    main()