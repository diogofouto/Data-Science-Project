import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, bar_chart, HEIGHT
from sklearn.metrics import recall_score


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


def get_model_evaluation(model, X_train, y_train, X_test, y_test, labels, file_tag, sufix):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return get_recall_score(labels, y_train, train_pred, y_test, test_pred)


def random_forests_study(X_train, y_train, X_test, y_test, max_depths, file_tag, min_imp):
    n_estimators = [400]
    max_features = [.7]
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
                yvalues.append(recall_score(y_test, y_pred, average='micro'))
                if yvalues[-1] > last_best:
                    best = (d, f, n)
                    last_best = yvalues[-1]
                    best_model = rf

            values[f] = yvalues
    
    return best_model

def main():
    
    lst_variables = []
    lst_recall_train = []
    lst_recall_test = []

    for (filename_train, filename_test, filetag, target) in [('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.0', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.0', 'No_correlated_vars_0.0', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.1', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.1', 'No_correlated_vars_0.1', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.2', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.2', 'No_correlated_vars_0.2', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.3', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.3', 'No_correlated_vars_0.3', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.4', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.4', 'No_correlated_vars_0.4', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.5', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.5', 'No_correlated_vars_0.5', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.6', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.6', 'No_correlated_vars_0.6', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.7', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.7', 'No_correlated_vars_0.7', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.8', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.8', 'No_correlated_vars_0.8', AQ_TARGET),
('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.9', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.9', 'No_correlated_vars_0.9', AQ_TARGET)]:


        train, X_train, y_train, X_dev, y_dev, test, X_test, y_test, labels = get_splits_tests_and_labels(filename_train, filename_test, target)

        best = random_forests_study(X_train, y_train, X_test, y_test, [25], filetag, min_imp=0.0005)

        recall_score = get_model_evaluation(best, X_train, y_train, X_dev, y_dev, labels, filetag, 'rf_best')
        
        lst_variables.append(filetag)
        lst_recall_train.append(recall_score[0])
        lst_recall_test.append(recall_score[1])

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    bar_chart(lst_variables, lst_recall_train, ax=axs[0], title='Air Quality Train set', xlabel='variables', ylabel='recall', rotation=45)
    bar_chart(lst_variables, lst_recall_test, ax=axs[1], title='Air Quality Test set', xlabel='variables', ylabel='recall', rotation=45)
    plt.savefig('images/lab6/feature_selection/air_quality_correlation_rf_study', dpi=300, bbox_inches="tight")
    plt.clf() # cleanup

if __name__ == '__main__':
    main()