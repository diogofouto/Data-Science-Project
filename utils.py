import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ds_charts import plot_evaluation_results

AQ_FILENAME = 'data/air_quality'
AQ_FILETAG = 'air_quality'
AQ_TARGET = 'ALARM'

NYC_FILENAME = 'data/NYC_collisions'
NYC_FILETAG = 'NYC_collisions'
NYC_TARGET = 'PERSON_INJURY'

def get_splits_and_labels(filename, target):
    train: pd.DataFrame = pd.read_csv(f'{filename}_train.csv')
    y_train: np.ndarray = train.pop(target).values
    X_train: np.ndarray = train.values
    labels = np.unique(y_train)
    labels.sort()

    test: pd.DataFrame = pd.read_csv(f'{filename}_test.csv')
    y_test: np.ndarray = test.pop(target).values
    X_test: np.ndarray = test.values

    return train, X_train, y_train, test, X_test, y_test, labels

def get_model_evaluation(model, X_train, y_train, X_test, y_test, labels, file_tag, sufix):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    plot_evaluation_results(labels, y_train, train_pred, y_test, test_pred)
    plt.savefig(f'images/{file_tag}_{sufix}.png')
    plt.show()
