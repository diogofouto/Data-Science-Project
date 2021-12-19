import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

AQ_FILENAME = 'air_quality_scaled_zscore'
AQ_TARGET = 'ALARM'

NYC_FILENAME = 'NYC_collisions_scaled_minmax'
NYC_TARGET = 'PERSON_INJURY'

def make_train_test_sets(filename, file_tag, target):
    data: pd.DataFrame = pd.read_csv(f'data/{filename}.csv', parse_dates=True, infer_datetime_format=True)

    y: np.ndarray = data.pop(target).values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)
    labels.sort()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)

    train = pd.concat([pd.DataFrame(X_train, columns=data.columns), pd.DataFrame(y_train,columns=[target])], axis=1)
    train.to_csv(f'data/{file_tag}_train.csv', index=False)

    test = pd.concat([pd.DataFrame(X_test, columns=data.columns), pd.DataFrame(y_test,columns=[target])], axis=1)
    test.to_csv(f'data/{file_tag}_test.csv', index=False)    

def main():
    make_train_test_sets(filename=AQ_FILENAME, file_tag='air_quality', target=AQ_TARGET)
    make_train_test_sets(filename=NYC_FILENAME, file_tag='NYC_collisions', target=NYC_TARGET)



if __name__ == '__main__':
    main()
