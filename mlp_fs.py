#%%
from typing import overload
from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from pandas.plotting import register_matplotlib_converters
from ds_charts import bar_chart, get_recall_score, bar_chart, HEIGHT



#%% 

AQ_FILENAME = 'data/air_quality_tabular_dummified'
AQ_FILETAG = 'air_quality_tabular'
AQ_TARGET = 'ALARM'

NYC_FILENAME = 'data/NYC_collisions_tabular_dummified'
NYC_FILETAG = 'NYC_collisions_tabular'
NYC_TARGET = 'PERSON_INJURY'


# TODO FIX ADD RIGHT FILE NAMES
file_tag_air = 'air_quality'
filename_air = 'data/air_quality'
target_air = 'ALARM'

"""
train_air: DataFrame = read_csv(f'{filename_air}_tabular_smote.csv')
trnY_air: ndarray = train_air.pop(target_air).values
trnX_air: ndarray = train_air.values
labels_air = unique(trnY_air)
labels_air.sort()

test_air: DataFrame = read_csv(f'{filename_air}_test.csv')
tstY_air: ndarray = test_air.pop(target_air).values
tstX_air: ndarray = test_air.values

lr_type_air = ['adaptive', 'constant', 'invscaling']
max_iter_air = [100, 300, 500, 750, 1000]
learning_rate_air = [.9, .7, .5, .3, .1]
best_air = ('', 0, 0)
last_best_air = 0
best_model_air = None

#%% 
file_tag_nyc = 'NYC_collisions'
filename_nyc = 'data/NYC_collisions'
target_nyc = 'PERSON_INJURY'

train_nyc: DataFrame = read_csv(f'{filename_nyc}_tabular_smote.csv')
trnY_nyc: ndarray = train_nyc.pop(target_nyc).values
trnX_nyc: ndarray = train_nyc.values
labels_nyc = unique(trnY_nyc)
labels_nyc.sort()

test_nyc: DataFrame = read_csv(f'{filename_nyc}_test.csv')
tstY_nyc: ndarray = test_nyc.pop(target_nyc).values
tstX_nyc: ndarray = test_nyc.values

lr_type_nyc = ['adaptive','constant', 'invscaling']
max_iter_nyc = [100, 300, 500, 750, 1000]
learning_rate_nyc = [.9,.7, .5, .3, .1]
best_nyc = ('', 0, 0)
last_best_nyc = 0
best_model_nyc = None
"""

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

#%%
def mlp(lr_types, learning_r, max_iter, file_tag, train_X, train_Y, test_X, test_Y, labels):
    cols = len(lr_types)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    print("mlp starting!")
    for k in range(len(lr_types)):
        print("-learning rate type:",lr_types[k])
        d = lr_types[k]
        values = {}
        for lr in learning_r:
            print("--learning rate:",lr)
            yvalues = []
            
            for n in max_iter:
                print("---max iter:",n)
        
                mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                    learning_rate_init=lr, max_iter=n, verbose=False)
                mlp.fit(train_X, train_Y)
                prdY = mlp.predict(test_X)

                
                acc_tt = recall_score(test_Y, prdY, average="micro")
                
                                
                yvalues.append(acc_tt)
                
                
                if yvalues[-1] > last_best:
                    best = (d, lr, n)
                    last_best = yvalues[-1]
                    best_model = mlp
            values[lr] = yvalues
            
    prd_trn = best_model.predict(train_X)
    prd_tst = best_model.predict(test_X)
    return best, get_recall_score(labels, train_Y, prd_trn, prd_tst, test_Y)


def main():
    register_matplotlib_converters()

    lst_variables = []
    lst_recall_train = []
    lst_recall_test = []
    
    #best_model = None
    best_model = ('invscaling', 0.3, 750) #AIR
    #best_model = ('constant', 0.1, 1000)

    for (filename_train, filename_test, filetag, target) in [
            #('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.0', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.0', 'No_correlated_vars_0.0', AQ_TARGET),
            #('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.1', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.1', 'No_correlated_vars_0.1', AQ_TARGET),
            #('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.2', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.2', 'No_correlated_vars_0.2', AQ_TARGET),
            #('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.3', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.3', 'No_correlated_vars_0.3', AQ_TARGET),
            #('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.4', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.4', 'No_correlated_vars_0.4', AQ_TARGET),
            #('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.5', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.5', 'No_correlated_vars_0.5', AQ_TARGET),
            #('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.6', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.6', 'No_correlated_vars_0.6', AQ_TARGET),
            #('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.7', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.7', 'No_correlated_vars_0.7', AQ_TARGET),
            #('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.8', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.8', 'No_correlated_vars_0.8', AQ_TARGET),
            #('air_quality_train_smote_no_low_variance_vars_no_correlated_vars_0.9', 'air_quality_test_no_low_variance_vars_no_correlated_vars_0.9', 'No_correlated_vars_0.9', AQ_TARGET)
            ('air_quality_train_smote_no_correlated_vars_0.4', 'air_quality_test_no_correlated_vars_0.4', 'corr_thres=0.4', AQ_TARGET),
    ('air_quality_train_smote_no_low_variance_vars_2.0', 'air_quality_test_no_low_variance_vars_2.0', 'var_thres=2.0', AQ_TARGET),
    ('air_quality_train_smote_no_low_variance_vars_2.0_no_correlated_vars_0.4', 'air_quality_test_no_low_variance_vars_2.0_no_correlated_vars_0.4', 'corr_thres=0.4, var_thres=2.0', AQ_TARGET),
    ('air_quality_train_smote_f_classif_k=4', 'air_quality_test_f_classif_k=4', 'k=4', AQ_TARGET)
            ]:
            
        X_train, X_test, y_train, y_test, labels = make_train_test_sets(filename_train, filename_test, target)
        
        if(best_model == None):
            best_model, recall_score = mlp(lr_type_nyc, learning_rate_nyc, max_iter_nyc, filetag, X_train, y_train, X_test, y_test, labels)
            print(best_model)
        else:
            mlp_model = MLPClassifier(activation='logistic', solver='sgd', learning_rate=best_model[0],
                                    learning_rate_init=best_model[1], max_iter=best_model[2], verbose=False)
            mlp_model.fit(X_train, y_train)
            prd_trn = mlp_model.predict(X_train)
            prd_tst = mlp_model.predict(X_test)
            
            recall_score = get_recall_score(labels, y_train, prd_trn, y_test, prd_tst)
        
        lst_variables.append(filetag)
        lst_recall_train.append(recall_score[0])
        lst_recall_test.append(recall_score[1])

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    bar_chart(lst_variables, lst_recall_train, ax=axs[0], title='Air Quality Train set', xlabel='fs', ylabel='recall', rotation=45)
    bar_chart(lst_variables, lst_recall_test, ax=axs[1], title='Air Quality Test set', xlabel='fs', ylabel='recall', rotation=45)
    plt.savefig('images/lab6/feature_selection/air_quality_fs_mlp_study', dpi=300, bbox_inches="tight")
    plt.clf() # cleanup

if __name__ == '__main__':
    main()