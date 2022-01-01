#%%
from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.neural_network import MLPClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score

#%% 
# TODO FIX ADD RIGHT FILE NAMES
file_tag_air = 'air'
filename_air = 'data/air'
target_air = 'DANGER'

train_air: DataFrame = read_csv(f'{filename_air}_train.csv')
trnY_air: ndarray = train_air.pop(target_air).values
trnX_air: ndarray = train_air.values
labels_air = unique(trnY_air)
labels_air.sort()

test_air: DataFrame = read_csv(f'{filename_air}_test.csv')
tstY_air: ndarray = test_air.pop(target_air).values
tstX_air: ndarray = test_air.values

lr_type_air = ['constant', 'invscaling', 'adaptive']
max_iter_air = [100, 300, 500, 750, 1000]
learning_rate_air = [.1, .3, .5, .7, .9]
best_air = ('', 0, 0)
last_best_air = 0
best_model_air = None

#%% 
# TODO FIX ADD RIGHT FILE NAMES
file_tag_nyc = 'NYC_collisions'
filename_nyc = 'data/NYC_collisions'
target_nyc = 'PERSON_INJURY'

train_nyc: DataFrame = read_csv(f'{filename_nyc}_train.csv')
trnY_nyc: ndarray = train_nyc.pop(target_nyc).values
trnX_nyc: ndarray = train_nyc.values
labels_nyc = unique(trnY_nyc)
labels_nyc.sort()

test_nyc: DataFrame = read_csv(f'{filename_nyc}_test.csv')
tstY_nyc: ndarray = test_nyc.pop(target_nyc).values
tstX_nyc: ndarray = test_nyc.values

lr_type_nyc = ['constant', 'invscaling', 'adaptive']
max_iter_nyc = [100, 300, 500, 750, 1000]
learning_rate_nyc = [.1, .3, .5, .7, .9]
best_nyc = ('', 0, 0)
last_best_nyc = 0
best_model_nyc = None

#%%
def mlp(lr_types, learning_r, max_iter, file_tag, train_X, train_Y, test_X, test_Y, labels):
    cols = len(lr_types)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(lr_types)):
        d = lr_types[k]
        values = {}
        for lr in learning_r:
            yvalues = []
            for n in max_iter:
                mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                    learning_rate_init=lr, max_iter=n, verbose=False)
                mlp.fit(train_X, train_Y)
                prdY = mlp.predict(test_X)
                yvalues.append(accuracy_score(test_Y, prdY))
                if yvalues[-1] > last_best:
                    best = (d, lr, n)
                    last_best = yvalues[-1]
                    best_model = mlp
            values[lr] = yvalues
        multiple_line_chart(max_iter, values, ax=axs[0, k], title=f'MLP with lr_type={d}',
                            xlabel='mx iter', ylabel='accuracy', percentage=True)
    savefig(f'images/{file_tag}_mlp_study.png')
    show()
    print(f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with accuracy={last_best}')


    prd_trn = best_model.predict(train_X)
    prd_tst = best_model.predict(test_X)
    plot_evaluation_results(labels, train_Y, prd_trn, test_Y, prd_tst)
    savefig(f'images/{file_tag}_mlp_best.png')
    show()
    
#%%
mlp(lr_type_air, learning_rate_air, max_iter_air, file_tag_air, trnX_air, trnY_air, tstX_air, tstY_air, labels_air)
#%%
mlp(lr_type_nyc, learning_rate_nyc, max_iter_nyc, file_tag_nyc, trnX_nyc, trnY_nyc, tstX_nyc, tstY_nyc, labels_nyc)