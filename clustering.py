#%%
from pandas import DataFrame, read_csv
from matplotlib.pyplot import subplots, show
from ds_charts import choose_grid, plot_clusters, plot_line, compute_mse, compute_centroids, bar_chart, multiple_bar_chart, compute_mae
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score

import numpy as np
from scipy.spatial.distance import pdist, squareform

# TODO change pop's and real csvs
data_air: DataFrame = read_csv('data/air_quality_scaled_zscore_fs.csv')
#data_air.pop('id')
data_air.pop('ALARM')
v1_air = 0
v2_air = 4

data_air_pca: DataFrame = read_csv('data/air_quality_pca.csv',dtype=np.float64)
data_air_pca.pop('i')
figname_air_pca = "air_quality_pca"
v2_air_pca = 2

N_CLUSTERS_AIR = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
rows_air, cols_air = choose_grid(len(N_CLUSTERS_AIR))
figname_air = "air_quality"

data_nyc: DataFrame = read_csv('data/NYC_collisions_scaled_minmax_fs.csv')
#data_nyc.pop('id')
data_nyc.pop('PERSON_INJURY')
data_nyc.replace({False: 0, True: 1}, inplace=True)
v1_nyc = 0
v2_nyc = 4

data_nyc_pca: DataFrame = read_csv('data/NYC_collisions_pca.csv',dtype=np.float64)
data_nyc_pca.pop('i')
figname_nyc_pca = "nyc_collisions_pca"

N_CLUSTERS_NYC = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
rows_nyc, cols_nyc = choose_grid(len(N_CLUSTERS_NYC))
figname_nyc = "nyc_collisions"

#%%
def kmeans(data, rows, cols, N_CLUSTERS, v1, v2, figname):
    print("KMeans running")
    mse: list = []
    sc: list = []
    #mae: list = []
    db: list = []
    fig, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        print("-",n)
        k = N_CLUSTERS[n]
        estimator = KMeans(n_clusters=k)
        estimator.fit(data)
        mse.append(estimator.inertia_)
        sc.append(silhouette_score(data, estimator.labels_))
        #mae.append(compute_mae(data.values, estimator.labels_, estimator.means_))
        db.append(davies_bouldin_score(data, estimator.labels_))
        plot_clusters(data, v2, v1, estimator.labels_.astype(float), estimator.cluster_centers_, k, f'KMeans k={k}', ax=axs[i,j])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    fig.savefig(f'images/lab8/clustering/{figname}_kmeans_scatter.png')
    show()
    #fig, ax = subplots(1, 4, figsize=(6, 3), squeeze=False)
    fig, ax = subplots(1, 3, figsize=(9, 3), squeeze=False)
    plot_line(N_CLUSTERS, mse, title='KMeans MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    plot_line(N_CLUSTERS, sc, title='KMeans SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
    plot_line(N_CLUSTERS, db, title='KMeans Davies Bouldin', xlabel='k', ylabel='Davies Bouldin', ax=ax[0, 2], percentage=True)
    #plot_line(N_CLUSTERS, mae, title='KMeans MAE', xlabel='k', ylabel='MAE', ax=ax[0, 3], percentage=True)
    fig.savefig(f'images/lab8/clustering/{figname}_kmeans_line.png')
    show()
    
#%%
#kmeans(data_nyc, rows_nyc, cols_nyc, N_CLUSTERS_NYC, v1_nyc, v2_nyc, figname_nyc)

#%%
#kmeans(data_nyc_pca, rows_nyc, cols_nyc, N_CLUSTERS_NYC, v1_nyc, v2_nyc, figname_nyc_pca)
#%%
#kmeans(data_air, rows_air, cols_air, N_CLUSTERS_AIR, v1_air, v2_air, figname_air)
#%%
# TODO:
# kmeans(data_air_pca, rows_air, cols_air, N_CLUSTERS_AIR, v1_air, v2_air, figname_air_pca)

#%%
def em(data, rows, cols, N_CLUSTERS, v1, v2, figname):
    print("EM running")
    mse: list = []
    sc: list = []
    mae: list = []
    db: list = []
    fig, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        print("-",n)
        k = N_CLUSTERS[n]
        estimator = GaussianMixture(n_components=k)
        estimator.fit(data)
        labels = estimator.predict(data)
        mse.append(compute_mse(data.values, labels, estimator.means_))
        sc.append(silhouette_score(data, labels))
        mae.append(compute_mae(data.values, labels, estimator.means_))
        db.append(davies_bouldin_score(data, labels))
        plot_clusters(data, v2, v1, labels.astype(float), estimator.means_, k,
                        f'EM k={k}', ax=axs[i,j])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    fig.savefig(f'images/lab8/clustering/{figname}_em_scatter.png')
    show()
    fig, ax = subplots(1, 4, figsize=(12, 3), squeeze=False)
    plot_line(N_CLUSTERS, mse, title='EM MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    plot_line(N_CLUSTERS, sc, title='EM SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
    plot_line(N_CLUSTERS, mae, title='EM MAE', xlabel='k', ylabel='MAE', ax=ax[0, 2], percentage=True)
    plot_line(N_CLUSTERS, db, title='EM Davies Bouldin', xlabel='k', ylabel='Davies Bouldin', ax=ax[0, 3], percentage=True)
    fig.savefig(f'images/lab8/clustering/{figname}_em_line.png')
    show()
    
#%% TODO
#em(data_air, rows_air, cols_air, N_CLUSTERS_AIR, v1_air, v2_air, figname_air)
#%% 
#em(data_air_pca, rows_air, cols_air, N_CLUSTERS_AIR, v1_air, v2_air_pca, figname_air_pca)
#%%
#em(data_nyc, rows_nyc, cols_nyc, N_CLUSTERS_NYC, v1_nyc, v2_nyc, figname_nyc)
#%%
#em(data_nyc_pca, rows_nyc, cols_nyc, N_CLUSTERS_NYC, v1_nyc, v2_nyc, figname_nyc_pca)

#%%
def eps_dsbased(data, rows, cols, N_CLUSTERS, v1, v2, figname):
    print("EPS DS running")
    EPS = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    mse: list = []
    sc: list = []
    mae: list = []
    db: list = []
    rows, cols = choose_grid(len(EPS))
    fig, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(EPS)):
        print("-",n)
        estimator = DBSCAN(eps=EPS[n], min_samples=2)
        estimator.fit(data)
        labels = estimator.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k > 1:
            centers = compute_centroids(data, labels)
            mse.append(compute_mse(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            mae.append(compute_mae(data.values, labels, centers))
            db.append(davies_bouldin_score(data, labels))
            plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k, f'DBSCAN eps={EPS[n]} k={k}', ax=axs[i,j])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        else:
            mse.append(0)
            sc.append(0)
    fig.savefig(f'images/lab8/clustering/{figname}_eps_dsbased_scatter.png')
    show()
    fig, ax = subplots(1, 4, figsize=(12, 3), squeeze=False)
    plot_line(EPS, mse, title='DBSCAN MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
    plot_line(EPS, sc, title='DBSCAN SC', xlabel='eps', ylabel='SC', ax=ax[0, 1], percentage=True)
    plot_line(EPS, mae, title='DBSCAN MAE', xlabel='eps', ylabel='MAE', ax=ax[0, 2], percentage=True)
    plot_line(EPS, db, title='DBSCAN Davies Bouldin', xlabel='eps', ylabel='Davies Bouldin', ax=ax[0, 3], percentage=True)
    fig.savefig(f'images/lab8/clustering/{figname}_eps_dsbased_line.png')
    show()
    
#%% TODO
#eps_dsbased(data_air, rows_air, cols_air, N_CLUSTERS_AIR, v1_air, v2_air, figname_air)
#%% TODO
#eps_dsbased(data_air_pca, rows_air, cols_air, N_CLUSTERS_AIR, v1_air, v2_air_pca, figname_air_pca)
#%% TODO
#eps_dsbased(data_nyc, rows_nyc, cols_nyc, N_CLUSTERS_NYC, v1_nyc, v2_nyc, figname_nyc)
#%% TODO
#eps_dsbased(data_nyc_pca, rows_nyc, cols_nyc, N_CLUSTERS_NYC, v1_nyc, v2_nyc, figname_nyc_pca)


#%%
def eps_metric(data, rows, cols, v1, v2, figname):
    print("EPS metric running")
    METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
    distances = []
    for m in METRICS:
        dist = np.mean(np.mean(squareform(pdist(data.values, metric=m))))
        distances.append(dist)

    print('AVG distances among records', distances)
    distances[0] *= 0.6
    distances[1] = 80
    distances[2] *= 0.6
    distances[3] *= 0.1
    distances[4] *= 0.15
    print('CHOSEN EPS', distances)

    mse: list = []
    sc: list = []
    mae: list = []
    db: list = []
    rows, cols = choose_grid(len(METRICS))
    fig, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(METRICS)):
        print("-",n)
        estimator = DBSCAN(eps=distances[n], min_samples=2, metric=METRICS[n])
        estimator.fit(data)
        labels = estimator.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k > 1:
            centers = compute_centroids(data, labels)
            mse.append(compute_mse(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            mae.append(compute_mae(data.values, labels, centers))
            db.append(davies_bouldin_score(data, labels))
            plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k, f'DBSCAN metric={METRICS[n]} eps={distances[n]:.2f} k={k}', ax=axs[i,j])
        else:
            mse.append(0)
            sc.append(0)
            mae.append(0)
            db.append(0)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    fig.savefig(f'images/lab8/clustering/{figname}_eps_metric_scatter.png')
    show()
    fig, ax = subplots(1, 4, figsize=(12, 3), squeeze=False)
    bar_chart(METRICS, mse, title='DBSCAN MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
    bar_chart(METRICS, sc, title='DBSCAN SC', xlabel='metric', ylabel='SC', ax=ax[0, 1], percentage=True)
    bar_chart(METRICS, mae, title='DBSCAN MAE', xlabel='metric', ylabel='MAE', ax=ax[0, 2])
    bar_chart(METRICS, db, title='DBSCAN Davies Bouldin', xlabel='metric', ylabel='Davies Bouldin', ax=ax[0, 3], percentage=True)
    fig.savefig(f'images/lab8/clustering/{figname}_eps_metric_bar.png')
    show()
    
#%% TODO
#eps_metric(data_air, rows_air, cols_air, v1_air, v2_air, figname_air)
#%% TODO
#eps_metric(data_air_pca, rows_air, cols_air, v1_air, v2_air_pca, figname_air_pca)
#%% TODO
#eps_metric(data_nyc, rows_nyc, cols_nyc, v1_nyc, v2_nyc, figname_nyc)

#%% TODO
#eps_metric(data_nyc_pca, rows_nyc, cols_nyc, v1_nyc, v2_nyc, figname_nyc_pca)

#%%
def hierarchical(data, rows, cols, N_CLUSTERS, v1, v2, figname):
    print("Hierarquical running")
    mse: list = []
    sc: list = []
    mae: list = []
    db: list = []
    rows, cols = choose_grid(len(N_CLUSTERS))
    _, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        print("-",n)
        k = N_CLUSTERS[n]
        estimator = AgglomerativeClustering(n_clusters=k)
        estimator.fit(data)
        labels = estimator.labels_
        centers = compute_centroids(data, labels)
        mse.append(compute_mse(data.values, labels, centers))
        sc.append(silhouette_score(data, labels))
        mae.append(compute_mae(data.values, labels, centers))
        db.append(davies_bouldin_score(data, labels))
        plot_clusters(data, v2, v1, labels, centers, k, f'Hierarchical k={k}', ax=axs[i,j])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    show()
    fig, ax = subplots(1, 4, figsize=(12, 3), squeeze=False)
    plot_line(N_CLUSTERS, mse, title='Hierarchical MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    plot_line(N_CLUSTERS, sc, title='Hierarchical SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
    plot_line(N_CLUSTERS, mae, title='Hierarchical MAE', xlabel='k', ylabel='MAE', ax=ax[0, 2], percentage=True)
    plot_line(N_CLUSTERS, db, title='Hierarchical Davies Bouldin', xlabel='k', ylabel='Davies Bouldin', ax=ax[0, 3], percentage=True)
    show()
    
    METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
    LINKS = ['complete', 'average']
    k = 3
    values_mse = {}
    values_sc = {}
    values_mae = {}
    values_db = {}
    rows = len(METRICS)
    cols = len(LINKS)
    _, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    for i in range(len(METRICS)):
        mse: list = []
        sc: list = []
        m = METRICS[i]
        for j in range(len(LINKS)):
            link = LINKS[j]
            estimator = AgglomerativeClustering(n_clusters=k, linkage=link, affinity=m )
            estimator.fit(data)
            labels = estimator.labels_
            centers = compute_centroids(data, labels)
            mse.append(compute_mse(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            mae.append(compute_mae(data.values, labels, centers))
            db.append(davies_bouldin_score(data, labels))
            plot_clusters(data, v2, v1, labels, centers, k, f'Hierarchical k={k} metric={m} link={link}', ax=axs[i,j])
        values_mse[m] = mse
        values_sc[m] = sc
        values_mae[m] = mae
        values_db[m] = db
    fig.savefig(f'images/lab8/clustering/{figname}_hierarchical_scatter.png')
    show()
    
    _, ax = subplots(1, 4, figsize=(12, 3), squeeze=False)
    multiple_bar_chart(LINKS, values_mse, title=f'Hierarchical MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
    multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[0, 1], percentage=True)
    multiple_bar_chart(LINKS, values_mae, title=f'Hierarchical MAE', xlabel='metric', ylabel='MAE', ax=ax[0, 2])
    multiple_bar_chart(LINKS, values_db, title=f'Hierarchical Davies Bouldin', xlabel='metric', ylabel='Davies Bouldin', ax=ax[0, 3], percentage=True)
    fig.savefig(f'images/lab8/clustering/{figname}_hierarquical_bar.png')
    show()

    
#%% TODO
#hierarchical(data_air, rows_air, cols_air, N_CLUSTERS_AIR, v1_air, v2_air, figname_air)
#%% TODO
#hierarchical(data_air_pca, rows_air, cols_air, N_CLUSTERS_AIR, v1_air, v2_air_pca, figname_air_pca)
#%% TODO
#hierarchical(data_nyc, rows_nyc, cols_nyc, N_CLUSTERS_NYC, v1_nyc, v2_nyc, figname_nyc)

#%% TODO
# hierarchical(data_nyc_pca, rows_nyc, cols_nyc, N_CLUSTERS_NYC, v1_nyc, v2_nyc, figname_nyc_pca)
