import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ds_charts import choose_grid, plot_clusters, plot_line
from utils import AQ_FILENAME, AQ_FILETAG, AQ_TARGET, NYC_FILENAME, NYC_FILETAG, NYC_TARGET

AQ_CLUSTERS = [2, 3, 5, 8, 13, 21, 34, 55, 89]
NYC_CLUSTERS = [2, 3, 5, 8, 13, 21, 34, 55, 89]

AQ_VAR_1 = None
AQ_VAR_2 = None

NYC_VAR_1 = None
NYC_VAR_2 = None

def kmeans_study(data: pd.DataFrame, target, n_clusters: list, var_1, var_2, file_tag):
    mse: list = []
    sc: list = []

    data.pop('id')
    data.pop(target)
    rows, cols = choose_grid(len(n_clusters))

    _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0

    for n in range(len(n_clusters)):
        k = n_clusters[n]
        estimator = KMeans(n_clusters=k)
        estimator.fit(data)
        mse.append(estimator.inertia_)
        sc.append(silhouette_score(data, estimator.labels_))
        plot_clusters(data, var_2, var_1, estimator.labels_.astype(float), estimator.cluster_centers_, k, f'KMeans k={k}', ax=axs[i,j])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.savefig(f'images/lab8/kmeans/{file_tag}_kmeans_study.png')
    plt.show()

    return mse, sc

def plot_results(n_clusters, mse, sc, file_tag):
    _, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
    plot_line(n_clusters, mse, title='KMeans MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    plot_line(n_clusters, sc, title='KMeans SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
    plt.savefig(f'images/lab8/kmeans/{file_tag}_kmeans_metrics.png')
    plt.show()

def main():
    for filename, file_tag, target, n_clusters, var_1, var_2 in [(  AQ_FILENAME,
                                                                    AQ_FILETAG,
                                                                    AQ_TARGET,
                                                                    AQ_CLUSTERS,
                                                                    AQ_VAR_1,
                                                                    AQ_VAR_2),
                                                                   (NYC_FILENAME,
                                                                    NYC_FILETAG,
                                                                    NYC_TARGET,
                                                                    NYC_CLUSTERS,
                                                                    NYC_VAR_1,
                                                                    NYC_VAR_2)]:
        
        data = pd.read_csv(filename)
        mse, sc = kmeans_study(data, target, n_clusters, var_1, var_2, file_tag)
        plot_results(n_clusters, mse, sc, file_tag)

if __name__ == '__main__':
    main()