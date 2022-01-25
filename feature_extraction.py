import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import svd

from utils import *

from sklearn.decomposition import PCA

def do_pca(data: pd.DataFrame, file_tag, x_axis, y_axis):
    def plot_variables_before_pca(data, variables):
        plt.figure()
        plt.xlabel(variables[x_axis])
        plt.ylabel(variables[y_axis])
        plt.scatter(data.iloc[:, x_axis], data.iloc[:, y_axis])
        plt.savefig(f'images/lab8/pca/{file_tag}_vars_before_pca.png')
        plt.show()

    def plot_variance_ratio(pca):
        fig = plt.figure(figsize=(4, 4))
        plt.title("Explained variance ratio")
        plt.xlabel("Principal Components")
        plt.ylabel("Ratio")
        x_values = [str(i) for i in range(1, len(pca.components_) + 1)]
        bar_width = 0.5
        ax = plt.gca()
        ax.set_xticklabels(x_values)
        ax.set_ylim(0.0, 1.0)
        ax.bar(x_values, pca.explained_variance_ratio_, width=bar_width)
        ax.plot(pca.explained_variance_ratio_)

        for i, v in enumerate(pca.explained_variance_ratio_):
            ax.text(i, v + 0.5, f'{v * 100: .1f}', ha='center', fontweight='bold')
        plt.savefig(f'images/lab8/pca/{file_tag}_pca_expvar.png')
        plt.show()

    def plot_variables_after_pca(data, transform, variables):
        _, axs = plt.subplots(1, 2, figsize=(2*5, 1*5), squeeze=False)
        axs[0, 0].set_xlabel(variables[x_axis])
        axs[0, 0].set_ylabel(variables[y_axis])
        axs[0, 0].scatter(data.iloc[:, x_axis], data.iloc[:, y_axis])

        axs[0, 1].set_xlabel('PC1')
        axs[0, 1].set_ylabel('PC2')
        axs[0, 1].scatter(transform[:, 0], transform[:, 1])
        plt.savefig(f'images/lab8/pca/{file_tag}_vars_after_pca.png')
        plt.show()

    variables = data.columns.values

    #plot_variables_before_pca(data, variables)

    mean = data.mean(axis=0).to_list()
    centered_data = data - mean

    pca = PCA(n_components=.80, svd_solver='full')
    pca.fit(centered_data)
    components = pca.components_
    variance = pca.explained_variance_

    #plot_variance_ratio(pca)

    transform = pca.transform(data)

    pd.DataFrame(transform).to_csv(f"data/{file_tag}_pca.csv")

    #plot_variables_after_pca(data, transform, variables)
    
def main():
    for filename, target, file_tag, var_1, var_2 in [("data/air_quality_scaled_zscore_fs.csv",
                                                        AQ_TARGET,
                                                        AQ_FILETAG,
                                                        2,
                                                        6), 
                                                    ("data/NYC_collisions_scaled_minmax_fs.csv",
                                                        NYC_TARGET, 
                                                        NYC_FILETAG,
                                                        1,
                                                        74)]:
        data = pd.read_csv(filename)
        data.pop(target)

        do_pca(data, file_tag, var_1, var_2)

if __name__ == '__main__':
    main()
