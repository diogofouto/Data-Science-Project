import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *

from sklearn.decomposition import PCA

def do_pca(data: pd.DataFrame, file_tag):
    def plot_variables_before_pca(data, variables):
        plt.figure()
        plt.xlabel(variables[x_axis])
        plt.ylabel(variables[y_axis])
        plt.scatter(data.iloc[:, x_axis], data.iloc[:, y_axis])
        plt.savefig(f'data/lab8/pca/{file_tag}_vars_before_pca.png')
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
        plt.savefig(f'data/lab8/pca/{file_tag}_pca_expvar.png')
        plt.show()

    def plot_variables_after_pca(data, transform, variables):
        _, axs = plt.subplots(1, 2, figsize=(2*5, 1*5), squeeze=False)
        axs[0, 0].set_xlabel(variables[x_axis])
        axs[0, 0].set_ylabel(variables[y_axis])
        axs[0, 0].scatter(data.iloc[:, x_axis], data.iloc[:, y_axis])

        axs[0, 1].set_xlabel('PC1')
        axs[0, 1].set_ylabel('PC2')
        axs[0, 1].scatter(transform[:, 0], transform[:, 1])
        plt.savefig(f'data/lab8/pca/{file_tag}_vars_after_pca.png')
        plt.show()

    variables = data.columns.values
    x_axis = 4
    y_axis = 7

    plot_variables_before_pca(data, variables)

    mean = data.mean(axis=0).to_list()
    centered_data = data - mean

    pca = PCA()
    pca.fit(centered_data)
    components = pca.components_
    variance = pca.explained_variance_

    plot_variance_ratio(pca)

    transform = pca.transform(data)

    plot_variables_after_pca(data, transform, variables)
    
def main():
    for filename, target, file_tag in [(AQ_FILENAME, AQ_TARGET, AQ_FILETAG), 
                                        (NYC_FILENAME, NYC_TARGET, NYC_FILETAG)]:
        data = pd.read_csv(filename)
        data.pop('id')
        data.pop(target)

        do_pca(data, file_tag)
