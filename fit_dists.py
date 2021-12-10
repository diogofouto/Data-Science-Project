import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, lognorm
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, choose_grid, multiple_line_chart, HEIGHT

def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(np.log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
    return distributions

def histogram_with_distributions(ax: plt.Axes, series: pd.Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    print("-- Computing Distributions --")
    distributions = compute_known_distributions(values)
    print("-- Plotting --")
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')
    print("-- Plotting Ended --")

def fit_distributions(data, path):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    splits = np.array_split(numeric_vars, 13)

    k = 0

    for split in splits:
        file = path + "_{}".format(k) + ".png"
        k += 1

        print("- Starting Split {}/13 -".format(k))
        rows, cols = choose_grid(len(split))
        fig, axs = plt.subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
        i, j = 0, 0
        for n in range(len(split)):
            histogram_with_distributions(axs[i, j], data[split[n]].dropna(), split[n])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        print("-- Saving Fig --")
        plt.savefig(file, bbox_inches='tight')
        print("-- Fig Saved --")
        del fig
        print("-- Fig deleted --")
        print("- Ending Split -")

AIR_QUALITY_FILE = "data/air_quality_tabular.csv"

register_matplotlib_converters()

air_data = pd.read_csv(AIR_QUALITY_FILE, index_col="FID", parse_dates=True, na_values='', infer_datetime_format=True)

fit_distributions(air_data, "images/lab1/distribution/air_quality_histograms_fits")
