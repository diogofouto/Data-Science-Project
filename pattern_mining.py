import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ds_charts import multiple_line_chart, plot_line, dummify
from  mlxtend.frequent_patterns import apriori, association_rules


MIN_SUP: float = 0.001
MIN_CONF: float = 0.1

var_min_sup = [0.2, 0.1] + [i*MIN_SUP for i  in range(100, 0, -10)]
var_min_conf = [i * MIN_CONF for i in range(10, 5, -1)]


def get_patterns(data, filetag):
	patterns: pd.DataFrame = apriori(data, min_support=MIN_SUP, use_colnames=True, verbose=True, low_memory=True)
	print(len(patterns),'patterns')
	nr_patterns = []
	for sup in var_min_sup:
	    pat = patterns[patterns['support']>=sup]
	    nr_patterns.append(len(pat))

	plt.figure(figsize=(6, 4))
	plot_line(var_min_sup, nr_patterns, title='Nr Patterns x Support', xlabel='support', ylabel='Nr Patterns')
	plt.savefig(f"images/lab9/pattern_mining/{filetag}_pat_sup.png")

	return patterns


def get_association_rules(patterns, filetag):
	rules = association_rules(patterns, metric='confidence', min_threshold=MIN_CONF*5, support_only=False)
	print(f'\tfound {len(rules)} rules for {filetag}')
	return rules


def plot_top_rules(rules: pd.DataFrame, metric: str, per_metric: str, filetag) -> None:
    _, ax = plt.subplots(figsize=(6, 3))
    ax.grid(False)
    ax.set_axis_off()
    ax.set_title(f'TOP 10 per Min {per_metric} - {metric}', fontweight="bold")
    text = ''
    cols = ['antecedents', 'consequents']
    rules[cols] = rules[cols].applymap(lambda x: tuple(x))
    for i in range(len(rules)):
        rule = rules.iloc[i]
        text += f"{rule['antecedents']} ==> {rule['consequents']}"
        text += f"(s: {rule['support']:.2f}, c: {rule['confidence']:.2f}, lift: {rule['lift']:.2f})\n"
    ax.text(0, 0, text)
    plt.savefig(f"images/lab9/pattern_mining/{filetag}_top_rules_{metric}_{per_metric}.png")


def analyse_per_metric(rules: pd.DataFrame, metric: str, metric_values: list, filetag) -> list:
    print(f'Analyse per {metric}...')
    conf = {'avg': [], 'top25%': [], 'top10': []}
    lift = {'avg': [], 'top25%': [], 'top10': []}
    top_conf = []
    top_lift = []
    nr_rules = []
    for m in metric_values:
        rs = rules[rules[metric] >= m]
        nr_rules.append(len(rs))
        conf['avg'].append(rs['confidence'].mean(axis=0))
        lift['avg'].append(rs['lift'].mean(axis=0))

        top_conf = rs.nlargest(int(0.25*len(rs)), 'confidence')
        conf['top25%'].append(top_conf['confidence'].mean(axis=0))
        top_lift = rs.nlargest(int(0.25*len(rs)), 'lift')
        lift['top25%'].append(top_lift['lift'].mean(axis=0))

        top_conf = rs.nlargest(10, 'confidence')
        conf['top10'].append(top_conf['confidence'].mean(axis=0))
        top_lift = rs.nlargest(10, 'lift')
        lift['top10'].append(top_lift['lift'].mean(axis=0))

    _, axs = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)
    multiple_line_chart(metric_values, conf, ax=axs[0, 0], title=f'Avg Confidence x {metric}',
                           xlabel=metric, ylabel='Avg confidence')
    multiple_line_chart(metric_values, lift, ax=axs[0, 1], title=f'Avg Lift x {metric}',
                           xlabel=metric, ylabel='Avg lift')
    plt.savefig(f"images/lab9/pattern_mining/{filetag}_quality_evaluation_per_{metric}.png")

    plot_top_rules(top_conf, 'confidence', metric, filetag)
    plot_top_rules(top_lift, 'lift', metric, filetag)

    return nr_rules


def main():
	for (filename, filetag) in [('data/air_quality_discretized_equal_width.csv', 'air_quality_discretized_equal_width'),
								('data/air_quality_discretized_equal_frequency.csv', 'air_quality_discretized_equal_frequency'),
								('data/NYC_collisions_discretized_no_numeric.csv', 'NYC_collisions_discretized_no_numeric'),
								('data/NYC_collisions_discretized_equal_width.csv', 'NYC_collisions_discretized_equal_width'),
								('data/NYC_collisions_discretized_equal_frequency.csv', 'NYC_collisions_discretized_equal_frequency')]:
		# Data
		data = pd.read_csv(filename)
		data = dummify(data, data.columns)

		# Patterns
		patterns = get_patterns(data, filetag)

		# Rules
		rules = get_association_rules(patterns, filetag)

		# Quality Evaluation per support
		nr_rules_sp = analyse_per_metric(rules, 'support', var_min_sup, filetag)
		plt.figure(figsize=(6, 4))
		plot_line(var_min_sup, nr_rules_sp, title='Nr Rules x Support', xlabel='support', ylabel='Nr Rules', percentage=False)
		plt.savefig(f"images/lab9/pattern_mining/{filetag}_rules_x_support.png")

		# Quality Evaluation per confidence
		nr_rules_cf = analyse_per_metric(rules, 'confidence', var_min_conf, filetag)
		plt.figure(figsize=(6, 4))
		plot_line(var_min_conf, nr_rules_cf, title='Nr Rules x Confidence', xlabel='confidence', ylabel='Nr Rules', percentage=False)
		plt.savefig(f"images/lab9/pattern_mining/{filetag}_rules_x_confidence.png")


if __name__ == '__main__':
	main()