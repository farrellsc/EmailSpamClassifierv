import matplotlib.pyplot as plt
import numpy as np


accus = np.array([
    [82.5, 84.5, 81.75, 87.5],
    [82.5, 84.5, 81.5, 87.43],
    [95.0, 97.0, 96.0, 95.45],
    [100.0, 97.5, 95.75, 95.59],
    [85.0, 94.0, 92.5, 90.62],
    [75.0, 93.5, 93.25, 95.31]
])      # 6 algorithms * 4 datasets
algoNames = ['Knn', 'sklearn-Knn', 'naiveBayes', "sklearn-naiveBayes", "decisionTree", "sklearn-decisionTree"]
datasetNames = ['Random-200', 'Random-1000', 'Random-2000', 'AllData-3621']
colors = ['aqua', 'navy', 'lightcoral', 'darkred', 'lightgreen', 'darkolivegreen']

n_groups = len(accus)
n_algos = len(accus[0])
bar_width = 0.1
opacity = 0.8
index = np.arange(n_algos)

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

for i in range(n_groups):
    ax.bar(index + i*bar_width, accus[i, :], bar_width, color=colors[i], alpha=opacity, label=algoNames[i])

ax.set_xlabel('Data Size')
ax.set_ylabel('Accuracy')
ax.set_title('Algorithm Performance Comparison')
ax.set_xticks(index + 5 * bar_width / 2)
ax.set_xticklabels(datasetNames)

mng = plt.get_current_fig_manager()
mng.window.showMaximized()
ax.legend(loc=1, bbox_to_anchor=(1.25, 1), borderaxespad=0.2)
plt.show("/root/EmailSpamClassifierv/data/analysis/algoRes.png")
