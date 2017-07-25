import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

add = ''
ynames = ["Relative Frobenius norm of Theta error"]
# ynames = ["eNMI", "error Z", "error fro theta", "time"]
# df_results = pd.DataFrame.from_csv('temp_results{}.csv'.format(add))
df_results = pd.DataFrame.from_csv('temp_results.csv')
for exp_name in ['Varying pure nodes number']: # list(df_results['experiment name'].unique()):
    plt.figure(figsize=(7, 7))
    data = df_results.loc[(df_results['experiment name'] == exp_name)]
    for i, yname in enumerate(ynames):
        print(i)
        plt.subplot(1, len(ynames), i + 1)
        data = data[np.logical_or(data['x'] % 3 == 0, (data['x'] == 1400))]
        ax = sns.pointplot(x="x", y=yname, hue="method", data=data)
        #print(np.arange(1,15,3))
        #ax.set(xticks=np.arange(1,15,3), xlabels = [])
        plt.title(exp_name)
        plt.xlabel(list(data['x name'])[0])
        plt.legend(loc='lower right')
        plt.grid()
    plt.tight_layout()
    plt.savefig(exp_name + '{}.pdf'.format(add))
    plt.savefig(exp_name + '{}.png'.format(add))
    plt.close()
