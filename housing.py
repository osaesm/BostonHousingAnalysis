import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


f = pd.read_csv('./all/train.csv')
#print(f)

f.hist(bins=50, figsize=(20,15))
plt.savefig("attribute_histogram_plots")
    #plt.show()
corr_matrix = f.corr()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(f.describe())
    print(corr_matrix)

attributes = ["crim", "indus", "nox", "rad",'tax']
scatter_matrix(f[attributes], figsize=(12, 8))
plt.savefig('matrix.png')