import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.io as pio
import plotly_express as px
import pandas as pd
import seaborn as sns
import time
from scipy.stats import linregress
import statsmodels.formula.api as smf
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_diabetes

data = load_diabetes()

df  = pd.DataFrame(data.data)

df.columns= data.feature_names
df['y']=data.target

os.makedirs('HOMEWORK7ADDPLOTS', exist_ok=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(df['sex'], df['bmi'], 
          df['y'],marker='o',alpha=0.3,s=50)

plt.style.use('ggplot')

plt.savefig(f'HOMEWORK7ADDPLOTS/3Dscatter.png', dpi=300)
plt.close()

for col1_idx, column1 in enumerate(df.columns):
    for col2_idx, column2 in enumerate(df.columns):
        if col1_idx < col2_idx:
            fig, axes = plt.subplots(1, 1, figsize=(5, 5))
            axes.scatter(df[column1], df[column2], label=f'{column1} to {column2}', color='green', marker='o')
            axes.set_title(f'{column1} to {column2}')
            axes.set_xlabel(column1)
            axes.set_ylabel(column2)
            axes.legend()
            plt.savefig(f'HOMEWORK7ADDPLOTS/DIABETES_{column1}_{column2}_scatter.png', dpi=300)
            plt.close(fig)

plt.close()

sns.set()

sorted_by_bp_df = df.sort_values('bp')
sorted_by_bmi_df = df.sort_values('bmi')
sns.lineplot('bp', 'y', data=sorted_by_bp_df)
sns.lineplot('bmi', 'y', data=sorted_by_bmi_df)
plt.legend(['bp vs y', 'bp vs bmi'])
plt.savefig('HOMEWORK7ADDPLOTS/attemptsort_lineplot.png')
plt.clf()