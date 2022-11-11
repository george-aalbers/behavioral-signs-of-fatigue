import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('model-evaluation.csv')

df_train = df[['id','features','target','split','r2_train']]
df_train.rename({'r2_train':'r2'}, axis = 1, inplace = True)
df_train['set'] = pd.Series('train').repeat(df_train.shape[0]).values

df_val = df[['id','features','target','split','r2_val']]
df_val.rename({'r2_val':'r2'}, axis = 1, inplace = True)
df_val['set'] = pd.Series('val').repeat(df_val.shape[0]).values

df_test = df[['id','features','target','split','r2_test']]
df_test.rename({'r2_test':'r2'}, axis = 1, inplace = True)
df_test['set'] = pd.Series('test').repeat(df_test.shape[0]).values

df = pd.concat([df_train, df_val, df_test], axis = 0)

df.columns = ['ID', 'Features', 'Target', 'Month', 'Person-specific R-squared', 'Set']
df.replace({'time':'Time', 
            'app':'App use',
            'time + app':'Time + App use',
            'train':'Train',
            'val':'Val',
            'test':'Test'}, inplace = True)

sns.set(style = 'white')
plot = sns.FacetGrid(data = df[df['Person-specific R-squared'].abs() < 1], row = 'Month', aspect = 2)
plot.map(sns.boxplot, 'Person-specific R-squared', 'Set', 'Features', palette = 'flare', order = ['Train', 'Val', 'Test'], hue_order = ['Time', 'App use', 'Time + App use'])
plot.refline(x = 0, color = "black", lw = 1.5)
plot.add_legend()
plt.setp(plot._legend.get_texts(), fontsize=12.5)
plt.savefig('boxplot.png')