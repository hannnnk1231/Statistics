#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
!pip install seaborn
import seaborn as sns
# %%
female = np.array([53, 56, 60, 60, 78, 87, 102, 117, 134, 160, 277])
male = np.array([46, 52, 58, 59, 77, 78, 80, 81, 84, 103, 114, 115, 133, 134, 167, 175])
# %% Q1
from scipy import stats
print(stats.ttest_ind(female, male))
print(stats.ranksums(female, male))
# %%
fig, ax = plt.subplots(2,1,figsize=(20,5))
sns.histplot(ax=ax[0],x=male, bins=np.arange(40, 300, 10))
ax[0].set_xticks(np.arange(40, 300, 10))
ax[0].set_ylabel('Count_male')
sns.histplot(ax=ax[1],x=female, bins=np.arange(40, 300, 10))
ax[1].set_xticks(np.arange(40, 300, 10))
ax[1].set_yticks([0,1,2,3])
ax[1].set_ylabel('Count_female')
# %%
df = pd.DataFrame(male, columns=['age'])
df['gender'] = ["male" for i in range(len(male))]
df
# %%
df2 = pd.DataFrame(female, columns=['age'])
df2['gender'] = ["female" for i in range(len(female))]
df2
# %%
df3 = pd.concat([df, df2])
df3.reset_index(inplace=True)
df3.drop(columns=['index'], inplace=True, axis=0)
df3.sort_values(by=['age'], inplace=True)
df3['rank'] = range(1, len(df3)+1)
df3
# %%
sns.kdeplot(data=df3, x='age', hue='gender', fill=True)
# %%
sns.displot(data=df3, x='age', hue='gender', kind="ecdf" )
# %%
df3[df3['gender']=='female']['rank'].sum()

# %%
len(male)
len(female)
# %%
df3
# %% Q2
g1 = [1, 5, 6, 6, 9, 10, 10, 10, 12, 12, 12, 12, 12, 13, 15, 16, 20, 24, 24, 27, 32, 34, 36, 36, 44]
g1_c = [0,0,1,1,0,1,1,0,1,1,1,1,0,0,0,0,0,1,0,0,1,0,0,0,0]
g1_g = ['treatment with Linoleic acid' for i in range(25)]
g2 = [3, 6, 6, 6, 6, 8, 8, 12, 12, 12, 15, 16, 18, 18, 20, 22, 24, 28, 28, 28, 30, 30, 33, 42]
g2_c = [0,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,1]
g2_g = ['Control group' for i in range(24)]

# %%
df = pd.DataFrame({'day':g1+g2, 'censored': g1_c+g2_c, 'group': g1_g+g2_g})
df
# %%
import kaplanmeier as km
# %%
out=km.fit(df['day'], df['censored'], df['group'])
# %%
km.plot(out)
# %%
from lifelines.statistics import logrank_test
# %%
res = logrank_test(g1, g2, event_observed_A=g1_c, event_observed_B=g2_c)
res.summary