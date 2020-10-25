#%%
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
def weibull(x, beta, gamma):
    return ((gamma*(x**(gamma-1)))/(2*(beta**gamma)))*math.exp(-(x**gamma)/(2*beta**gamma))

def log_likelihood_func(x, beta, gamma):
    return sum([np.log(weibull(_x, beta, gamma)) for _x in x])

def empirical_func(X, x):
    return sum(X<x)/len(X)

def quantile_func(beta, gamma, p):
    return beta*(-2*np.log(1-p))**(1/gamma)

#%%
x = """9.4 12.7  3.9  9.8  9.5 15.0  8.1 15.7  7.8  9.1 14.5 10.3  9.0 11.1  5.2 3.0  5.7 10.1  7.6 17.7  6.4  9.1 12.9  9.6  6.2  8.6 17.0 17.0 16.8  5.1 8.6 20.1  4.6  7.3 14.3  8.7  8.8  4.4  8.2 10.8  5.3  5.5 10.8 21.9  4.4 6.9  5.2  6.9 12.2 11.6 16.0 16.8  9.9  7.3  4.6  3.2  2.4  5.1  5.6  3.5 6.4  3.9  9.9  7.7 10.3 15.9 17.3 12.7 10.0  9.0 16.2 12.2  6.6 11.2 14.3 14.5 13.0 11.9  8.8 10.0 18.5 23.2 21.5  8.9  6.4  9.9  7.9  8.0 12.9  8.3 6.5 9.8 12.9 16.2  5.9 11.4  4.6  7.2 12.5  5.3 15.8  5.5 10.9  8.6 15.0 10.3 7.8 11.8 10.7 12.6 11.5 16.1  5.9  6.3  9.6 19.2 13.9  6.9  7.6  5.1 10.3  4.1 15.2  9.8  9.9 11.4  6.9 15.6  5.8 10.7 15.0  6.8  6.0  8.4 10.4 8.7 7.8 21.6 14.9  5.0 18.5 10.3"""
x.split(' ')
x = np.array(sorted([float(_x) for _x in x.split(' ') if _x!='']))

# Question 1.C
#%%
gamma_best = 2.45
beta_best = (np.sum((x**gamma_best))/(2*len(x)))**(1/gamma_best)
print('shape:', gamma_best, 'scale:', beta_best)
# %%
y = [weibull(_x, beta_best, gamma_best) for _x in x]
plt.plot(x,y,'-')
plt.show()

# Question 1.D
#%%
gamma_best2 = 2
beta_best2 = (np.sum((x**gamma_best2))/(2*len(x)))**(1/gamma_best2)
print('shape:', gamma_best2, 'scale:', beta_best2)
from scipy.stats.distributions import chi2
LR = -2*np.log(np.exp(log_likelihood_func(x, beta_best2, gamma_best2))/np.exp(log_likelihood_func(x, beta_best, gamma_best)))
print(LR)
p = chi2.sf(LR, 1) # L2 has 1 DoF more than L1
p

# Question 1.E
# %%
x_q = [quantile_func(beta_best, gamma_best, p) for p in np.arange(0.01,1,0.01)]
y_q = [np.quantile(x,p) for p in np.arange(0.01,1,0.01)]
x_qr = [quantile_func(beta_best,gamma_best, p) for p in np.arange(0.1,1,0.1)]
y_qr = [np.quantile(x,p) for p in np.arange(0.1,1,0.1)]
x_45 = np.arange(0,30,1)
y_45 = x_45
plt.plot(y_q,x_q,'o', mfc='#5B5B5B', mec='#000000', ms=5)
plt.plot(y_qr,x_qr,'o', mfc='#E92A2A', mec='#000000', ms=5)
plt.plot(x_45, y_45, 'k-')
plt.grid(color='#D1D1D1', linestyle='dotted')
plt.ylabel("Weibull(2.45, 8.69) theoretical quantiles")
plt.xlabel("Data quantiles")
plt.savefig('fig1')
plt.show()

# Question 2.C
# %%
from scipy.stats import wilcoxon
y1 = [5,4,12,7,17,4,3,4,6,15,9,7]
y2 = [19,10,4,10,17,12,17,14,3,3,9,10]
wilcoxon(y1,y2,mode='exact')
# WilcoxonResult(statistic=15.5, pvalue=0.21946693040984377)

# Q3.A
#%%
df = pd.read_csv('Soil.csv')
#%%
x1 = np.array(df['Place_1'])[:-8]
x2 = np.array(df['Place_2'])
# %%
x_empirical = np.arange(5, 28, 0.1)
y_empirical = [empirical_func(x1, _x) for _x in x_empirical]
y_empirical2 = [empirical_func(x2, _x) for _x in x_empirical]
plt.step(x_empirical, y_empirical)
plt.step(x_empirical, y_empirical2)
plt.grid(color='#D1D1D1', linestyle='dotted')
plt.xlabel("X")
plt.ylabel("Cumulative Probability")
plt.legend(['Place 1', 'Place 2'])
plt.savefig('fig3')
plt.show()

# Q3.B
# %%
s1 = np.std(x1)
s2 = np.std(x2)
print(s1, s2)
# %%
sp = np.sqrt(((len(x1)-1)*s1**2+(len(x2)-1)*s2**2)/(len(x1)+len(x2)-2))
sp
# %%
t = (11.42-10.65)/(sp*np.sqrt(1/len(x1)+1/len(x2)))
t

#Q3.C
# %%
m = len(x1)
n = len(x2)
E = m*(m+n+1)/2
SD = np.sqrt(m*n*(m+n+1)/12)
# %%
xx1 = [(_x,1) for _x in x1]
xx2 = [(_x,2) for _x in x2]
xx = xx1+xx2
xx = sorted(xx)
res = []
for i, _x in enumerate(xx):
    if _x[1] == 1:
        res.append(i)
res
# %%
W_m = sum(res)

# %%
df3 = pd.read_csv('soil2.csv')
# %%
W_m = sum(df3[df3['Place']==1]['Rank'])
# %%
Z = (W_m-E)/SD
Z