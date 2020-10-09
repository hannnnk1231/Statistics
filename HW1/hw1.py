#%%
import math
import numpy as np
import matplotlib.pyplot as plt

# %%
def rayleigh(x, beta):
    return (x/(beta**2))*math.exp(-(x**2)/(2*beta**2))

def rayleigh_distribution(x, beta):
    return 1-math.exp(-(x**2/(2*beta**2)))

def score_func(x, beta):
    return sum([np.log(rayleigh(_x, beta)) for _x in x])

def empirical_func(X, x):
    return sum(X<x)/len(X)

def quantile_func(beta, p):
    return beta*np.sqrt(-2*np.log(1-p))

#%%
x = """9.4 12.7  3.9  9.8  9.5 15.0  8.1 15.7  7.8  9.1 14.5 10.3  9.0 11.1  5.2 3.0  5.7 10.1  7.6 17.7  6.4  9.1 12.9  9.6  6.2  8.6 17.0 17.0 16.8  5.1 8.6 20.1  4.6  7.3 14.3  8.7  8.8  4.4  8.2 10.8  5.3  5.5 10.8 21.9  4.4 6.9  5.2  6.9 12.2 11.6 16.0 16.8  9.9  7.3  4.6  3.2  2.4  5.1  5.6  3.5 6.4  3.9  9.9  7.7 10.3 15.9 17.3 12.7 10.0  9.0 16.2 12.2  6.6 11.2 14.3 14.5 13.0 11.9  8.8 10.0 18.5 23.2 21.5  8.9  6.4  9.9  7.9  8.0 12.9  8.3 6.5 9.8 12.9 16.2  5.9 11.4  4.6  7.2 12.5  5.3 15.8  5.5 10.9  8.6 15.0 10.3 7.8 11.8 10.7 12.6 11.5 16.1  5.9  6.3  9.6 19.2 13.9  6.9  7.6  5.1 10.3  4.1 15.2  9.8  9.9 11.4  6.9 15.6  5.8 10.7 15.0  6.8  6.0  8.4 10.4 8.7 7.8 21.6 14.9  5.0 18.5 10.3"""
x.split(' ')
x = np.array(sorted([float(_x) for _x in x.split(' ') if _x!='']))

# Section B
# %%
beta_best = math.sqrt(sum(x**2)/(2*len(x))) # 7.872894308240928
print(beta_best)
# %% 
theta = np.arange(0,25, 0.1)
theta_y = [score_func(x, _theta) for _theta in theta]
plt.plot(theta, theta_y, '-')
plt.title("Score function")
plt.show()

# Section C
#%%
y_dist = [rayleigh_distribution(_x, beta_best) for _x in x]
x_empirical = np.arange(0, 24, 0.1)
y_empirical = [empirical_func(x, _x) for _x in x_empirical]
plt.plot(x, y_dist, '-')
plt.step(x_empirical, y_empirical)
plt.grid(color='#D1D1D1', linestyle='dotted')
plt.xlabel("X")
plt.ylabel("Cumulative Probability")
plt.show()

# Section D
# %%
x_q = [quantile_func(beta_best,p) for p in np.arange(0.01,1,0.01)]
y_q = [np.quantile(x,p) for p in np.arange(0.01,1,0.01)]
x_qr = [quantile_func(beta_best,p) for p in np.arange(0.1,1,0.1)]
y_qr = [np.quantile(x,p) for p in np.arange(0.1,1,0.1)]
x_45 = np.arange(0,30,1)
y_45 = x_45
plt.plot(x_q,y_q,'o', mfc='#5B5B5B', mec='#000000', ms=5)
plt.plot(x_qr,y_qr,'o', mfc='#E92A2A', mec='#000000', ms=5)
plt.plot(x_45, y_45, 'k-')
plt.grid(color='#D1D1D1', linestyle='dotted')
plt.xlabel("Rayleigh(7.87289) theoretical quantiles")
plt.ylabel("Data quantiles")
plt.show()

# %%
y = [rayleigh(_x, beta_best) for _x in x]
plt.plot(x,y,'-')
plt.show()
# %%
