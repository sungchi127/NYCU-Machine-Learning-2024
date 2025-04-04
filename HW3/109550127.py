import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import matmul as mul

# Univariate gaussian data generator
def Gaussian(mean, var): # Box-Muller transform
    U, V = np.random.uniform(-1, 1, 2)
    S = U**2 + V**2
    while S >= 1:
        U, V = np.random.uniform(-1, 1, 2)
        S = U**2 + V**2
    factor = np.sqrt(-2 * np.log(S) / S) 
    return mean + np.sqrt(var) * U * factor

# Polynomial basis linear model data generator
def CreateData(n, a, w):
    x = np.random.uniform(-1, 1)
    y = sum(w[i] * (x**i) for i in range(n)) + Gaussian(0, a)
    return x, y


def PolyM(x, N):
    return np.array([[x**i for i in range(N)]])

def Output(x, y, m, var, post_m, post_var):
    print("Add data point (%.5f, %.5f)" % (x, y))
    print("\nPosterior mean:")
    print(m)
    print("\nPosterior variance:")
    print(var)
    print("\nPredictive distribution ~ N(%.5f, %.5f)" % (post_m, post_var))
    print("--------------------------------------------------")

def GraghDisplay(ax, title, index=-1, LOG=None, sample=500):
    ax.set_title(title)
    ax.set_ylim([-25, 25])
    ax.set_xlim([-2, 2])
    if title == 'Ground truth':
        return
    
    if index >= len(LOG['mean']) or index < 0:
        print("Error: Invalid index for LOG data.")
        return
    
    x = np.linspace(-2, 2, sample)
    y_varp = np.zeros(sample)
    y_varn = np.zeros(sample)
    for i in range(sample):
        y_varp[i] = LOG['mean'][index][i] + LOG['var'][index][i]
        y_varn[i] = LOG['mean'][index][i] - LOG['var'][index][i]
    ax.plot(x, LOG['mean'][index], "k")
    ax.plot(x, y_varp, "r")
    ax.plot(x, y_varn, "r")

def EachPoint(X, Y, n, a, w, sample, m, _lambda):
    LOG_dyna = {'mean': [], 'var': []}
    LOG_dyna['mean'].append([])
    LOG_dyna['var'].append([])
    
    sample_x = np.linspace(-2, 2, sample)
    for i in range(sample):
        tPolM = PolyM(sample_x[i], n)
        LOG_dyna['mean'][0].append(float(mul(m.T, tPolM.T)))
        LOG_dyna['var'][0].append(float(1/a + mul(mul(tPolM, inv(_lambda)), tPolM.T)))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, Y, "b.")
    GraghDisplay(ax, "Experiment : "+str(len(X)), 0, LOG_dyna, sample)
    plt.show()

def visualization(X, Y, LOG, N, a, w, sample):
    fig = plt.figure(figsize=(10, 8))
    x = np.linspace(-2, 2, sample)
    y = np.zeros(sample)
    y_varp = np.zeros(sample)
    y_varn = np.zeros(sample)
    
    ax = fig.add_subplot(2, 2, 1)
    GraghDisplay(ax, "Ground truth")
    for i in range(sample):
        for n in range(N):
            y[i] += w[n] * (x[i]**n)
        y_varp[i] = y[i] + math.sqrt(a)
        y_varn[i] = y[i] - math.sqrt(a)
    ax.plot(x, y, "k")
    ax.plot(x, y_varp, "r")
    ax.plot(x, y_varn, "r")
    
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(X, Y, "b.")
    GraghDisplay(ax, "Predict result", 2, LOG, sample)
    
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(X[:10], Y[:10], "b.")
    GraghDisplay(ax, "After 10 incomes", 0, LOG, sample)
    
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(X[:50], Y[:50], "b.")
    GraghDisplay(ax, "After 50 incomes", 1, LOG, sample)
    
    plt.show()

def StoreData(LOG, sample, mean, var, n, a): # 對於一組均勻分佈的 x 值，計算每一點的後驗均值和方差。
    LOG['mean'].append([])
    LOG['var'].append([])
    sample_x = np.linspace(-2, 2, sample)
    for i in range(sample):
        tPolM = PolyM(sample_x[i], n)
        LOG['mean'][len(LOG['mean'])-1].append(float(mul(mean.T, tPolM.T)))
        LOG['var'][len(LOG['mean'])-1].append(float(1/a + mul(mul(tPolM, inv(var)), tPolM.T)))

def SequentialEstimator(m, s):
    print(f"Data point source function: N({m:.2f}, {s:.2f})")

    n, mean, square_mean = 0, 0.0, 0.0
    con_count, con_thres = 0, 5e-3

    while con_count < CONVERGE:
        d = Gaussian(m, s)
        n += 1
        square_mean = ((n - 1) * square_mean + d ** 2) / n # E[X^2] = (n-1)/n * E[X^2] + 1/n * X^2
        mean = ((n - 1) * mean + d) / n # E[X] = (n-1)/n * E[X] + 1/n * X
        var = square_mean - mean ** 2 # Var[X] = E[X^2] - E[X]^2

        print(f"Add data point: {d}, Mean = {mean}, Variance = {var}")
        # 收斂檢查
        if abs(mean - (mean - (d - mean) / n)) < con_thres: 
            con_count += 1
        else:
            con_count = 0


def BayesianLinearRegression(b, n, a, w):
    X ,Y= [], [] 
    m = np.zeros((n, 1))
    _lambda = np.eye(n) * b
    data_num = 0
    a_inv = 1.0 / a
    LOG = {'mean': [], 'var': []}
    sample = 500
    con_count = 0
    con_thres = 1e-3
    
    while con_count < CONVERGE:
        x, y = CreateData(n, a, w)
        X.append(x)
        Y.append(y)
        data_num += 1
        
        PolM = PolyM(x, n)
        
        # Update of prior distribution
        old_lambda = _lambda # prior precision matrix
        _lambda = old_lambda + a_inv * mul(PolM.T, PolM) # posterior precision matrix

        # posterior mean vector
        tmp = np.dot(old_lambda, m) # prior mean vector
        tmp1 = a_inv * np.dot(PolM.T, np.array([[y]])) # likelihood mean vector
        old_m = m 
        m = np.linalg.solve(_lambda, tmp + tmp1)   # 解矩陣方程得 posterior mean vector
        
        # Posterior mean, variance
        post_m = np.dot(m.T, PolM.T)
        S = np.linalg.inv(_lambda) 
        post_var = a_inv + np.dot(PolM, np.dot(S, PolM.T)) 

        if data_num % 1000 == 0:
            print("Number of data:", data_num)
            Output(x, y, m, inv(_lambda), post_m, post_var)
        
        # 收斂檢查
        if np.all(np.abs(old_m - m) < con_thres):
            con_count += 1
        else:
            con_count = 0        

        # Log data
        if data_num == 10 or data_num == 50:
            StoreData(LOG, sample, m, _lambda, n, a_inv)
            
        # EachPoint(X, Y, n, a, w, sample, m, _lambda)
        
    StoreData(LOG, sample, m, _lambda, n, a_inv)
    visualization(X, Y, LOG, n, a_inv, w, sample)



if __name__ == "__main__":
    CONVERGE = 20 
    m, s = 3, 5  # Mean and variance for Gaussian data generator
    b, n, a, w = 100, 4, 1, [1, 2, 3, 4]  # Parameters for Bayesian Linear Regression
    # SequentialEstimator(m, s)
    BayesianLinearRegression(b, n, a, w)
