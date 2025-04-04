import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def loadData():
    X, Y = [], []
    with open("input.data", 'r') as file:
        for line in file:
            x, y = map(float, line.split())
            X.append(x)
            Y.append(y)
    return np.array(X), np.array(Y)


def k(x1, x2, params):
    sigma, alpha, length = params
    d = (x1 - x2) ** 2
    return (1 + d / (2 * alpha * length ** 2)) ** -alpha * sigma ** 2

def Cov(X, beta, params):
    sigma, alpha, length = params
    size = len(X)
    C = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):  # 因為symmetry
            C[i][j] = C[j][i] = k(X[i], X[j], params)
        C[i][i] += 1 / beta
    return C

def predict(X, Y, C, pre_x, params):
    K = np.array([[k(Xi, x, params) for Xi in X] for x in pre_x]) # 
    C_inv = np.linalg.inv(C)
    mean = K.dot(C_inv.dot(Y))
    var = np.zeros(pre_x.shape)
    
    for i in range(len(pre_x)):
        k_ss = k(pre_x[i], pre_x[i], params) + 1 / beta
        var[i] = k_ss - K[i, :].dot(C_inv).dot(K[i, :].T)
    
    return mean, var


def visualize(title, data, pre_m, pre_v):
    x = np.linspace(-60, 60, len(pre_m))
    interval = 1.96 * np.sqrt(pre_v)
    plt.figure()
    plt.title(title)
    plt.plot(x, pre_m, 'r-')
    plt.fill_between(x, pre_m + interval, pre_m - interval, color='pink')
    plt.scatter(*data, color='black')  
    plt.xlim([-60, 60])
    plt.ylim([-5, 5])
    plt.show()


def negLogLikelihood(params, X, Y, beta):
    C = Cov(X, beta, params)
    C_inv = np.linalg.inv(C)
    Y = Y.reshape(-1, 1)  # 確保Y是二維列向量
    return 0.5 * (np.log(np.linalg.det(C)) + Y.T.dot(C_inv).dot(Y) + len(X) * np.log(2 * np.pi))

def GaussianProcess(X, Y, beta, params, plt_title=''):
    C = Cov(X, beta, params)
    pre_x = np.linspace(-60, 60, 1000)
    mean, var = predict(X, Y, C, pre_x, params)
    data = (X, Y)
    print(X)
    print(Y)
    visualize(plt_title, data, mean, var)

if __name__ == "__main__":
    X, Y = loadData()
    beta, initial_params = 5, [1, 1, 1] # sigma, alpha, length
    
    GaussianProcess(X, Y, beta, initial_params, "Original Params")
    print("Origin Finished")

    opt_params = minimize(negLogLikelihood, initial_params, args=(X, Y, beta), bounds=[(1e-6, 1e6)]*3).x
    
    GaussianProcess(X, Y, beta, opt_params, "Optimized Params: sigma={:.4f}, alpha={:.4f}, length={:.4f}".format(*opt_params))
    print("Optimized Finished")