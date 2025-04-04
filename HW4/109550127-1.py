import numpy as np
from numpy.linalg import inv, det
from numpy import matmul as mul
import matplotlib.pyplot as plt

#Marsaglia polar method
def gaussian(mean, var):
    s, u, v = 0, 0, 0 
    while True:
        u, v = np.random.uniform(-1, 1, 2)
        s = u**2 + v**2
        if s < 1:
            break
    t_s = np.sqrt(-2 * np.log(s) / s) # t_s = sqrt(-2ln(s) / s)
    return mean + t_s * u * np.sqrt(var) # mean + t_s * u * sqrt(var)
 
def create_design_matrix(x1, x2):
    """ Create design matrix with a bias term. """
    return np.hstack((np.ones((len(x1) + len(x2), 1)), np.vstack((x1, x2))))

def logistic_regression(dmat, y, lr, threshold, max_iter, converge=5, method='gradient_descent'):
    w = np.zeros((dmat.shape[1], 1))
    # last_w = np.zeros((dmat.shape[1], 1))
    conv_count = 0
    _iter = 0
    
    while conv_count < converge and _iter < max_iter:
        _iter += 1
        
        # Compute logistic regression predictions and gradient
        z = dmat.dot(w) # dmat * w
        prob = 1 / (1 + np.exp(-z)) # 1 / (1 + e^(-z))
        grad = dmat.T.dot(prob - y) # dmat.T * (prob - y)

        if method == 'newton':
            # Calculate Hessian matrix for Newton's method
            s = prob * (1 - prob) 
            hessian = dmat.T.dot(np.diagflat(s)).dot(dmat) # dmat.T * np.diag(s) * dmat
           
            if det(hessian) != 0:
                grad = inv(hessian).dot(grad)
            else:
                print("Hessian isn't invertible, switching to gradient descent")
                method = 'gradient_descent'

        w_new = w - lr * grad # update w
        # 判斷收斂
        if np.linalg.norm(w_new - w, 1) < threshold:
            conv_count += 1
        else:
            conv_count = 0
        
        w = w_new

    # print("Converge iteration:", _iter)
    return w

def predict(w, dmat):
    predictions = 1 / (1 + np.exp(-dmat.dot(w))) >= 0.5 # 1 / (1 + e^(-dmat * w)) >= 0.5
    return predictions.astype(int)

def confusionMat(y, pre, N):
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(2 * N):
        TP = TP + 1 if y[i]==0 and y[i]==pre[i] else TP
        FN = FN + 1 if y[i]==0 and y[i]!=pre[i] else FN
        FP = FP + 1 if y[i]==1 and y[i]!=pre[i] else FP
        TN = TN + 1 if y[i]==1 and y[i]==pre[i] else TN
    return [TP, FN, FP, TN]

def output(w, table, _type=""):
    print(_type)
    print("\nConfusion Matrix:")
    print("\t\t\tPredict cluster 1\tPredict cluster 2")
    print("In cluster 1\t\t%d\t\t\t\t\t%d" % (table[0], table[1]))
    print("In cluster 2\t\t%d\t\t\t\t\t%d" % (table[2], table[3]))
    print("\nSensitivity (Successfully predict cluster 1):", table[0] / (table[0] + table[1]))
    print("Specificity (Successfully predict cluster 2):", table[3] / (table[2] + table[3]))
    

def visualize(N, D1, D2, pre_gd, pre_nt):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Ground truth', 'Gradient descent', 'Newton\'s method']
    predictions = [None, pre_gd, pre_nt]  

    for ax, title, prediction in zip(axes, titles, predictions):
        ax.set_title(title)
        # Plot D1 points
        for i in range(N):
            color = 'r' if (prediction is None or prediction[i] == 0) else 'b'
            ax.plot(D1[i][0], D1[i][1], color + '.')
        # Plot D2 points
        for j in range(N):
            color = 'b' if (prediction is None or prediction[N + j] == 1) else 'r'
            ax.plot(D2[j][0], D2[j][1], color + '.')

    plt.tight_layout()
    plt.show()



def main():
    lr = 1e-2
    N = 50
    mx1, my1, mx2, my2 = 1, 1, 10, 10
    vx1, vy1, vx2, vy2 = 2, 2, 2, 2

    # Generate data
    D1 = np.array([[gaussian(mx1, vx1), gaussian(my1, vy1)] for _ in range(N)])
    D2 = np.array([[gaussian(mx2, vx2), gaussian(my2, vy2)] for _ in range(N)])
    dMat = create_design_matrix(D1, D2)
    y = np.vstack((np.zeros((N, 1)), np.ones((N, 1))))

    # Run logistic regression
    converge_threshold = 5  
    max_iterations = 500
    threshold = 1e-3
    w_gd = logistic_regression(dMat, y, lr, threshold, max_iterations, converge_threshold, 'gradient_descent')
    w_nt = logistic_regression(dMat, y, lr, threshold, max_iterations, converge_threshold, 'newton')

    # Predict and evaluate
    pred_gd = predict(w_gd, dMat)
    pred_nt = predict(w_nt, dMat)

    print("Gradient Descent Weights:\n", w_gd)
    confusionMat_gd = confusionMat(y, pred_gd, N)
    output(w_gd, confusionMat_gd, "\nGradient Descent':")

    print("Newton's Method Weights:\n", w_nt)
    confusionMat_nt = confusionMat(y, pred_nt, N)
    output(w_nt, confusionMat_nt, "\nNewton's method':")

    visualize(N, D1, D2, pred_gd, pred_nt)

if __name__ == "__main__":
    main()
