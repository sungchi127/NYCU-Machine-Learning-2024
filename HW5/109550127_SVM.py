import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from libsvm.svmutil import *
from tqdm import tqdm

train_num = 5000
test_num = 2500
img_size = 28
img_length = img_size * img_size

kernel = {'linear':0, 'polynomial':1, 'RBF':2}
LOG = []

def loadData():
    def load_csv(filename, num_rows, num_cols):
        data = np.loadtxt(filename, delimiter=',', dtype=float)
        return data.reshape(num_rows, num_cols) if num_cols > 1 else data

    train_img = load_csv('X_train.csv', train_num, img_length)
    train_label = load_csv('Y_train.csv', train_num, 1)
    test_img = load_csv('X_test.csv', test_num, img_length)
    test_label = load_csv('Y_test.csv', test_num, 1)
    
    return train_img, train_label.flatten(), test_img, test_label.flatten()

def compare(_iter, X, Y, opt_option, opt_acc, cur_opt):
    acc = svm_train(Y, X, cur_opt)
    LOG.append([cur_opt, acc])
    if acc > opt_acc:
        return _iter+1, cur_opt, acc
    return _iter+1, opt_option, opt_acc


def gridSearch(X, Y):
    opt_option = '-t 0 -v 4'
    opt_acc = 0
    _iter = 0
    params = {
        'cost': [0.001, 0.01, 0.1, 1, 10],
        'gamma': [1e-4, 1/img_length, 0.1, 1],
        'degree': [2, 3, 4],
        'coef': [0, 1, 2]
    }
    for k, v in kernel.items():
        options = create_options(k, v, params)
        for opt in options:
            _iter, opt_option, opt_acc = compare(_iter, X, Y, opt_option, opt_acc, opt)
            print(f'Iter:{_iter}, Kernel:{k}, Acc:{opt_acc:.4f}, Opt:{opt_option}\n')

    return opt_option

def create_options(kernel_type, kernel_value, params):
    options = []
    if kernel_type == 'linear':
        for c in params['cost']:
            options.append(f'-t {kernel_value} -v 4 -c {c}')
    elif kernel_type == 'polynomial':
        for c in params['cost']:
            for g in params['gamma']:
                for d in params['degree']:
                    for coe in params['coef']:
                        options.append(f'-t {kernel_value} -v 4 -c {c} -g {g} -d {d} -r {coe}')
    elif kernel_type == 'RBF':
        for c in params['cost']:
            for g in params['gamma']:
                options.append(f'-t {kernel_value} -v 4 -c {c} -g {g}')
    return options


def linearKernel(X1, X2):
    return X1.dot(X2.T)

def RBFKernel(X1, X2, gamma):
    X1_sq = np.sum(X1**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2**2, axis=1)
    dist = X1_sq + X2_sq - 2 * X1.dot(X2.T)
    return np.exp(-gamma * dist)

def perform_part1(train_img, train_label, test_img, test_label):
    # -t 0: linear kernel , -t 1: polynomial kernel , -t 2: RBF kernel
    model = svm_train(train_label, train_img, '-t 2')
    result = svm_predict(test_label, test_img, model)

def perform_part2(train_img, train_label, test_img, test_label):
    option = gridSearch(train_img, train_label)
    option = option.replace(option[4:9], '')
    model = svm_train(train_label, train_img, option)
    result = svm_predict(test_label, test_img, model)

def perform_part3(train_img, train_label, test_img, test_label, img_length):
    gamma = 1 / img_length
    train_kernel = linearKernel(train_img, train_img) + RBFKernel(train_img, train_img, gamma)
    test_kernel = linearKernel(test_img, train_img) + RBFKernel(test_img, train_img, gamma)
    train_kernel = np.hstack((np.arange(1, len(train_label)+1).reshape(-1, 1), train_kernel))
    test_kernel = np.hstack((np.arange(1, len(test_label)+1).reshape(-1, 1), test_kernel))
    model = svm_train(train_label, train_kernel, '-t 4')
    result = svm_predict(test_label, test_kernel, model)

if __name__ == "__main__":
    train_img, train_label, test_img, test_label = loadData()
    
    _type = int(input('Run which part?(1, 2, 3):'))
    
    start = time.time()
    
    if _type == 1:
        perform_part1(train_img, train_label, test_img, test_label)
    elif _type == 2:
        perform_part2(train_img, train_label, test_img, test_label)
    elif _type == 3:
        perform_part3(train_img, train_label, test_img, test_label, img_length)
    
    print(f"Cost:{time.time() - start} s")
