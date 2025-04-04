import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

subject_num = 15
image_num = 11
train_num = 9
test_num = 2

height, width = 231, 195
expression = ['centerlight', 'glasses', 'happy', 'leftlight',
              'noglasses', 'normal', 'rightlight', 'sad',
              'sleepy', 'surprised', 'wink']

def readPGM(file):
    if not os.path.isfile(file):
        return None
    with open(file, 'rb') as f:
        f.readline() # P5
        f.readline() # Comment line
        width, height = [int(i) for i in f.readline().split()]
        assert int(f.readline()) <= 255 # Depth
        img = np.zeros((height, width))
        for r in range(height):
            for c in range(width):
                img[r][c] = ord(f.read(1))
        return img.reshape(-1)

def readData():
    train_data = []
    file_path = "./Yale_Face_Database/Yale_Face_Database/Training/"
    for sub in range(subject_num):
        for i in range(image_num):
            file_name = f"subject{sub+1:02d}.{expression[i]}.pgm"
            d = readPGM(file_path + file_name)
            if d is not None:
                train_data.append(d)
    
    test_data = []
    file_path = "./Yale_Face_Database/Yale_Face_Database/Testing/"
    for sub in range(subject_num):
        for i in range(image_num):
            file_name = f"subject{sub+1:02d}.{expression[i]}.pgm"
            d = readPGM(file_path + file_name)
            if d is not None:
                test_data.append(d)
    print("File read.")
    return np.array(train_data), np.array(test_data)



def PCA(data, k=25):
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    cov = (centered_data) @ (centered_data).T

    eigenvalue, eigenvector = np.linalg.eig(cov)
    eigenvector = data.T @ eigenvector
    
    # Normalize W
    for i in range(len(eigenvector[0])):
        eigenvector[:,i] = eigenvector[:,i] / np.linalg.norm(eigenvector[:,i])
        
    # 按eigenvalues降序排列
    idx = np.argsort(eigenvalue)[::-1]
    eigenvector = eigenvector[:, idx]
    W = eigenvector[:, :k].real
    
    return W, mean


def Reduction(data, S):
    num_images = len(data)
    reduced_height = height // S
    reduced_width = width // S

    d = np.zeros((num_images, reduced_height, reduced_width))

    for n in range(num_images):
        img = data[n].reshape(height, width)

        for i in range(0, height, S):
            for j in range(0, width, S):
                block = img[i:i+S, j:j+S]
                block_mean = np.mean(block)
                d[n, i // S, j // S] = block_mean

    return d.reshape(num_images, -1)


def LDA(data, labels, k=25):
    (n, d) = data.shape
    labels = np.asarray(labels)
    c = np.unique(labels)
    mean = np.mean(data, axis=0)
    Sw = np.zeros((d, d), dtype=np.float64)
    Sb = np.zeros((d, d), dtype=np.float64)
    # Compute within-class and between-class scatter matrices
    for i in c:
        X_i = data[np.where(labels == i)[0], :]
        mu_i = np.mean(X_i, axis=0)
        Sw += (X_i - mu_i).T @ (X_i - mu_i)
        Sb += X_i.shape[0] * (mu_i - mean).T @ (mu_i - mean)
    # Compute eigenvalues and eigenvectors
    eigen_val, eigen_vec = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    # Normalize eigenvectors
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / np.linalg.norm(eigen_vec[:, i])

    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigen_val)[::-1]
    W = eigen_vec[:, idx][:, :k].real

    return W, mean

def linearKernel(datai, dataj):
    return datai @ dataj.T

def polynomialKernel(datai, dataj, gamma=1e-2, c=0.1, d=2):
    return (gamma * (datai @ dataj.T) + c) ** d

def rbfKernel(datai, dataj, gamma=1e-8):
    K = np.zeros((len(datai), len(dataj)))
    for i in range(len(datai)):
        for j in range(len(dataj)):
            K[i][j] = np.exp(-gamma * np.sum((datai[i] - dataj[j]) ** 2))
    return K

def computeKernel(datai, dataj, _type):
    if _type == 'linear':
        return linearKernel(datai, dataj)
    if _type == 'polynomial':
        return polynomialKernel(datai, dataj)
    if _type == 'rbf':
        return rbfKernel(datai, dataj)

def centered(K):
    n = len(K)
    _1N = np.full((n, n), 1 / n)
    KC = K - _1N @ K - K @ _1N + _1N @ K @ _1N
    return KC

def kernelPCA(data, kernel_type, k=25):
    K = computeKernel(data, data, kernel_type)
    
    eigenvalues, eigenvectors = np.linalg.eig(K)

    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    
    idx = np.argsort(-eigenvalues)
    W = eigenvectors[:, idx][:, :k].real
    
    return W, K


def kernelLDA(data, kernel_type, k=25):
    K = computeKernel(data, data, kernel_type)
    
    Z = np.full((len(data), len(data)), 1 / train_num)
    
    Sw = K @ K.T
    Sb = K @ Z @ K.T
    
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    
    idx = np.argsort(-eigenvalues)
    W = eigenvectors[:, idx][:, :k].real
    
    return W, K


def EigenVisualFaces(W, file_path, k=25, S=1):
    # Save individual faces
    for i in range(k):
        img = W[:,i].reshape(height//S, width//S)
        plt.imsave(f'{file_path}eigenface_{i:02d}.jpg', img, cmap='gray')
    
    # Create and save composite figure
    fig, axes = plt.subplots(int(np.sqrt(k)), int(np.sqrt(k)), figsize=(12, 9))
    for i, ax in enumerate(axes.flat):
        if i < k:
            img = W[:,i].reshape(height//S, width//S)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
    plt.tight_layout()
    fig.savefig(f'{file_path}../eigenfaces_{k}.jpg')
    plt.show()


def ReconstructFace(W, mean, data, file_path, S=1):
    if mean is None:
        mean = np.zeros(W.shape[0])
    
    sel = np.random.choice(subject_num * train_num, 10, replace=False)
    img = []

    for index in sel:
        x = data[index].reshape(1, -1)
        reconstruct = (x - mean) @ W @ W.T + mean
        img.append(reconstruct.reshape(height//S, width//S))

        # Plot and save original and reconstructed face side-by-side
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(x.reshape(height//S, width//S), cmap='gray')
        ax[1].imshow(reconstruct.reshape(height//S, width//S), cmap='gray')
        fig.tight_layout()
        fig.savefig(f'{file_path}reconfaces_{len(img)}.jpg')
        plt.close(fig)

    # Plot all reconstructed faces together
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        if i < len(img):
            ax.imshow(img[i], cmap='gray')
    fig.tight_layout()
    fig.savefig(f'{file_path}../reconfaces.jpg')
    plt.show()


def distance(test, train_data):
    dist = np.zeros(len(train_data), dtype=np.float32)
    for j in range(len(train_data)):
        dist[j] = np.sum((test - train_data[j]) ** 2) # Euclidean distance
    return dist


def faceRecongnition(W, mean, train_data, test_data, K):
    if mean is None:
        mean = np.zeros(W.shape[0])
    
    # Project training and testing data into the lower-dimensional space
    low_train = (train_data - mean) @ W
    low_test = (test_data - mean) @ W
    
    err = 0
    for i in range(len(low_test)):
        dist = np.linalg.norm(low_train - low_test[i], axis=1)
        nearest = np.argsort(dist)[:K]
        vote = np.bincount(nearest // train_num, minlength=subject_num)
        predict = np.argmax(vote) + 1
        if predict != i // 2 + 1:
            err += 1
    
    accuracy = 1 - err / len(low_test)
    print(f"K={K}: Accuracy:{accuracy:.4f} ({len(low_test) - err}/{len(low_test)})")
    return accuracy


def centeredTest(K_test, K):
    n, l = len(K), len(K_test)
    _1N = np.full((n, n), 1 / n)
    _1NL = np.full((n, l), 1 / n)
    K_testC = K_test - K_test @ _1N - _1NL.T @ K + _1NL.T @ K @ _1N
    return K_testC
    
def kernelFaceRecongnition(W, train_data, test_data, kernel_type, kernel, K):
    # Project training and testing data into the lower-dimensional space
    low_train = kernel @ W
    K_test = computeKernel(test_data, train_data, kernel_type)    
    low_test = K_test @ W
    
    # KNN
    err = 0
    for i in range(len(low_test)):
        vote = np.zeros(subject_num, dtype=int)
        dist = distance(low_test[i], low_train)
        nearest = np.argsort(dist)[:K]
        for n in nearest:
            vote[n // train_num] += 1
        predict = np.argmax(vote) + 1
        if predict != i // 2 + 1:
            err += 1
    accuracy = 1 - err / len(low_test)
    print(f"K={K}: Accuracy:{accuracy:.4f} ({len(low_test) - err}/{len(low_test)})")
    return accuracy

if __name__ == "__main__":
    train_data, test_data = readData()
    # train_data, test_data, train_labels, test_labels = readData()
    dim = 25
    acc = 0
    task = 4
    
    # PCA
    if task == 1:
        PCA_file = './Result/PCA_LDA/PCA/'
        W_PCA, mean_PCA = PCA(train_data, k=dim)
        EigenVisualFaces(W_PCA, PCA_file + 'eigenfaces/', k=dim)
        ReconstructFace(W_PCA, mean_PCA, train_data, PCA_file + 'reconstruct/')
        for i in range(1, 20, 2):
            acc += faceRecongnition(W_PCA, mean_PCA, train_data, test_data, i)
            
    
    # LDA
    if task == 2:
        scalar = 3  
        labels = np.repeat(np.arange(1, subject_num + 1), train_num)
        LDA_file = './Result/PCA_LDA/LDA/'
        data = Reduction(train_data, scalar)
        compress_test = Reduction(test_data, scalar)

        W_LDA, mean_LDA = LDA(data, labels, k = dim)
        EigenVisualFaces(W_LDA, LDA_file + 'fisherfaces/', k = dim, S = scalar)
        ReconstructFace(W_LDA, None, data, LDA_file + 'reconstruct/', S = scalar)
        for i in trange(1, 20, 2):
            acc += faceRecongnition(W_LDA, None, data, compress_test, i)
    
    # Kernel PCA
    if task == 3:
        kernel_type = 'rbf'
        avgFace = np.mean(train_data, axis=0)
        centered_train = train_data - avgFace
        centered_test = test_data - avgFace

        W_kPCA, kernel = kernelPCA(centered_train, kernel_type, k=dim)
        for i in range(1, 20, 2):
            acc += kernelFaceRecongnition(W_kPCA, centered_train, centered_test, kernel_type, kernel, i)
    
    # Kernel LDA
    if task == 4:
        kernel_type = 'linear'
        avgFace = np.mean(train_data, axis=0)
        centered_train = train_data - avgFace
        centered_test = test_data - avgFace
        
        W_kLDA, kernel = kernelLDA(centered_train, kernel_type)
        for i in range(1, 20, 2):
            acc += kernelFaceRecongnition(W_kLDA, centered_train, centered_test, kernel_type, kernel, i)

    print(f"Average accuracy:{acc / 10: .4f}")
