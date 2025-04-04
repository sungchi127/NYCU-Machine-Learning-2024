import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import trange

IMAGE = 2  # {1|2}
img_length = 100
img_size = 10000

K = 2  # {2|3|4}
gamma_s = 1 / img_size
gamma_c = 1 / (256 * 256)
kernel = None

METHOD = "random"  # {random|kmeans++|naive_sharding}

CUT = "ratio"  # {normalized|ratio}

COLOR = [[[153, 102, 51], [0, 153, 255]],
         [[0, 102, 204], [51, 204, 204], [153, 102, 51]],
         [[0, 102, 204], [51, 204, 204], [153, 102, 51], [153, 153, 153]]]

COLOR_MATPLOT = [['tab:orange', 'tab:blue'], 
                 ['tab:orange', 'tab:blue', 'tab:olive'],
                 ['tab:orange', 'tab:blue', 'tab:olive', 'tab:gray']]

def readImg(fileName):
    img = cv2.imread(fileName)
    return img.reshape(img_size, 3)

def compute_kernel(img):
    kernel = np.zeros((img_size, img_size))

    def spatial_similarity(p1, p2):
        return ((p1 // img_length - p2 // img_length) ** 2 +
                (p1 % img_length - p2 % img_length) ** 2)

    def color_similarity(c1, c2):
        return np.sum((np.array(c1, dtype=np.uint32) - np.array(c2, dtype=np.uint32)) ** 2)

    def kernel_function(x1, x2):
        spatial_sim = np.exp(-gamma_s * spatial_similarity(x1[0], x2[0]))
        color_sim = np.exp(-gamma_c * color_similarity(x1[1], x2[1]))
        return spatial_sim * color_sim

    for p in trange(img_size):
        for q in range(p, img_size):
            kernel[p][q] = kernel[q][p] = kernel_function([p, img[p]], [q, img[q]])
    
    return kernel

def min_distance(means, point):
    distances = np.sum((means - point)**2, axis=1)
    return np.min(distances)

def initKMeans(U):
    img_size, features = U.shape
    m = []
    cluster = np.full(img_size, -1, dtype=np.uint32)  
    
    if METHOD == "random":
        indices = np.random.choice(img_size, K, replace=False)
        m = U[indices].tolist()  # Store initial centers

    elif METHOD == "kmeans++":
        first_index = np.random.randint(img_size)
        m = [U[first_index]]
        for _ in range(1, K):
            distances = np.array([min_distance(m, U[n]) for n in range(img_size)])
            probabilities = distances / np.sum(distances)
            cumulative_probabilities = np.cumsum(probabilities)
            next_index = np.searchsorted(cumulative_probabilities, np.random.rand())
            m.append(U[next_index])

    elif METHOD == "naive_sharding":
        sorted_indices = np.argsort(U[:, 0])  
        slice_size = img_size // K
        for i in range(K):
            slice_indices = sorted_indices[i * slice_size:(i + 1) * slice_size]
            m.append(np.mean(U[slice_indices], axis=0))  # Compute mean of each slice

    return m, cluster


def distance(x, m):
    dist = 0
    for k in range(K):
        dist += (x[k] - m[k]) ** 2
    return dist

def E_Step(U, means):
    cluster = np.zeros(img_size, dtype=np.uint32)
    for n in range(img_size):
        min_k, min_dist = None, 1e8
        for k in range(K):
            d = distance(U[n], means[k])
            if d < min_dist:
                min_dist = d
                min_k = k
        if min_k is not None:
            cluster[n] = min_k
    return cluster
    
def M_Step(U, cluster):
    m = np.zeros((K, K), dtype=np.float64)
    for k in range(K):
        size = np.sum(cluster==k)
        for n in range(img_size):
            if cluster[n] == k:
                m[k] += U[n]
        m[k] /= size
    return m

def computeDelta(pre_cluster, cluster):
    delta = 0
    for n in range(img_size):
        if pre_cluster[n] != cluster[n]:
            delta += 1
    return delta

def visualize(img, cluster, _iter, result_file_path):
    im = np.zeros((img_length, img_length, 3), dtype=np.uint8)
    for n in range(img_size):
        im[n//img_length][n%img_length] = COLOR[K-2][cluster[n]]
    
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 2, 1)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    ax.imshow(im)
    plt.title(f"Iteration: {_iter}")
    
    ax = fig.add_subplot(1, 2, 2)
    img = cv2.cvtColor(img.reshape(img_length, img_length, 3), cv2.COLOR_RGB2BGR)
    ax.imshow(img)
    plt.show()
    
    fig.savefig(f'{result_file_path}/{_iter}.jpg')

def drawEigenspace(cluster, U, result_file_path):
    pt_x, pt_y, pt_z = [], [], []
    for k in range(K):
        pt_x.append([])
        pt_y.append([])
        pt_z.append([])
    fig = plt.figure()
    if K == 2:
        for n in range(img_size):
            pt_x[cluster[n]].append(U[n][0])
            pt_y[cluster[n]].append(U[n][1])
        for k in range(K):
            plt.scatter(pt_x[k], pt_y[k], c=COLOR_MATPLOT[K - 2][k], s=0.5)
    if K == 3:
        ax = fig.add_subplot(projection='3d')
        for n in range(img_size):
            pt_x[cluster[n]].append(U[n][0])
            pt_y[cluster[n]].append(U[n][1])
            pt_z[cluster[n]].append(U[n][2])
        for k in range(K):
            ax.scatter(pt_x[k], pt_y[k], pt_z[k], c=COLOR_MATPLOT[K - 2][k], s=0.5)
    plt.show()
    
    fig.savefig(f'{result_file_path}/eigen.jpg')

def kMeans(U, img):
    # Init means
    means, cluster = initKMeans(U)

    pre_cluster = None
    delta = img_size 
    _iter = 0   
    
    result_file_path = f'./img{IMAGE}_{K}_class_{METHOD}_{CUT}_cut'

    try:
        os.makedirs(result_file_path, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {result_file_path}: {e}")

    while delta > 0:
        _iter += 1
        pre_cluster = cluster.copy()
        
        # E Step: clustering
        cluster = E_Step(U, means)
        
        # M Step: update means
        means = M_Step(U, cluster)
        
        # Validate
        delta = computeDelta(pre_cluster, cluster)
        print(f"Iter:{_iter}, delta:{delta}")
        visualize(img, cluster, _iter, result_file_path)
        
    if K < 4:
        drawEigenspace(cluster, U, result_file_path)

def compute_laplacian_and_eigen(W, normalized=True):
    D = np.diag(np.sum(W, axis=1))
    if normalized:
        with np.errstate(divide='ignore', invalid='ignore'):
            D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
            D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
        L = np.identity(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt
    else:  # "ratio"
        L = D - W
    eigenvalues, eigenvectors = np.linalg.eigh(L)  
    return eigenvectors[:, 1:K+1]  

if __name__ == "__main__":
    img = readImg(f'image{IMAGE}.png')
    
    W = compute_kernel(img)

    U = compute_laplacian_and_eigen(W, normalized=(CUT == "normalized"))
    
    kMeans(U, img)


    