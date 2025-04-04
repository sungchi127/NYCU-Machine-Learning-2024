import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import trange
from scipy.spatial.distance import cdist

IMAGE = 1  # {1|2}
img_length = 100
img_size = 10000

K = 2  # {2|3|4}
gamma_s = 1 / img_size
gamma_c = 1 / (256 * 256)
kernel = None

METHOD = "random"  # {random|kmeans++|naive_sharding}

COLOR = [[[153, 102, 51], [0, 153, 255]],
         [[0, 102, 204], [51, 204, 204], [153, 102, 51]],
         [[0, 102, 204], [51, 204, 204], [153, 102, 51], [153, 153, 153]]]

def readImg(fileName):
    img = cv2.imread(fileName)
    return img.reshape(img_size, 3)



def compute_distance(pos, m, img_length):
    x, y = pos % img_length, pos // img_length
    min_distance = float('inf')
    for center in m:
        center_x, center_y = center[0] % img_length, center[0] // img_length
        distance = (x - center_x) ** 2 + (y - center_y) ** 2
        if distance < min_distance:
            min_distance = distance
    return min_distance

def initialize_centers(img, img_size, K, METHOD, img_length):
    centers = []
    if METHOD == "random":
        indices = np.random.choice(img_size, K, replace=False)
        centers = [[index, img[index]] for index in indices]

    elif METHOD == "kmeans++":
        first_center = np.random.randint(img_size)
        centers = [[first_center, img[first_center]]]
        for _ in range(1, K):
            distances = np.array([compute_distance(n, centers, img_length) for n in range(img_size)])
            probabilities = distances / distances.sum()
            cumulative_probabilities = np.cumsum(probabilities)
            random_value = np.random.rand()
            next_center = np.searchsorted(cumulative_probabilities, random_value)
            centers.append([next_center, img[next_center]])

    elif METHOD == "naive_sharding":
        attributes = np.array([[n] + list(img[n]) for n in range(img_size)])
        attributes_sorted = attributes[np.argsort(attributes[:, 1])]
        slice_size = img_size // K
        for i in range(K):
            slice_indices = attributes_sorted[i * slice_size:(i + 1) * slice_size, 0]
            mean_index = int(np.mean(slice_indices))
            mean_color = np.mean(attributes_sorted[i * slice_size:(i + 1) * slice_size, 2:], axis=0).astype(np.uint8)
            centers.append([mean_index, mean_color])

    return centers

def init(img, K, METHOD='random'):
    img_length = int(np.sqrt(len(img)))  
    img_size = len(img)
    centers = initialize_centers(img, img_size, K, METHOD, img_length)
    alpha = np.zeros((img_size, K), dtype=np.uint8)
    for i, center in enumerate(centers):
        alpha[center[0], i] = 1
    c = updateCk(alpha) 
    return centers, c, alpha

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

def distance(c, alpha):
    dist = np.ones((K, img_size))
    
    for k in range(K):
        a = alpha[:, k].reshape(-1, 1).T
        second_term = a @ kernel
        second_term = second_term * 2 / c[k]
        dist[k] -= second_term.flatten()
        
        indicator = alpha[:, k].reshape(-1,1)
        third_term = np.sum(indicator.T @ kernel @ indicator)
        third_term /= (c[k] ** 2)
        dist[k] += third_term
    
    return dist    

def distance(c, alpha):
    dist = np.ones((K, img_size))
    
    for k in range(K):
        cluster_mask = alpha[:, k].reshape(-1, 1)

        second_term = (2 / c[k]) * (cluster_mask.T @ kernel).flatten()

        third_term = (1 / c[k] ** 2) * np.sum(cluster_mask.T @ kernel @ cluster_mask)
        
        dist[k] -= second_term
        dist[k] += third_term
    
    return dist

def E_Step(img, c, alpha):
    a = np.zeros((img_size, K), dtype=np.uint8)
    
    # Distance between all data points
    dist = distance(c, alpha)
    
    # Clustering
    for n in range(img_size):
        min_dist = 1e8
        min_k = None
        for k in range(K):
            if dist[k][n] < min_dist:
                min_dist = dist[k][n]
                min_k = k
        if min_k is not None:
            a[n][min_k] = 1
    return a

def updateCk(alpha):
    c = np.zeros(K)
    for k in range(K):
        c[k] = np.sum(alpha[:, k]==1)
    return c

def computeChange(pre_alpha, alpha):
    delta = 0
    for n in range(img_size):
        if np.argmax(pre_alpha[n]) != np.argmax(alpha[n]):
            delta += 1
    return delta

def visualize(img, alpha, _iter, result_file_path):    
    im = np.zeros((img_length, img_length, 3), dtype=np.uint8)

    for n in range(img_size):
        cluster = np.where(alpha[n] == 1)[0][0]
        im[n // img_length][n % img_length] = COLOR[K - 2][cluster]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im_converted = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    ax[0].imshow(im_converted)
    ax[0].set_title(f"Iteration: {_iter}")
    
    img_reshaped = cv2.cvtColor(img.reshape(img_length, img_length, 3), cv2.COLOR_RGB2BGR)
    ax[1].imshow(img_reshaped)

    plt.show()
    fig.savefig(f'{result_file_path}/{_iter}.jpg')

if __name__ == "__main__":
    img = readImg(f'image{IMAGE}.png')
    pre_alpha = None
    delta = img_size
    _iter = 0
    
    means, c, alpha = init(img, K, METHOD)

    kernel = compute_kernel(img)


    result_file_path = f'./img{IMAGE}_{K}_class_{METHOD}'
    try:
        os.makedirs(result_file_path, exist_ok=True)
    except:
        pass
    
    while delta > 0:#Convergence Check
        _iter += 1
        
        pre_alpha = alpha.copy()
        #In the E-step, each data point is reassigned to the cluster that minimizes the distance D(k,x)
        alpha = E_Step(img, c, alpha)
        c = updateCk(alpha)
        
        delta = computeChange(pre_alpha, alpha)
        print(f"Iter:{_iter}, delta:{delta}")
        

        visualize(img, alpha, _iter, result_file_path) 


