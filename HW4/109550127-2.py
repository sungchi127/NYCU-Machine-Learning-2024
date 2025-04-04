import numpy as np
from numpy import matmul as mul
from tqdm import trange
import random

random.seed(42)  # For reproducibility
train_num = 60000
test_num = 10000
img_size = 28


def loadData(train_num, img_size, batch_size=1000):
    TRAIN_IMAGE_FILE = "train-images.idx3-ubyte_"
    TRAIN_LABEL_FILE = "train-labels.idx1-ubyte_"

    train_img = []
    train_label = []

    with open(TRAIN_IMAGE_FILE, 'rb') as img_file, open(TRAIN_LABEL_FILE, 'rb') as label_file:
        img_file.read(16)  
        label_file.read(8)  
        
        for _ in trange(0, train_num, batch_size):
            batch_img_size = batch_size * img_size * img_size
            img_data = np.frombuffer(img_file.read(batch_img_size), dtype=np.uint8).reshape(-1, img_size, img_size)
            label_data = np.frombuffer(label_file.read(batch_size), dtype=np.uint8)
            train_img.extend(img_data)
            train_label.extend(label_data)

   
    train_img = np.array(train_img)
    train_label = np.array(train_label)

    return train_img, train_label


def dataPreprocess(img):
    tmp_img = (img > 127).astype(np.uint8)
    return tmp_img


def init_params(img_size):
    lam = np.full((10), 1/10)
    p = np.random.rand(10, img_size, img_size) / 2 + 0.25
    return lam, p

def e_step(train_num, img, lam, p):
    w = np.zeros((train_num, 10))
    # 計算該成分生成每個影像的機率、用bernoulli
    # lam * p^x * (1-p)^(1-x)
    for n in range(10):
        w[:, n] = lam[n] * np.prod(p[n]**img * (1-p[n])**(1-img), axis=(1, 2)) 
    w /= w.sum(axis=1, keepdims=True)
    return w

def m_step(w, img):
    lam = w.mean(axis=0)
    # 計算每個成分的新 p
    # w.T * img / w.sum
    p = mul(w.T, img.reshape(train_num, -1)).reshape(10, img_size, img_size) / w.sum(axis=0)[:, None, None] 
    p = np.clip(p, 1e-5, 1-1e-5)  # 避免機率值達到 0 或 1 造成 log 無法計算
    return lam, p

def assignLabel(train_label, train_num, w):
    mapping = np.zeros((10), dtype=np.uint32)
    counting = np.zeros((10, 10), dtype=np.uint32)
    for k in range(train_num):
        counting[train_label[k]][np.argmax(w[k])] += 1
    for n in range(10):
        index = np.argmax(counting) # return a 0~99 value
        label = index // 10 # return a 0~9 value
        _class = index % 10 # return a 0~9 value
        mapping[label] = _class # label->cluster
        counting[:,_class] = 0 
        counting[label,:] = 0 
    return mapping 

def printImagination(p, img_size, mapping, labeled=False):
    for n in range(10):
        if labeled:
            print("labeled", end=" ")
        print("class %d:" % n)
        real_label = mapping[n]
        for i in range(img_size):
            for j in range(img_size):
                print("*", end=" ") if p[real_label][i][j] > 0.5 else print(".", end=" ")
            print()
        print()
        
def printResult(train_label, train_num, mapping, w, _iter):
    err = train_num
    tb = np.zeros((10, 2, 2), dtype=np.uint32)
    
    mapping_inv = np.zeros((10), dtype=np.int32) # idx->value = cluster->label
    for i in range(10):
        mapping_inv[i] = np.where(mapping == i)[0]
        
    for k in range(train_num):
        pred = mapping_inv[np.argmax(w[k])]
        truth = train_label[k]
        for n in range(10):
            tb[n][0][0] = tb[n][0][0] + 1 if truth==n and pred==n else tb[n][0][0] #TP
            tb[n][0][1] = tb[n][0][1] + 1 if truth==n and pred!=n else tb[n][0][1] #FN
            tb[n][1][0] = tb[n][1][0] + 1 if truth!=n and pred==n else tb[n][1][0] #FP
            tb[n][1][1] = tb[n][1][1] + 1 if truth!=n and pred!=n else tb[n][1][1] #TN
        
    for n in range(10):
        print("--------------------------------------------------------")
        print(f"Confusion Matrix {n}:")
        print(f"\t\t\tPredict {n}\tPredict not {n}")
        print(f"Is {n}\t\t\t{tb[n][0][0]}\t\t{tb[n][0][1]}")
        print(f"Isn't {n}\t\t\t{tb[n][1][0]}\t\t{tb[n][1][1]}")
        sens = tb[n][0][0] / (tb[n][0][0] + tb[n][0][1])
        spec = tb[n][1][1] / (tb[n][1][0] + tb[n][1][1])
        print(f"\nSensitivity (Successfully predict number {n})\t: {sens}")
        print(f"Specificity (Successfully predict not number {n}): {spec}")
        err -= tb[n][0][0]
    
    print("--------------------------------------------------------")
    print(f"Total iteration to converge: {_iter}")
    print(f"Total error rate: {err/train_num}")

def main():
    train_img, train_label = loadData(60000, 28)
    print("\nData loaded.")
    
    img = dataPreprocess(train_img)
    lam, p = init_params(28)
    last_p = np.zeros_like(p)
    max_iter = 17
    conv_thres = 10
    mapping = np.array([i for i in range(10)], dtype=np.uint32)
    _iter = 0
    for _iter in trange(max_iter):
        w = e_step(60000, img, lam, p)
        lam, p = m_step(w, img)
        delta = np.abs(p - last_p).sum()
        printImagination(p, img_size, mapping)
        print(f"No. of Iteration: {_iter+1}, Difference: {delta}\n")
        print("--------------------------------------------------------")

        if delta < conv_thres:
            break
        last_p = p.copy()

    print("--------------------------------------------------------")
    mapping = assignLabel(train_label, train_num, w)
    printImagination(p, img_size, mapping, labeled=True)
    printResult(train_label, train_num, mapping, w, _iter+1)

if __name__ == "__main__":
    main()