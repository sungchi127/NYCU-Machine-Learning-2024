import numpy as np
import math
from tqdm import trange


train_image_path = "./train-images.idx3-ubyte_"
train_label_path = "./train-labels.idx1-ubyte_"
test_image_path = "./t10k-images.idx3-ubyte_"
test_label_path = "./t10k-labels.idx1-ubyte_"

def load(train_num, test_num, img_size):
    train_img = []
    train_label = []
    test_img = []
    test_label = []
    # Train image
    with open(train_image_path, 'rb') as file:
        file.read(16)
        for k in trange(train_num):
            tmp = np.zeros((img_size, img_size), np.uint8)
            for i in range(img_size):
                for j in range(img_size):
                    _bytes = file.read(1)
                    tmp[i][j] = int.from_bytes(_bytes, 'big')
            train_img.append(tmp)
    
    # Train label
    with open(train_label_path, 'rb') as file:
        file.read(8)
        for k in trange(train_num):
            _bytes = file.read(1)
            train_label.append(int.from_bytes(_bytes, 'big'))
            
    # Test image
    with open(test_image_path , 'rb') as file:
        file.read(16)
        for k in trange(test_num):
            tmp = np.zeros((img_size, img_size), np.uint8)
            for i in range(img_size):
                for j in range(img_size):
                    _bytes = file.read(1)
                    tmp[i][j] = int.from_bytes(_bytes, 'big')
            test_img.append(tmp)
        
    # Test label
    with open(test_label_path, 'rb') as file:
        file.read(8)
        for k in trange(test_num):
            _bytes = file.read(1)
            test_label.append(int.from_bytes(_bytes, 'big'))
        
    return np.array(train_img), np.array(train_label), np.array(test_img), np.array(test_label)


def discrete_img(px, img_size):
    for n in range(10):
        print(n, ":")
        for i in range(img_size):
            for j in range(img_size):
                pre = np.argmax(px[n][i*img_size + j])
                print("1", end=" ") if pre > 16 else print("0", end=" ")
            print()
        print()

def continuous_img(mean, img_size):
    for n in range(10):
        print(n, ":")
        for i in range(img_size):
            for j in range(img_size):
               print("1", end=" ") if mean[n][i * img_size + j] > 128 else print("0", end=" ")
            print()
        print()
        
def print_pre(post, pred, label):
    print("Posterior (in log scale):")
    for n in range(10):
        print(n, ": ", post[n], sep="")
    print("Prediction:", pred, ", Ans:", label)
    print()


PI = 3.1415926
train_num = 60000
test_num = 10000
img_size = 28
train_img, train_label, test_img, test_label = load(train_num, test_num, img_size)


#Discrete
# init
cnt = np.zeros(10, np.uint32)
pixel_counts = np.ones((10, img_size * img_size, 32), np.uint32)  # 避免除以零

train_imgs_reshaped = train_img.reshape(-1, img_size * img_size)

# 統計
for num in trange(10):
    imgs_of_class = train_imgs_reshaped[train_label == num]
    cnt[num] = len(imgs_of_class)
    for pix in range(img_size * img_size):
        pixel_values = imgs_of_class[:, pix] // 8  # 將像素值除以8以簡化索引
        for val in range(32):
            pixel_counts[num, pix, val] += np.sum(pixel_values == val)

# 先驗機率
prior_prob = cnt / np.sum(cnt)

#%%
# test
err = 0
test_imgs_flat = test_img.reshape(test_num, -1)  # 展平測試圖像

# 預先計算一些概率的對數值
log_prior_prob = np.log(prior_prob)
pixel_sum = np.sum(pixel_counts, axis=2, keepdims=True)
log_pixel_prob = np.log(pixel_counts) - np.log(pixel_sum)

for k in trange(test_num):
    img = test_imgs_flat[k] // 8  # 計算測試圖像的像素值
    img_one_hot = (np.arange(32) == img[:, None]).astype(int)  # 轉換為one-hot編碼
    # 計算後驗機率
    log_post = log_prior_prob + np.sum(log_pixel_prob * img_one_hot[None, :, :], axis=(1, 2))
    pred = np.argmax(log_post)
    # print_pre(log_post, pred, test_label[k])
    if pred != test_label[k]:
        err += 1

error_rate = err / test_num
print("Error rate:", error_rate)

print("Imagination of numbers in Bayesian classifier:\n")
discrete_img(pixel_counts, img_size)

#%%
#Continuous

# init
num_classes = 10
cnt2 = np.zeros(num_classes)
mean = np.zeros((num_classes, img_size * img_size))
var = np.zeros((num_classes, img_size * img_size))

# 將訓練圖像重塑以簡化索引
train_imgs_reshaped = train_img.reshape(-1, img_size * img_size)
test_imgs_reshaped = test_img.reshape(-1, img_size * img_size)

# 計算每個類別的均值和方差
for num in range(num_classes):
    imgs_of_class = train_imgs_reshaped[train_label == num]
    cnt2[num] = len(imgs_of_class)
    if cnt2[num] > 0:
        mean[num] = np.mean(imgs_of_class, axis=0)
        var[num] = np.var(imgs_of_class, axis=0)

epsilon = 1e-9
var += epsilon

# 計算先驗機率的log值
prior_log = np.log(cnt2 / np.sum(cnt2))

# test
err2 = 0
for k in trange(len(test_imgs_reshaped)):
    post_log = prior_log.copy()
    for n in range(num_classes):
        diff = test_imgs_reshaped[k] - mean[n]
        post_log[n] -= 0.5 * np.sum(np.log(2 * np.pi * var[n]) + (diff**2 / var[n]))
    
    pred = np.argmax(post_log)  # 找到最大的後驗機率
    # print_pre(post_log, pred, test_label[k])
    if pred != test_label[k]:
        err2 += 1


error_rate = err2 / len(test_imgs_reshaped)
print("\nContinuous mode--------")
print("Error rate:", error_rate)


continuous_img(mean, img_size)  
