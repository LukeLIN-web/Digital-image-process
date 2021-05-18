import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


# read the first 5 pictures as the training data
def read_train(img_dir):
    images = []
    for i in range(5):
        path = img_dir + str(i + 1) + ".pgm"
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
    return images


# the train process
def mytrain(feature_num, model_file_name, images, if_show_feature=0, norm_method=0):
    image = np.array(images)  # train pictures together
    n = image.shape[0]  # number of pictures
    height = image.shape[1]  # height
    width = image.shape[2]  # width
    d = height * width
    # calculate mean face
    image_arr = []
    for i in range(n):
        image_arr.append(image[i].flatten())
    mean_arr = np.mean(np.array(image_arr).T, axis=1).astype(np.uint8)
    mean_image = (np.mean(np.array(image_arr).T, axis=1).astype(np.uint8)).reshape((height, width))
    cv2.imwrite("mean_face.png", mean_image)  # write mean face
    # calculate covariance matrix
    normalized = []
    for i in range(n):
        normalized.append(image_arr[i] - mean_arr)
    X = np.array(normalized).T
    if n > d:
        C = np.dot(X, X.T)
        [eigenvalues, eigenvectors] = np.linalg.eig(C)
    else:
        C = np.dot(X.T, X)
        [eigenvalues, eigenvectors] = np.linalg.eig(C)
        eigenvectors = np.dot(X, eigenvectors)
    idx = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors[:, 0:feature_num].copy()
    # get the eigenvectors
    if (norm_method == 0):
        for i in range(feature_num):
            eigenvectors[:, i] = eigenvectors[:, i] / (np.linalg.norm(eigenvectors[:, i]))
    else:
        for i in range(feature_num):
            eigenvectors[:, i] = (eigenvectors[:, i] - eigenvectors[:, i].min()) / (
                        eigenvectors[:, i].max() - eigenvectors[:, i].min())
    # get eigenface image
    if (if_show_feature == 1):
        img = np.zeros((height * 1, width * 10))
        for i in range(feature_num):
            tmp = np.asarray(eigenvectors[:, i]).reshape((height, width))
            tmp = 255 * tmp
            x = 0
            y = i % 10
            img[x * height:(x + 1) * height, y * width:(y + 1) * width] = tmp
        cv2.imwrite("eigenface.png", img)
    # save model
    np.save(model_file_name, eigenvectors)
    weight = []
    for i in range(n):
        weight.append(np.matmul(normalized[i], eigenvectors))
    return mean_arr, np.asarray(weight)


# reconstruct process
def myreconstruct(img, model_file_name, pc_num):
    h = img.shape[0]
    w = img.shape[1]
    eigenvectors = np.load(model_file_name)
    f = (np.matmul(eigenvectors, np.matmul(eigenvectors.T, (img.flatten()).T))).reshape((h, w))
    f = 255 * (f - f.min()) / (f.max() - f.min())
    cv2.imwrite("reconstruct" + str(pc_num) + ".jpg", f)


# identify function
def mytest(img, model_file_name, mean_arr, train_weight, if_show):
    # img: image to identify
    # model_file_name: train model file name
    # mean_arr: mean image
    # train_weight: train vector
    height = img.shape[0]  # height
    width = img.shape[1]  # width
    ori_img = img
    eigenvectors = np.load(model_file_name)
    # img - mean
    img = np.asarray(img).flatten() - mean_arr
    # 特征脸对人脸的表示
    weight = np.matmul(img.T, eigenvectors)
    min_dis = np.sum((train_weight[0] - weight) ** 2)
    min_idx = 0
    for i in range(train_weight.shape[0]):
        dist = np.sum((train_weight[i] - weight) ** 2)  # 求欧氏距离
        dist = dist ** 0.5
        if (dist < min_dis):  # find the closest face
            min_dis = dist
            min_idx = i
    # calculate the dir path
    if (if_show == 1):
        x = math.floor(min_idx / 5) + 1
        y = min_idx % 5 + 1
        print(x)
        print(y)
        min_img = cv2.imread(path + "s" + str(x) + "/" + str(y) + ".pgm")
        min_img = cv2.cvtColor(min_img, cv2.COLOR_BGR2GRAY)
        show_img = np.hstack([ori_img, min_img])
        # show the result
        cv2.imshow("show identify result", show_img)
        cv2.waitKey(100)
    return min_idx


path = "./data/"
images = []
for i in range(40):
    img_src = path + "s" + str(i + 1) + "/"
    images = images + read_train(img_src)
# mytrain and mytest
# train process
mean_arr, train_weight = mytrain(10, "model.npy", images, norm_method=1, if_show_feature=1)
# choose a test picture
img = cv2.imread(path + "s28/" + "6.pgm")
h = img.shape[0]
w = img.shape[1]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# identify process
print(mytest(img, "model.npy", mean_arr, train_weight, if_show=1))
# reconstruct
pc_list = [10, 25, 50, 100]
for i in range(len(pc_list)):
    mean_arr, train_weight = mytrain(pc_list[i], "model.npy", images, norm_method=1)
    img = cv2.imread("trans.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    myreconstruct(img, "model.npy", pc_list[i])
# show plot
x_ = []
y_ = []
for xx in range(20):
    pc_num = (xx + 1) * 10
    x_.append(pc_num)
    cnt = 0
    mean_arr, train_weight = mytrain(pc_num, "model.npy", images)
    for i in range(40):
        img_src = path + "s" + str(i + 1) + "/"
        for j in range(5):
            img_path = img_src + str(j + 6) + ".pgm"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rst = mytest(img, "model.npy", mean_arr, train_weight, 0)
            x = math.floor(rst / 5)
            if (x == i): cnt = cnt + 1
    y_.append(cnt / 200)
plt.plot(x_, y_)
plt.show()
