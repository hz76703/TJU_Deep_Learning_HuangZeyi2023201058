# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 00:50:52 2022

@author: THINKPAD
"""
# 加载数据
# import pickle
try:
    import cPickle
except BaseException:
    import _pickle as cPickle

#step 1
training_file ="train.p"
validation_file ='valid.p'
testing_file = 'test.p'

train = cPickle.load(open(training_file, 'rb'))
valid = cPickle.load(open(validation_file, 'rb'))
test = cPickle.load(open(testing_file, 'rb'))

# training_file = '.spyder-py3/traffic-signs-data/train.p'
# validation_file = ".spyder-py3/traffic-signs-data/valid.p"
# testing_file = ".spyder-py3/traffic-signs-data/test.p"

# # 打开文件
# with open(training_file,mode="rb") as f:
#   train = pickle.load(f)
# with open(validation_file,mode="rb") as f:
#   valid = pickle.load(f)
# with open(testing_file,mode="rb") as f:
#   test = pickle.load(f)

# 获取数据集的特征及标签数据
X_train,y_train = train["features"],train["labels"]
X_valid,y_valid = valid["features"],valid["labels"]
X_test,y_test = test["features"],test["labels"]

# 查看数据量
print("Number of training examples =",X_train.shape[0])
print("Number of validtion examples =",X_valid.shape[0])
print("Number of testing examples=",X_test.shape[0])

# 查看数据格式
print("Image data shape =",X_train.shape[1:])

#训练集图片情况
print("训练集中图片维度大小：" , X_train.shape)
print("训练集中标签数量：" , y_train.shape)

# 查看数据的标签的数量
import numpy as np

sum = np.unique(y_train)
print("number of classes =",len(sum))

# 查看标签数据
import pandas as pd

sign_names_file = "signnames.csv"
sign_names = pd.read_csv(sign_names_file)
print(sign_names)

# 定义将标签id转换成name的函数
sign_names = np.array(sign_names)
def id_to_name(id):
  return sign_names[id][1]

# 验证是否id_to_name函数
print("id为0时对应的name: ",id_to_name(0))

# 绘制交通标志图
import matplotlib.pyplot as plt

fig,axes = plt.subplots(2,5,figsize=(18,5)) #随机绘制训练集中2*5=10张交通标志图
# 解决总标题中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.suptitle('训练集中随机10张图片及其含义')  #设置总标题
ax_array = axes.ravel()
for ax in ax_array:
  index = np.random.randint(0,len(X_train))
  ax.imshow(X_train[index])
  ax.axis("off")
  ax.set_title(id_to_name(y_train[index]))

plt.show()

# 用直方图展示图像训练集的各个类别的分布情况
n_classes = len(sum)
def plot_y_train_hist():
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,1,1)
    hist = ax.hist(y_train,bins = n_classes)
    ax.set_title("The frequency of each category sign")
    ax.set_xlabel("signs")
    ax.set_ylabel("frequency")
    plt.show()
    return hist

hist = plot_y_train_hist()

# 数据重采样，使样本个数分配均匀
bin_edges = hist[1]
bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2
for i in range(len(bin_centers)):
    if hist[0][i] < 1000:
        train_data = [X_train[j] for j in range(len(y_train)) if y_train[j] == i]
        need_resample_num = int(1000 - hist[0][i])
        new_data_x = [np.copy(train_data[np.random.randint(len(train_data))]) for k in range(need_resample_num)]
        new_data_y = [i for x in range(need_resample_num)]
        X_train = np.vstack((X_train, np.array(new_data_x)))
        y_train = np.hstack((y_train, np.array(new_data_y)))
print("数据重采样之后的图片维度大小：",X_train.shape)
print("数据重采样之后的标签数量：",y_train.shape)
plot_y_train_hist()
