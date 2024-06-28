# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 13:16:01 2022

@author: asus
"""

from mindspore import context

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
import numpy as np
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 加载数据
# import pickle
try:
    import cPickle
except BaseException:
    import _pickle as cPickle

# step 1
training_file = 'train.p'
validation_file = 'valid.p'
testing_file = 'test.p'

train = cPickle.load(open(training_file, 'rb'))
valid = cPickle.load(open(validation_file, 'rb'))
test = cPickle.load(open(testing_file, 'rb'))

# 获取数据集的特征及标签数据
X_train, y_train = train["features"], train["labels"]
X_valid, y_valid = valid["features"], valid["labels"]
X_test, y_test = test["features"], test["labels"]

# 查看数据量
print("Number of training examples =", X_train.shape[0])
print("Number of validtion examples =", X_valid.shape[0])
print("Number of testing examples=", X_test.shape[0])

# 查看数据格式
print("Image data shape =", X_train.shape[1:])

# 训练集图片情况
print("训练集中图片维度大小：", X_train.shape)
print("训练集中标签数量：", y_train.shape)

# X_test <class 'numpy.ndarray'>
# y_test <class 'numpy.ndarray'>


import matplotlib.pyplot as plt
import pandas as pd
import random
np.random.seed(0)

sign_names_file = "signnames.csv"
sign_names = pd.read_csv(sign_names_file)


## 对图片进行相关颜色上的处理


import cv2

plt.imshow(X_train[1000])
plt.axis("off")
plt.show()
print('未处理的图像shape', X_train[1000].shape)
print(y_train[1000])


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

img = grayscale(X_train[1000])
plt.imshow(img)
plt.axis("off")
plt.show()
print('BGR2GRAY的图像shape', img.shape)

def equalize(img):
    img = cv2.equalizeHist(img)
    return img


img = equalize(img)
plt.imshow(img)
plt.axis("off")
plt.show()
print('equalizeHist的图像shape', img.shape)

def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    return img

# 对所有的数据进行处理
X_train = np.array(list(map(preprocess, X_train)))
X_test = np.array(list(map(preprocess, X_test)))
X_val = np.array(list(map(preprocess, X_valid)))

plt.imshow(X_train[random.randint(0, len(X_train) - 1)])
plt.axis('off')
plt.show()
print(X_train.shape)


# 将所用数据变为可以进入网络训练数据

X_train = X_train.reshape(34799, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)
class DatasetGenerator:
    def __init__(self):
        self.data = X_train
        self.label = y_train

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
class DatasetGenerator1:
    def __init__(self):
        self.data = X_val
        self.label = y_valid

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
dataset_generator = DatasetGenerator()
dataset_train = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
dataset_generator1 = DatasetGenerator1()
dataset_te = ds.GeneratorDataset(dataset_generator1, ["data", "label"], shuffle=False)
def create_dataset( dataset,batch_size=12, repeat_size=1,
                   num_parallel_workers=1):
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.2
    shift_nml = -1 * 0.1307 / 0.2

    # 根据上面所定义的参数生成对应的数据增强方法，即实例化对象
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # 将数据增强处理方法映射到（使用）在对应数据集的相应部分（data，label）
    mn = dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mn = mn.map(operations=resize_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    mn = mn.map(operations=rescale_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    mn = mn.map(operations=rescale_nml_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    mn = mn.map(operations=hwc2chw_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    # 处理生成的数据集
    buffer_size = 1000
    mn = mn.shuffle(buffer_size=buffer_size)
    mn = mn.batch(batch_size, drop_remainder=False)
    mn = mn.repeat(repeat_size)
    return mn
import mindspore.nn as nn
from mindspore.common.initializer import Normal

class LeNet5(nn.Cell):
    """
    Lenet网络结构
    """ 
    def __init__(self, num_class=43, num_channel=1):
        super(LeNet5, self).__init__()
        # 定义所需要的运算
        self.conv1 = nn.Conv2d(num_channel, 60, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(60, 60, 5, pad_mode='valid')
        self.conv3 = nn.Conv2d(60, 30, 3, pad_mode='valid')
        self.conv4 = nn.Conv2d(30, 30, 3, pad_mode='valid')
        #self.fc1 = nn.Dense(30* 3 * 3, 400, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(9720, 500, weight_init=Normal(0.02))
        #self.fc3 = nn.Dense(500,100,weight_init=Normal(0.02))
        self.fc4 = nn.Dense(500, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout(keep_prob=0.5)
        self.flatten = nn.Flatten()

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        #x = self.fc1(x)
        #x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        #x = self.drop(x)
        #x = self.fc3(x)
        #x = self.relu(x)
        x = self.fc4(x)
        return x

# 实例化网络
net = LeNet5()
from mindspore.nn import SoftmaxCrossEntropyWithLogits

lr = 0.01	#learingrate,学习率，可以使梯度下降的幅度变小，从而可以更好的训练参数
momentum = 0.5
network = LeNet5()

#使用了流行的Momentum优化器进行优化
#vt+1=vt∗u+gradients
#pt+1=pt−(grad∗lr+vt+1∗u∗lr)
#pt+1=pt−lr∗vt+1
#其中grad、lr、p、v和u分别表示梯度、学习率、参数、力矩和动量。
net_opt = nn.Momentum(network.trainable_params(), lr, momentum)

#相当于softmax分类器
#sparse指定标签（label）是否使用稀疏模式，默认为false,reduction为损失的减少类型：mean表示平均值，一般
#情况下都是选择平均地减少
net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
from mindspore.train.callback import Callback

# custom callback function
class StepLossAccInfo(Callback):
    def __init__(self, model, eval_dataset, steps_loss, steps_eval):
        self.model = model	#计算图模型Model
        self.eval_dataset = eval_dataset	#测试数据集
        self.steps_loss = steps_loss	
        #收集step和loss值之间的关系，数据格式{"step": [], "loss_value": []}，会在后面定义
        self.steps_eval = steps_eval
        #收集step对应模型精度值accuracy的信息，数据格式为{"step": [], "acc": []}，会在后面定义

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        #cur_epoch_num是CallbackParam中的定义，获得当前处于第几个epoch,一个epoch意味着训练集
        #中每一个样本都训练了一次
        cur_epoch = cb_params.cur_epoch_num
        
        #同理，cur_step_num是CallbackParam中的定义，获得当前执行到多少step
        cur_step = (cur_epoch-1)*2900 + cb_params.cur_step_num
        self.steps_loss["loss_value"].append(str(cb_params.net_outputs))
        self.steps_loss["step"].append(str(cur_step))
        if cur_step % 100 == 0:
            #调用model.eval返回测试数据集下模型的损失值和度量值，dic对象
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.steps_eval["step"].append(cur_step)
            self.steps_eval["acc"].append(acc["Accuracy"])
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.nn import Accuracy
from mindspore import Model

epoch_size = 5  #每个epoch需要遍历完成图片的batch数,这里是只要遍历一次
tr =create_dataset(dataset_train)
te =create_dataset(dataset_te)
print('Number of groups in the dataset:', tr.get_dataset_size())

model_path = "./models/mindspore_quick_start_4/"
#调用Model高级API，将LeNet-5网络与损失函数和优化器连接到一起，具有训练和推理功能的对象。
#metrics 参数是指训练和测试期，模型要评估的一组度量，这里设置的是"Accuracy"准确度
model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()} )

#保存训练好的模型参数的路径
config_ck = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=400)
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=model_path,config=config_ck)

#回调类中提到的我们要声明的数据格式
steps_loss = {"step": [], "loss_value": []}
steps_eval = {"step": [], "acc": []}
#使用model等对象实例化StepLossAccInfo，得到具体的对象
step_loss_acc_info = StepLossAccInfo(model , tr, steps_loss, steps_eval)

#调用Model类的train方法进行训练，LossMonitor(100)每隔100个step打印训练过程中的loss值,dataset_sink_mode为设置数据下沉模式，但该模式不支持CPU，所以这里我们只能设置为False
model.train(epoch_size, tr, callbacks=[ckpoint_cb, LossMonitor(100), step_loss_acc_info], dataset_sink_mode=False)
from mindspore import load_checkpoint, load_param_into_net
def test_net(network, model):
    """Define the evaluation method."""
    print("============== Starting Testing ==============")
    for i in range(1,6):
        
        param_dict = load_checkpoint("./models/mindspore_quick_start_4/checkpoint_lenet-%d_2900.ckpt"%i)
    # load the saved model for evaluation)
    # load testing dataset
        load_param_into_net(network, param_dict)

        acc = model.eval(te, dataset_sink_mode=False)
        print('%d次训练'%i)
        print("============== Accuracy:{} ==============".format(acc))
test_net(network,model)