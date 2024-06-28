# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:35:59 2022

@author: asus
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:05:22 2022

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
# -*- coding: utf-8 -*-
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


# 获取数据集的特征及标签数据
X_train,y_train = train["features"],train["labels"]
X_valid,y_valid = valid["features"],valid["labels"]
X_test,y_test = test["features"],test["labels"]

## 对图片进行相关颜色上的处理

import cv2
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    return img

# 对所有的数据进行处理
X_val = np.array(list(map(preprocess, X_valid)))
# 将所用数据变为可以进入网络训练数据

X_val = X_val.reshape(4410, 32, 32, 1)
class DatasetGenerator1:
    def __init__(self):
        self.data = X_val
        self.label = y_valid

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
dataset_generator1 = DatasetGenerator1()
dataset_te = ds.GeneratorDataset(dataset_generator1, ["data", "label"], shuffle=False)
def create_dataset( dataset,batch_size=17, repeat_size=1,
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
    mnist_ds = dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="data", num_parallel_workers=num_parallel_workers)
    # 处理生成的数据集
    buffer_size = 1000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)
    return mnist_ds
te =create_dataset(dataset_te)
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
        #x = self.fc3(x)
        #x = self.relu(x)
        x = self.fc4(x)
        return x

# 实例化网络
net = LeNet5()
from mindspore.nn import SoftmaxCrossEntropyWithLogits

lr = 0.02	#learingrate,学习率，可以使梯度下降的幅度变小，从而可以更好的训练参数
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
from mindspore.nn import Accuracy
from mindspore import Model

model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()} )

from mindspore import load_checkpoint, load_param_into_net
def test_net(network, model, mnist_path):
    """Define the evaluation method."""
    print("============== Starting Testing ==============")
    # for i in range(12):

    #param_dict = load_checkpoint("./models/mindspore_quick_start_4/checkpoint_lenet-%d_%s00.ckpt" %(i,j))
    # load the saved model for evaluation)
    # load testing dataset
    param_dict = load_checkpoint("./models/4/checkpoint_lenet_1-5_2900.ckpt" )
    load_param_into_net(network, param_dict)

    acc = model.eval(te, dataset_sink_mode=False)
    print('训练')
    print("============== Accuracy:{} ==============".format(acc))


#for i in range(1,6):
 #   for j in range(1, 30):
  #      print("./models/mindspore_quick_start_4/checkpoint_lenet-%d_%s00.ckpt" %(i,j))
  #      test_net(network, model, test, i, j)
test_net(network, model, test)