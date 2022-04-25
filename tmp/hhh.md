## **全连接层和激活函数的反向实现**

1.实现全连接层的反向传播

```python
import numpy as np

class FullyConnected:
    def __init__(self, W, b):
        r'''
        全连接层的初始化。

        Parameter:
        - W: numpy.array, (D_in, D_out)
        - b: numpy.array, (D_out)
        '''
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None

        self.dW = None
        self.db = None

    def forward(self, x):
        r'''
        全连接层的前向传播。

        Parameter:
        - x: numpy.array, (B, d1, d2, ..., dk)

        Return:
        - y: numpy.array, (B, M)
        '''
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        r'''
        全连接层的反向传播

        Parameter:
        - dout: numpy.array, (B, M)

        Return:
        - dx: numpy.array, (B, d1, d2, ..., dk) 与self.original_x_shape形状相同

        另外，还需计算以下结果：
        - self.dW: numpy.array, (N, M) 与self.W形状相同
        - self.db: numpy.array, (M,)
        '''
        ########## Begin ##########
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx
        ########## End ##########
```

2.实现常用激活函数的反向传播

~~~python
```python
import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        r'''
        Sigmoid激活函数的前向传播。

        Parameter:
        - x: numpy.array, (B, d1, d2, ..., dk)

        Return:
        - y: numpy.array, (B, d1, d2, ..., dk)
        '''
        out = 1. / (1. + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        r'''
        sigmoid的反向传播

        Parameter:
        - dout: numpy.array, (B, d1, d2, ..., dk)

        Return:
        - dx: numpy.array, (B, d1, d2, ..., dk)
        '''
        ########## Begin ##########
        dx = dout * (1.0 - self.out) * self.out
        return dx
        ########## End ##########


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        r'''
        ReLU激活函数的前向传播。

        Parameter:
        - x: numpy.array, (B, d1, d2, ..., dk)

        Return:
        - y: numpy.array, (B, d1, d2, ..., dk)
        '''
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        r'''
        relu的反向传播

        Parameter:
        - dout: numpy.array, (B, d1, d2, ..., dk)

        Return:
        - dx: numpy.array, (B, d1, d2, ..., dk)
        '''
        ########## Begin ##########
        dout[self.mask] = 0
        dx = dout
        return dx
        ########## End ##########

~~~









## tensorflow入门

1.tensorflow基本运算

```python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

def simple_func(a,b,c,d):
    '''
    返回值:
    result: 一个标量值
    '''
    # 请在此添加代码 完成本关任务
    # ********** Begin *********#
    tensor_a = tf.constant(a)
    tensor_b = tf.constant(b)
    tensor_c = tf.constant(c)
    tensor_d = tf.constant(d)
    sum_1 = tf.add(tensor_a, tensor_b)
    sum_2 = tf.add(tensor_c, tensor_d)
    mul = tf.multiply(sum_1, sum_2)
    with tf.Session() as sess:
        result = sess.run(mul)
    # ********** End **********#

    # 返回result

    return result

a = int(input())
b = int(input())
c = int(input())
d = int(input())

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(simple_func(a,b,c,d))
```







## tensorflow入门——构建神经网络

1.神经元与激活函数

```python
# -*- coding: utf-8 -*-
import tensorflow as tf

# 模拟一个 M-P 神经元的工作原理
# input_value 是输入值， 类型为一维的tf.constant
# weight 是这个神经元的权重， 类型为一维的tf.constant
# threshold 是这个神经元的阈值， 类型为零维的tf.constant
# 返回值是一个浮点数
def neuron(input_value, weight, threshold):
    # 请在此添加代码 完成本关任务
    # ********** Begin *********#

    # input_value = tf.constant([1., 0.])
    # weight =  tf.constant([1., 1.])  
    # threshold = tf.constant(2.)
    # 神经元接收来自 n 个其他神经元传递过来的输入信号通过权值进行连接 
    y = tf.multiply(input_value, weight)
    y = tf.reduce_sum(y)
    # 将连接后的结果与神经元的阈值进行比较
    y = tf.subtract(y, threshold)
    y = tf.sigmoid(y)
    return y.eval()
    # ********** End **********#
    
```

2.神经元与激活函数——tanh方法

```python
# -*- coding: utf-8 -*-
import tensorflow as tf

# 模拟一个 M-P 神经元
class neuron(object):

    # 构造函数
    # weight为本神经元的权重，类型为一维的tf.constant
    # threshold 是这个神经元的阈值， 类型为零维的tf.constant
    def __init__(self, weight, threshold):
    # 请在此添加代码 完成本关任务
    # ********** Begin *********#
        self.weight = weight
        self.threshold = threshold
    # ********** End **********#

    # 计算函数
    # input_value 是输入值， 类型为一维的tf.constant
    # 返回值是一个浮点数
    def computes(self, input_value):
        # 请在此添加代码 完成本关任务
        # ********** Begin *********#
        # input_value = tf.constant([1., 0.])
        # weight =  tf.constant([1., 1.])  
        # threshold = tf.constant(2.)
        # 神经元接收来自 n 个其他神经元传递过来的输入信号通过权值进行连接 
        y = tf.multiply(input_value, self.weight)
        y = tf.reduce_sum(y)
        # 将连接后的结果与神经元的阈值进行比较
        y = tf.subtract(y, self.threshold)
        y = tf.tanh(y)

        return y.eval()
    # ********** End **********#
    
```

3.构建简单的单隐层前馈神经网络

```python
# -*- coding: utf-8 -*-
import tensorflow as tf

# 模拟一个 M-P 神经元
class neuron(object):

    # 构造函数
    # weight为本神经元的权重，类型为一维的tf.constant
    # threshold 是这个神经元的阈值， 类型为零维的tf.constant
    def __init__(self, weight, threshold):
        # 请在此添加代码 完成本关任务
        # ********** Begin *********#
        self.weight = weight
        self.threshold = threshold

    # ********** End **********#

    # 计算函数
    # input_value 是输入值， 类型为一维的tf.constant
    # 返回值是一个浮点数
    def computes(self, input_value):
        # 请在此添加代码 完成本关任务
        # ********** Begin *********#
        y = tf.multiply(input_value, self.weight)
        y = tf.reduce_sum(y)
        y = tf.subtract(y, self.threshold)
        y = tf.nn.relu(y)
        return y.eval()
    # ********** End **********#


# 模拟神经网络中的一层
class Dense(object):

    # 构造函数
    # weights 为本层中每个神经元的权重，元素类型为一维的tf.constant，weights的类型是python的列表
    # thresholds 为本层中每个神经元的权重，元素类型为零维的tf.constant，thresholds的类型是python的列表
    def __init__(self, weights, thresholds):
        # 请在此添加代码 完成本关任务
        # ********** Begin *********#
        weights_len = len(weights)
        # weights_len = tf.size(weights).eval()
        # print(weights_len)
        # print(weights.shape)
        # weights_len = weights.shape[0]
        # print(weights_len)
        self.neurons = []
        for i in range(weights_len):
            self.neurons.append(neuron(weights[i], thresholds[i]))
            # ********** End **********#

    # 计算函数
    # input_value 是输入值， 类型为一维的tf.constant
    # 返回值应为一个 1 维， 长度为n的Tensor， n是本层中神经元的数量
    def computes(self, input_value):
        # 请在此添加代码 完成本关任务
        # ********** Begin *********#
        L = []
        neurons_len = len(self.neurons)
        # neurons_len = tf.size(self.neurons)
        # neurons_len = self.neurons.shape[0]
        for i in range(neurons_len):
            L.append(self.neurons[i].computes(input_value))
        return tf.constant(L)
        # ********** End **********#


# 模拟一个简单的神经网络
# input_value是这个神经网络的输入，类型为一维的tf.constant
# wegihtsOfMiddle 是这个神经网络中间层每个神经元的权重， 元素类型为一维的tf.constant，wegihtsOfMiddle的类型是python的列表
# thresholdsOfMiddle 是这个神经网络中间层每个神经元的阈值， 元素类型为零维的tf.constant，thresholdsOfMiddle的类型是python的列表
# wegihtsOfOut 是这个神经网络输出层每个神经元的权重， 元素类型为一维的tf.constant，wegihtsOfOut 的类型是python的列表
# thresholdsOfOut 是这个神经网络输出层每个神经元的阈值， 元素类型为零维的tf.constant，thresholdsOfOut 的类型是python的列表
# 返回值是一个一维浮点数组 （注意不是Tensor），数组的长度为输出层神经元的数量
def NetWork(input_value, wegihtsOfMiddle, thresholdsOfMiddle, weightsOfOut, thresholdsOfOut):
    # 请在此添加代码 完成本关任务
    # ********** Begin *********#
    middle = Dense(wegihtsOfMiddle, thresholdsOfMiddle)
    out = Dense(weightsOfOut, thresholdsOfOut)
    return out.computes(middle.computes(input_value)).eval()
    # ********** End **********#
```