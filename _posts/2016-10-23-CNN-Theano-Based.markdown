---
layout: post
title:  "CNN--Theano Based"
date:   2016-10-23
categories: Herb
comments: true
use_math: true
---

# 激活函数
- relu
- 
```python
def relu(x):
return T.maximum(0.0, x)
```

- tanh
- 
```python
def tanh(x):
    return T.tanh(x)
```

# 卷积层

```python
class ConvLayer:
    """
    convolutional layer.
    """

    def __init__(self, rng, input, filter_shape, image_shape, activation = "relu", w = None, b = None):
        """
        :param rng: A random number generator. Type: numpy.random.RandomState(*).
        :param input: A mini_batch input. Type: 4-tensor. shape = (mini_batch_size, feature_maps, img_high, img_width)
        :param filter_shape: Filter shape. Type: tuple or list size 4, (filter_nums, input_feature_map_num, filter_high, filter_width)
        :param image_shape: Image shape, the shape of input. Type: tuple or list size 4, (mini_batch_size, feature_maps, image_high, image_width).
        :param activation: Fingure which activation to use. Type: string.
        :param w: The filters, if the user given. Type: 4-tensor. shape=(fitler_nums, input_feature_map_num, filter_high, filter_width).
        :param b: The bias, if user given. Type: list. shape = (filter_num).
        """

        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.activation = activation

        if w == None:
            self.w = shared(np.asarray(rng.uniform(low = -0.01, high = 0.01, size = filter_shape), dtype = theano.config.floatX), borrow = True, name = "W_conv")
        else:
            self.w = shared(w, borrow = True, name = "w_conv")

        if b == None:
            self.b = shared(np.asarray(np.zeros(filter_shape[0]), dtype = theano.config.floatX), borrow = True, name = "b_conv")
        else:
            self.b = shared(b, borrow = True, name = "b_conv")

        conv_output = conv.conv2d(input = input, filters = self.w, image_shape = self.image_shape, filter_shape = self.shape) + self.b.dimshuffle('x', 0, 'x', 'x');

        if activation == None or activation == "relu":
            self.output = relu(conv_output)

        self.params = [self.w, self.b]
```

# 池化层

```python
class PoolLayer:
    """
    pooling layer
    """

    def __init__(self, input, poolsize):
        """
        :param input: The input. Type: 4-tensor. shape(mini_batch, input_feature_maps, image_high, image_width
        :param poolsize: The poolsize. Type: tuple or list size 2, (pool_high, pool_width)
        """
        self.output = pool.pool_2d(input, ds = poolsize)
```

# 隐藏层（全连接层）

```python
class HiddenLayer:
    """
    The full connection layer.
    """

    def __init__(self, rng, input, filter_shape, activation = "relu", w = None, b = None):
        """
        :param input: The input. Type: 4-tensor. shape = (mini_batch, dim_in).
        :param filter_shape: The ws' shape. Type: tuple or list size 2, (dim_in, dim_out).
        :param activation: Choose activation. Type: string.
        :param w: The w, if user given.
        :param b: The b, if user given.
        """
        self.input = input
        if w == None:
            self.w = shared(np.asarray(0.01 * rng.standard_random(size = filter_shape), dtype = theano.config.floatX), borrow = True, name = "w_hidden_layer")
        else:
            self.w = shared(w, borrow = True, name = "w_hidden_layer")

        if b == None:
            self.b = shared(np.asarray(np.zeros(filter_shape[1]), dtype = theano.config.floatX), borrow = True, name = "b_hidden_layer")
        else:
            self.b = shared(b, borrow = True, name = "b_hedden_layer")

        self.output = T.dot(self.input, self.w) + self.b

        if activation == "relu":
            self.output = relu(self.output)

        self.params = [self.w, self.b]
```

# 多层感知机

```python
class MLP:
    """
    The multiple layers perception.
    """
    def __init__(self, rng, input, layer_size):
        layers = []

        # ithe output of the input_layer
        layer_output = input

        # other layers
        for i in range(len(layer_size))[:-1]:
            dim_in = layer_size[i]
            dim_out = layer_size[i+1]
            layer = HiddenLayer(rng, layer_output, (dim_in, dim_out))
            layer_output = layer.output
            layers.append(layer)

        self.layers = layers
```

# Logression (use softmax)

```python
class LogressionLayer:
    """
    softmax output
    """
    def __init__(self, rng, input, y, filter_shape, w = None, b = None, regular = 0.01):
        if w == None:
            self.w = shared(np.asarray(rng.standard_random(filter_shape), dtype = theano.config.floatX), borrow = True, name = "m_logression_layer")
        else:
            self.w = shared(w, borrow = True, name = "m_logresssion_layer")

        if b == None:
            self.b = shared(np.asarray(np.zeros(filter_shape[1]), dtype = theano.config.floatX), borrow = True, name = "b_logression_layer")
        else:
            self.b = shared(b, borrow = True, name = "b_logression_layer")

        # predict
        self.p_given_x = T.nnet.softmax(T.dot(input, self.w) + self.b)
        self.y_given_x = T.argmax(self.p_given_x, axis = 1)

        # error
        self.acc = T.mean(T.neq(y, self.y_given_x))

        # cross entropy
        cross_entropy = -T.mean(T.log(self.p_given_x)[np.range(len(y)), y])
        self.cost = cross_entropy + regular * (w ** 2).sum()

        self.params = [self.w, self.b]
```

# 最后
根据所需的模型结构，采用以上各层进行组合。写一个参数更新函数对所有参数进行更新。定义 theano.function  作为模型的入口（train, test, predict, etc）。