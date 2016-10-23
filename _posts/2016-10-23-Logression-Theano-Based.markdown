---
layout: post
title:  "Logistic Regression--Theano Based"
date:   2016-10-23
categories: Herb
comments: true
use_math: true
---

# 关于theano的一些认识
- theano四个概念：
	- 符号变量
		符号变量是theano中定义表达式的基本组成，其特点是不保持实际的数值，作为占位符而存在。
        声明： theano.scalar, theano.matrix, ...
        
    - op
    	op 是定义在符号变量上的一组运算。
        例子： x, y 是符号变量， z = x + y 是一个op
    	注意： theano将符号表达式解析为graph，该图的节点包括：符号变量, op, apply.
    - shared 变量
    	共享变量是指可以在不同的function间共享的变量，其可以保存值，需要在声明时指定需要共享的变量的具体的数值。通常用于定义模型参数，以便与保存和更新。
        声明：w = theano.shared(value=, name="").  value是需要共享的变量， name是为其指定的名字（可以不指定）。
        取值：w.get_value()
        赋值：w.set_value(new_value)
        更新：theano.function([], target, updates=[(w, new_w)]) or function([], target, updates={w:new_w})
        
    - function
    	function是符号表达式执行的入口，通过functon为符号变量赋值，并执行相应的操作。
        声明：theano.function(inputs, outputs, updates, givens). inputs是一组符号变量的list，作为function的输入; outpus是需要的输出，可以是一个符号变量（or op），也可以list; updates对共享变量进行更新; givens是一个pairs(Var1, Var2)的list, tuple or dict，用第Var2来替换图中的节点Var1.
        注意：可以采用 In 来为符号来为function的输入指定默认值，如： theano.function([x, theano.In(y, value=1)], x+y)
        
# Logistic Regression

- 定义输入
	定义输入 $x$ 和输出 $y$，其中前者是矩阵，维度0代表样本数，维度1代表样本的表示；后者是向量，对应于样本 $y$ 的正确输出。    
```python
# define the input
x = T.dmatrix(name="x")
y = T.dvector(name="y")  
```

- 初始化模型参数
	根据输入的样本的表示的维度 $dim$ ，初始化模型参数 $w$ 和 $b$，并进行shared。 前者为向量，维度为 $dim$；后者为标量。
```python
# 如果未给定w，則随机初始化
if w == None:
	self.w = shared(rng.randn(dim), name="w")
else:
	self.w = shared(w, name = "w")
# 如果未给定b， 则随机初始化
if b == None:
	self.b = shared(0., name = "b")
else:
	self.b = shared(b, name = "b")
```

- 定义LogisticRegression， Prediction， error， cost
	- LogisticRegression函数定义： $g(x) = \frac {1.0} {1.0 + e^{-x}}$
	
    - Prediction函数定义： $predict = \begin{cases} 1,\ g(x) > 0.5 \\ 0,\ g(x) < 0.5  \end{cases}$
    
    - error函数定义： $error = -\frac{1}{N} \sum_{i=1}^{N} y_i * \log g_i$
    
    - cost是在cross entropy的基础上，为 $w$ 加上一个平方正则化项。  
- 
```python
# define the regression(g), prediction, error, cost
self.g = 1./(1. + T.exp(-T.dot(x,self.w) - self.b))
self.prediction = self.g > 0.5
self.error = -y*T.log(self.g) - (1-y)*T.log(1-self.g)
cost = self.error.mean() + regular * (self.w ** 2).sum()
```

- 计算梯度
```python
# compute gradient
gw, gb = T.grad(cost, [self.w, self.b])
```
- 定义入口函数
```python
# define reg, train, test
self.predict = theano.function([x], self.prediction)
self.train = theano.function([x, y], self.error, updates = [(self.w, self.w - step * gw), (self.b, self.b - step * gb)])
self.test = theano.function([x,y], [self.prediction, 1 - T.mean(T.neq(self.prediction, y))])
```

