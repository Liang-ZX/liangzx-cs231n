# Numpy
```python
import numpy as np
```
1. concatenate
```python
np.concatenate([A,B], axis = 1)
#axis = 1 按行做, axis = 0 按列做
```
2. vstack && hstack
```python
np.vstack([A,B]) #垂直叠放
np.hstack([A,B]) #水平叠放
```
3. split && vsplit &&hsplit
```python
```
4. nditer(迭代器)
```python
it = np.nditer(x, flags = , op_flags = )
it.iternext()
```
# Class 1 KNN(K近邻)
1. 全向量化表达$(X+Y)^2=X^2+Y^2+2XY$

# Class 2 SVM && softmax
## SVM(supported vector machine)
- 损失函数Loss Function: $L=\sum_{j\neq y_i}max(0, ω_j^Tx_i-ω_{y_i}^Txi+\Delta)$;
+ 因为损失函数的可行解是不唯一的, 所以需要正则化(regularization): $L=\frac{1}{N}\sum_i\sum_{j\neq y_i}max(0, ω_j^Txi-ω_{y_i}^Txi+\Delta)+\lambda \sum_k\sum_lW_{k,l}^2$, 不失一般性地$\Delta$和$\lambda$反映的是同一个特征, 所以取$\Delta$=1.0。
## softmax classifier
- Score function: $f(x_i;W)=Wx_i$ stays unchanged
+ Loss function: replace the hinge loss(折叶损失) with a **cross-entropy loss**(交叉熵) that has the form:$$L_i = -log(\frac{e^{f_{y_i}}}{\sum_{j}{e^{f_j}}})$$**or equivalently**$$L_i = -f_{y_i}+log\sum_j{e^{f_j}}$$&nbsp;&nbsp;&nbsp;&nbsp;It takes a vector of arbitrary real-valued scores(任意的实值函数) (in $z$) and squashes it to a vector of values between zero and one that sum to one.(规整化) The full cross-entropy loss is relatively easy to motivate.
- Information theory view
+ Probabilistic interpretation: logistic regression的推广，把score理解为未归一化的对数概率
- Numeric stability

# Class 3 Optimization
## Visualizing the loss function
- convex optimization(凸优化)
+ non-differentiable loss functions -> subgradient(次梯度)
## Optimization - Gradient Descent
*Our strategy will be to start with random weights and iterativefy refine them over time to get lower loss*
- Computing the gradient numerically with finite differences(有限差分法)
**centered difference formula(中心差分公式):**$$[f(x+h)-f(x-h)]/2h$$
+ Computing the gradient analytically with Calculus
**$$\nabla_{w_{y_i}}L_i=-(\sum_{j\neq{y_i}}1(w_j^Tx_i-w_{y_i}^Tx_i+\Delta>0))x_i$$** For the other rows $j\neq y_i$ the gradient is: **$$\nabla _{w_j}L_i=1(w_j^Tx_i-w_{y_i}^Tx_i+\Delta>0)x_i$$** Once you derive the expression for the gradient it is straight-forward to implement the expressions and use them to perform the gradient update.
    - *总是使用分析梯度，但是为确保正确性，需做gradient check*
    + *learning rate is a important hyperparameter*
- **Minibatch Gradient Descent**

# Class 4 Backpropogation
## Gradients for vectorized operations

# Class 5 neural-networks-1
## Commonly used activation functions 
**non-linearity**
- *sigmoid function* $\sigma(x)=1/(1+e^{-x})$
	drawbacks: 
	- Sigmoids saturate and kill gradients.
	+ Sigmoid outputs are not zero-centered.
+ *Tanh(双曲正切)* $tanh(x)=2\sigma(2x)-1$ 
- *ReLU(Rectified Linear Unit)(整流线性单元)* $f(x)=max(0,x)$
	drawbacks:
	- ReLU units can be fragile during training and can "die" but with a proper setting of **the learning rate** this is less frequently an issue.
+ *Leaky ReLU* $f(x)=1(x<0)(\alpha x)+1(x>=0)(x)$ where $\alpha$ is a small constant
- *Maxout* $max(\omega_1^Tx+b_1, \omega_2^Tx+b_2)$ it generalizes the ReLU and its leaky version
**Last comment: very rare to mix and match different types of neurons in the same network**
*"What neuron type should I use?"* Use the ReLU non-linearity, be careful with your learning rates and possibly monitor the fraction of "dead" units in a network.
## Fully-Connected layers
## Setting number of layers and their sizes
*use as **big** of a neural network as your computational budget allows, and use other **regularization** techniques to control overfitting*

## Data Processing
- **Mean Subtraction** In numpy, this operation would be implemented as: `X -= np.mean(X, axis = 0)`. With images specifically, for convenience it can be common to subtract a single value from all pixels (e.g. `X -= np.mean(X)`), or to do so separately across the three color channels.
+ **Normalization** refers to normalizing the data dimensions so that they are of approximately the same scale. `X /= np.std(X, axis = 0)`
- **PCA and Whitening(主成分分析与白化处理)**
主成分分析即提取主要的特征方向，白化处理是单位化数据的特征值（即使得均值为0，协方差矩阵为单位阵）
	```python
	# Assume input data matrix X of size [N x D]
	X -= np.mean(X, axis = 0) # zero-center the data (important)
	cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix

	U,S,V = np.linalg.svd(cov)
	#the columns of U are the eigenvectors 
	#and S is a 1-D array of the singular values(奇异值，方阵时为特征值的平方)

	Xrot = np.dot(X, U) # decorrelate the data into the eigenbasis
	```
	+ Notice that the columns of U are a set of orthonormal(正交) vectors (norm of 1, and orthogonal to each other), so they can be regarded as **basis vectors**.
	```python
	#U, the eigenvector columns are sorted by their eigenvalues. 
	Xrot_reduced = np.dot(X, U[:,:100]) # Xrot_reduced becomes [N x 100]
	```
	- The **whitening** operation takes the data in the eigenbasis and divides every dimension by the eigenvalue to normalize the scale.
	```python
	# whiten the data:
	# divide by the eigenvalues (which are square roots of the singular values)
	Xwhite = Xrot / np.sqrt(S + 1e-5)
	#adding 1e-5 (or a small constant) to prevent division by zero
	```
	+ **Instead, the mean must be computed only over the training data and then subtracted equally from all splits (train/val/test).**
+ In practice, center the data to have mean of zero, and normalize its scale to [-1, 1] along each feature

## Weight Initialization
- Small random numbers
+ calibrating the variances with 1/sqrt(n)
- In practice the current recommendation is to use RrLU units and use the ```w = np.random.randn(n) * sqrt(2.0/n)```
+ **Batch Normalization**: take on a unit gaussian distribution at the beginning of the training

## Regularization
- L2 regularization
+ **inverted dropout**
- In practice, it's most common to use a single, global L2 regularization strength that is cross-validated. It's also common to combine this with dropout applied after all layers. The value of **p = 0.5** is a reasonable default, but this can be tuned on validation data.