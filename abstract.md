### Numpy
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
## Class 1 KNN(K近邻)
1. 全向量化表达$(X+Y)^2=X^2+Y^2+2XY$

## Class 2 SVM && softmax
### SVM(supported vector machine)
- 损失函数Loss Function: $L=\sum_{j\neq y_i}max(0, ω_j^Tx_i-ω_{y_i}^Txi+\Delta)$
+ 因为损失函数的可行解是不唯一的, 所以需要正则化(regularization): $$L=\frac{1}{N}\sum_i\sum_{j\neq y_i}max(0, ω_j^Txi-ω_{y_i}^Txi+\Delta)+\lambda \sum_k\sum_lW_{k,l}^2$$不失一般性地$\Delta$和$\lambda$反映的是同一个特征, 所以取$\Delta$=1.0。
### softmax
- Score function: $f(x_i;W)=Wx_i$ stays unchanged
+ Loss function: replace the hinge loss(折叶损失) with a **cross-entropy loss**(交叉熵) that has the form:$$L_i = -log(\frac{e^{f_{y_i}}}{\sum_{j}{e^{f_j}}})$$ **or equivalently** $$L_i = -f_{y_i}+log\sum_j{e^{f_j}}$$It squashes real-valued scores to a vector of values **between zero and one** that sum to one.(规整化) The full cross-entropy loss is relatively easy to motivate.
$$L = \frac{1}{N}\sum_i(-f_{y_i}+log\sum_je^{f_j})+\lambda\sum_k\sum_lW_{k,l}^2$$
- Information theory view
+ Probabilistic interpretation: logistic regression的推广，把score理解为未归一化的对数概率
- Numeric stability：编程实现softmax函数计算的时候，因为存在指数函数，所以可能导致数值计算的不稳定，所以通常先将```scores -= np.max(scores)```，使得最大值为0，再进行后续的计算。

## Class 3 Optimization
#### Visualizing the loss function
- convex optimization(凸优化)
+ non-differentiable loss functions -> subgradient(次梯度)
#### Optimization - Gradient Descent
*Our strategy will be to start with random weights and iterativefy refine them over time to get lower loss*
- Computing the gradient numerically with finite differences(有限差分法)
**centered difference formula(中心差分公式):**$$[f(x+h)-f(x-h)]/2h$$
+ Computing the gradient analytically with Calculus
**$$\nabla_{w_{y_i}}L_i=-(\sum_{j\neq{y_i}}1(w_j^Tx_i-w_{y_i}^Tx_i+\Delta>0))x_i$$** For the other rows $j\neq y_i$ the gradient is: **$$\nabla_{w_j}L_i=1(w_j^Tx_i-w_{y_i}^Tx_i+\Delta>0)x_i$$** Once you derive the expression for the gradient it is straight-forward to implement the expressions and use them to perform the gradient update.
    - *总是使用分析梯度，但是为确保正确性，需做gradient check*
    + *learning rate is a important hyperparameter*
- **Minibatch Gradient Descent**

## Class 4 Backpropogation
- **在不同分支的梯度要相加(Gradients add up at forks)**：如果变量x，y在前向传播的表达式中出现**多次**，那么进行反向传播的时候就要非常小心，使用```+=```而不是```=```来累计这些变量的梯度（不然就会造成覆写）。这是遵循了在微积分中的*多元链式法则*，该法则指出如果变量在线路中分支走向不同的部分，那么梯度在回传的时候，就应该进行累加。
+ **Matrix-Matrix multiply gradient**：要分析维度！注意不需要去记忆dW和dX的表达，因为它们很容易通过**维度推导**出来。

## Class 5 Training neural-networks
#### Commonly used activation functions 
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
#### Fully-Connected layers
#### Setting number of layers and their sizes
*use as **big** of a neural network as your computational budget allows, and use other **regularization** techniques to control overfitting*

#### Data Preprocessing
- **Mean Subtraction**: `X -= np.mean(X, axis = 0)`. For convenience it can be common to subtract a single value from all pixels (e.g. `X -= np.mean(X)`), or to do so separately across the three color channels.
+ **Normalization**: be of approximately the same scale `X /= np.std(X, axis = 0)`
- **PCA and Whitening(主成分分析与白化处理)**
主成分分析即提取主要的特征方向，使得协方差矩阵为对角阵。白化处理是单位化数据的特征值，使得协方差矩阵为单位阵。
	```python
	# Assume input data matrix X of size [N x D]
	X -= np.mean(X, axis = 0) # zero-center the data (important)
	cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix(协方差矩阵)

	U,S,V = np.linalg.svd(cov) #奇异值分解
	#the columns of U are the eigenvectors 
	#and S is a 1-D array of the singular values
	#(因为cov是对称且半正定的，所以S中元素是特征值的平方)

	Xrot = np.dot(X, U) # decorrelate the data into the eigenbasis
	```
	+ **np.linalg.svd**的一个良好性质是在它的返回值U中，特征向量是按照特征值的大小排列的。我们可以利用这个性质来对数据降维，即主成分分析（Principal Component Analysis 简称PCA）降维。
	```python
	#U, the eigenvector columns are sorted by their eigenvalues. 
	Xrot_reduced = np.dot(X, U[:,:100]) # Xrot_reduced becomes [N x 100]
	```
	- **whitening** operation
	```python
	# whiten the data:
	# divide by the eigenvalues (which are square roots of the singular values)
	Xwhite = Xrot / np.sqrt(S + 1e-5)
	#adding 1e-5 (or a small constant) to prevent division by zero
	```
	+ 该变换的一个缺陷是在变换的过程中可能会夸大数据中的噪声，这是因为它将所有维度都拉伸到相同的数值范围，这些维度中也包含了那些只有极少差异性(方差小)而大多是噪声的维度。在实际操作中，这个问题可以用更强的平滑来解决（例如：采用比1e-5更大的值）。
+ **Attention: The mean must be computed only over the *training data* and then subtracted equally from all splits (train/val/test).**
- 实际上，在卷积神经网络中并不会采用PCA或白化操作。然而对数据进行零中心化操作还是非常重要的，对每个像素进行归一化也很常见。

#### Weight Initialization
- Small random numbers
+ calibrating the variances with 1/sqrt(n)
- In practice the current recommendation is to use ReLU units and use the ```w = np.random.randn(n) * sqrt(2.0/n)``` [Reference Article](https://arxiv-web3.library.cornell.edu/abs/1502.01852/)
+ [*Batch Normalization*](https://arxiv.org/abs/1502.03167/): take on a unit gaussian distribution at the beginning of the training

#### Regularization
- L2 regularization
+ **inverted dropout**
- In practice, it's most common to use a single, global L2 regularization strength that is cross-validated. It's also common to combine this with dropout applied after all layers. The value of **p = 0.5** is a reasonable default, but this can be tuned on validation data.

#### Parameter Updating
- Momentum update
	```python
	# Momentum update
	v = mu * v - learning_rate * dx # integrate velocity
	x += v # integrate position
	```
	Note that this is different from the SGD update shown above, where the gradient directly integrates the position. Instead, the physics view suggests an update in which the gradient only directly influences the velocity, which in turn has an effect on the position.

	Here we see an introduction of a v variable that is initialized at zero, and an **additional hyperparameter (mu)**.

	This parameter is usually set to values such as [0.5, 0.9, 0.95, 0.99]. A typical setting is to start with momentum of about 0.5 and anneal it to 0.99 or so over multiple epochs. 
	
	*With Momentum update, the parameter vector will build up velocity in any direction that has consistent gradient.*
+ Nesterov Momentum
	```python
	x_ahead = x + mu * v
	# evaluate dx_ahead (the gradient at x_ahead instead of at x)
	v = mu * v - learning_rate * dx_ahead
	x += v
	```
	treat the future approximate position x + mu * v as a "lookahead"

	In practice people prefer to express the update to look as similar to vanilla SGD or to the previous momentum update as possible.
	```python
	v_prev = v # back this up
	v = mu * v - learning_rate * dx # velocity update stays the same
	x += -mu * v_prev + (1 + mu) * v # position update changes form
	```
