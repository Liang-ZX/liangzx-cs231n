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
- 因此，更为一般地，我们有:
$$L=\frac{1}{N}\sum_i\sum_{j\neq y_i}max(0, f_j-f_{y_i}+1.0)+\lambda \sum_k\sum_lW_{k,l}^2$$
### softmax
- Score function: $f(x_i;W)=Wx_i$ stays unchanged
+ Loss function: replace the hinge loss(折叶损失) with a **cross-entropy loss**(交叉熵) that has the form:$$L_i = -log(\frac{e^{f_{y_i}}}{\sum_{j}{e^{f_j}}})$$ **or equivalently** $$L_i = -f_{y_i}+log\sum_j{e^{f_j}}$$It squashes real-valued scores to a vector of values **between zero and one** that sum to one.(规整化) The full cross-entropy loss is relatively easy to motivate.
$$L = \frac{1}{N}\sum_i(-f_{y_i}+log\sum_je^{f_j})+\lambda\sum_k\sum_lW_{k,l}^2$$
- Information theory view
+ Probabilistic interpretation: logistic regression的推广，把score理解为未归一化的对数概率
- Numeric stability：编程实现softmax函数计算的时候，因为存在指数函数，所以可能导致数值计算的不稳定，所以通常先将```scores -= np.max(scores)```，使得最大值为0，再进行后续的计算。

## Class 3 Optimization
### Visualizing the loss function
- convex optimization(凸优化)
+ non-differentiable loss functions -> subgradient(次梯度)
### Optimization - Gradient Descent
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
### Commonly used activation functions 
- *sigmoid function*: $\sigma(x)=1/(1+e^{-x})$
	drawbacks: 
	- Sigmoids saturate(饱和) and kill gradients.
	+ Sigmoid outputs are not zero-centered.
+ *Tanh(双曲正切)*: $tanh(x)=2\sigma(2x)-1$ 
- *ReLU(Rectified Linear Unit)(整流线性单元)*: $f(x)=max(0,x)$
	strengths:
	- 提高收敛速度(e.g. a factor of 6 in [Krizhevsky et al.](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf))

	drawbacks:
	- ReLU units can be fragile during training and can "die" but with a proper setting of **the learning rate** this is less frequently an issue.
+ *Leaky ReLU*: $f(x)=1(x<0)(\alpha x)+1(x>=0)(x)$ where $\alpha$ is a small constant
- *Maxout*: $max(\omega_1^Tx+b_1, \omega_2^Tx+b_2)$ it generalizes the ReLU and its leaky version

+ **Last comment**: very rare to mix and match different types of neurons in the same network 

	*"What neuron type should I use?"* Use the **ReLU non-linearity**, be careful with your learning rates and possibly monitor the fraction of "dead" units in a network. If this concerns you, give *Leaky ReLU* or *Maxout* a try.

### Fully-Connected layers
+ 神经网络可以近似任何连续函数。([Approximation by Superpositions of Sigmoidal Function](http://www.dartmouth.edu/~gvc/Cybenko_MCSS.pdf))虽然在理论上深层网络（使用了多个隐层）和单层网络的表达能力是一样的，但是就实践经验而言，深度网络效果比单层网络好。
- A fully-connected neural network with an arbitrary number of hidden layers, ReLU nonlinearities, and a *softmax* loss function. This will also implement *dropout and batch/layer normalization* as options. For a network with $L$ layers, the architecture will be

		{affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

	where batch/layer normalization and dropout are optional, and the {...} block is
	repeated L - 1 times.
+ 正确性检查(sanity check)
	- numerical gradient
	+ **initial loss check**
		由 $Var(X) = E(X^2)-E^2(X)$ 得
		$$E(\lambda\sum_k\sum_lW_{k,l}^2)=\lambda\sum_k\sum_lE(W_{k,l}^2)=\lambda\sum_k\sum_l(Var(W_{k,l})+E^2(W_{k,l}))$$又$W_{k,l}$服从于$N(0, weight\_scale^2)$，所以
		$$E(\lambda\sum_k\sum_lW_{k,l}^2)=\lambda weight\_scale^2*W元素个数$$
	- $SVM: loss = C-1+ E(\lambda\sum_k\sum_lW_{k,l}^2)$
	&nbsp;
	+ $softmax: loss = logC + E(\lambda\sum_k\sum_lW_{k,l}^2)$

### Setting number of layers and their sizes
*use as **big** of a neural network as your computational budget allows, and use other **regularization** techniques to control overfitting*

### Data Preprocessing
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

### Weight Initialization
- Small random numbers
+ calibrating the variances with `1/sqrt(n)`
- In practice the current recommendation is to use ReLU units and use the ```w = np.random.randn(n) * sqrt(2.0/n)``` [Reference Article](https://arxiv-web3.library.cornell.edu/abs/1502.01852/)
+ [*Batch Normalization*](https://arxiv.org/abs/1502.03167/): During training the sample mean and (uncorrected) sample variance are computed from **minibatch statistics** and used to normalize the incoming data.
	```python
	# During training we also keep an exponentially decaying running mean of the mean and variance of each feature, and these averages are used to normalize data at test-time.

	running_mean = momentum * running_mean + (1 - momentum) * sample_mean
	running_var = momentum * running_var + (1 - momentum) * sample_var
	```
	*Strength*: Batch normalization can mitigate the influence of weight initialization, but is relative to the **batch size**.
- [*Layer Normalization*](https://arxiv.org/pdf/1607.06450.pdf): 逐一正则化每张图的均值与方差，训练集与测试集的操作相同。

### Regularization
- L2 regularization
+ [*inverted dropout*](https://arxiv.org/abs/1207.0580): 随机失活的实现方法是让神经元以超参数$p$的概率被激活或者被设置为0。
	```python
	""" 
	Inverted Dropout: Recommended implementation example.
	We drop and scale at train time and don't do anything at test time.
	"""

	p = 0.5 # probability of keeping a unit active. higher = less dropout

	def train_step(X):
	# forward pass for example 3-layer neural network
	H1 = np.maximum(0, np.dot(W1, X) + b1)
	U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask. Notice /p!
	H1 *= U1 # drop!
	H2 = np.maximum(0, np.dot(W2, H1) + b2)
	U2 = (np.random.rand(*H2.shape) < p) / p # second dropout mask. Notice /p!
	H2 *= U2 # drop!
	out = np.dot(W3, H2) + b3
	
	# backward pass: compute gradients... (not shown)
	# perform parameter update... (not shown)
	
	def predict(X):
	# ensembled forward pass
	H1 = np.maximum(0, np.dot(W1, X) + b1) # no scaling necessary
	H2 = np.maximum(0, np.dot(W2, H1) + b2)
	out = np.dot(W3, H2) + b3
	```
	*Strength*: drop out can **alleviate the overfitting**
- In practice, it's most common to use a single, global L2 regularization strength that is cross-validated. It's also common to combine this with dropout applied after all layers. The value of **p = 0.5** is a reasonable default, but this can be tuned on validation data.
+ Reference paper
	- [Dropout paper](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) by Srivastava et al. 2014.
	+ [Dropout Training as Adaptive Regularization](http://papers.nips.cc/paper/4882-dropout-training-as-adaptive-regularization.pdf)：“我们认为：在使用"Fisher information matrix"的对角逆矩阵的期望对特征进行数值范围调整后，再进行L2正则化这一操作，与随机失活正则化是一阶相等的。”

### Parameter Updating
#### SGD and bells and whistles
- Momentum update
	```python
	# Momentum update
	v = mu * v - learning_rate * dx # integrate velocity
	x += v # integrate position
	```
	Note that this is different from the SGD update shown above, where the gradient directly integrates the position. Instead, the physics view suggests an update in which the gradient only directly influences the velocity, which in turn has an effect on the position.

	Here we see an introduction of a v variable that is initialized at zero, and an **additional hyperparameter (mu)**.

	This parameter is usually set to values such as `[0.5, 0.9, 0.95, 0.99]`. A typical setting is to start with momentum of about `0.5` and anneal it to `0.99` or so over multiple epochs. 
	
	*With Momentum update, the parameter vector will build up velocity in any direction that has consistent gradient.*
+ Nesterov Momentum
	```python
	x_ahead = x + mu * v
	# evaluate dx_ahead (the gradient at x_ahead instead of at x)
	v = mu * v - learning_rate * dx_ahead
	x += v
	```
	treat the future approximate position `x + mu * v` as a "lookahead"

	In practice people prefer to express the update to look as similar to vanilla SGD or to the previous momentum update as possible.
	```python
	v_prev = v # back this up
	v = mu * v - learning_rate * dx # velocity update stays the same
	x += -mu * v_prev + (1 + mu) * v # position update changes form
	```
#### Per-parameter adaptive learning rate methods
- [*Adagrad*](http://jmlr.org/papers/v12/duchi11a.html): 
	```python
	# Assume the gradient dx and parameter vector x
	cache += dx**2
	x += - learning_rate * dx / (np.sqrt(cache) + eps)
	```
	接收到高梯度值的权重更新的效果被减弱，而接收到低梯度值的权重的更新效果将会增强。Adagrad的一个缺点是，在深度学习中单调的学习率被证明通常过于激进且过早停止学习。
+ [*RMSprop*](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf): 对Adagrad的改进，其更新不会让学习率单调变小。
	```python
	cache = decay_rate * cache + (1 - decay_rate) * dx**2
	x += - learning_rate * dx / (np.sqrt(cache) + eps)
	```
	`decay_rate`是一个超参数，常用的值是`[0.9,0.99,0.999]`
- [*Adam*](https://arxiv.org/abs/1412.6980):
	```python
	# t is your iteration counter going from 1 to infinity
	m = beta1*m + (1-beta1)*dx
	mt = m / (1-beta1**t)
	v = beta2*v + (1-beta2)*(dx**2)
	vt = v / (1-beta2**t)
	x += - learning_rate * mt / (np.sqrt(vt) + eps)
	```
	论文中推荐的参数值`eps=1e-8, beta1=0.9, beta2=0.999`。
#### Parameter Updating Summary
In practice *Adam* is currently recommended as the default algorithm to use, and often works slightly better than RMSProp. However, it is often also worth trying *SGD+Nesterov* Momentum as an alternative.

### Learning and Evaluating
#### Gradient Checks
- Use the centered formula (second order approximation)
	$$\frac{df(x)}{dx} = \frac{f(x+h)-f(x-h)}{2h}$$
+ Use relative error for the comparison
	$$\frac{|f_a^{'} - f_n^{'}|}{max(|f_a^{'}|,|f_n^{'}|)}$$
#### Babysitting the learning process
- Loss Function
	过低的学习率导致算法的改善是线性的。高一些的学习率会看起来呈几何指数下降，更高的学习率会让损失值很快下降，但是接着就停在一个不好的损失值上。

	损失值的噪音很大(波动明显)说明批数据的数量可能太小。
+ Train/Val accuracy
	在训练集准确率和验证集准确率中间的空隙指明了模型过拟合的程度。就应该增大正则化强度（更强的L2权重惩罚，更多的随机失活等）或收集更多的数据。
	
	另一种可能是验证集曲线和训练集曲线几乎重合，这种情况说明模型容量还不够大，应该通过增加**参数数量**让模型容量更大些。
- **Ratio of weights:updates**
	Note: *updates*, not the raw gradients (e.g. in vanilla sgd this would be the gradient multiplied by the learning rate). You might want to evaluate and track this ratio for every set of parameters independently. 
	
	A rough heuristic is that this ratio should be somewhere around `1e-3`. If it is lower than this then the learning rate might be too low. If it is higher then the learning rate is likely too high. 

	```python
	# assume parameter vector W and its gradient vector dW
	param_scale = np.linalg.norm(W.ravel())
	update = -learning_rate*dW # simple SGD update
	update_scale = np.linalg.norm(update.ravel())
	W += update # the actual update
	print update_scale / param_scale # want ~1e-3
	```
+ First-layer Visualizations
	充满了噪音的图像，暗示了网络可能出现了问题：网络没有收敛，学习率设置不恰当，正则化惩罚的权重过低。
	
	训练过程良好的图: nice, smooth, clean and diverse features
