import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  df = np.zeros((num_train, num_classes))
  for i in range(num_train):
    scores = X[i].dot(W)
    tmp = np.sum(np.exp(scores))
    loss = loss - scores[y[i]]+ np.log(tmp)
    df[i] += np.exp(scores) / tmp
    for j in range(num_classes):
        if j == y[i]:
            df[i][j] -= 1
    dW += (X[i].reshape((X[i].shape[0],1))).dot(df[i].reshape((1,df[i].shape[0])))
  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  tmp = np.sum(np.exp(scores),axis = 1)
  df = np.exp(scores) / (tmp.reshape((num_train, 1)))
  df[range(num_train), y] -= 1
  dW = (X.T).dot(df)
  loss = loss - scores[range(num_train), y].sum()+ np.log(tmp).sum()
  loss = loss / num_train + reg * np.sum(W*W)
  dW = dW / num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

