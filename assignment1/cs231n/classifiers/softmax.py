import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_train = X.shape[0]
  #############################################################################
  # DONE: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
      scores = np.exp(X[i].dot(W)) 
      correct_class_scores = scores[y[i]]
      prob_dist = correct_class_scores/float(np.sum(scores))
      loss += -np.log(prob_dist) # cross-entropy loss function
      dscores = prob_dist
      dscores[y[i]] -=1  # dscores = P(class scores) - 1 if(class = image_label)
      dW += (X[i].T).dot(dscores) # update dW by chain rule

  loss = loss/num_train + reg * np.sum(W*W)
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
  num_train = X.shape[0]

  #############################################################################
  # DONE: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.exp(X.dot(W))
  correct_class_scores = scores[[0,-1],y]
  prob_dist = np.divide(correct_class_scores,float(np.sum(scores,axis=1)))
  loss = np.sum(-np.log(prob_dist))/float(num_train) + reg * np.sum(W*W) # cross entropy loss function
  dscores = prob_dist
  dscores[[0,-1],y] -= 1 # dscores = P(class scores) - 1 if(class=image_label)
  dW += (X.T).dot(dscores) #update dW by chain rule
#############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################


  return loss, dW

