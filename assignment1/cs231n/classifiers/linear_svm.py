import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    incorrect_classification_count = 0 # count (incorrect_class_score - correct_class_score + safety_margin > 0)
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        np.add(dW[:,j],X[i].T)
        incorrect_classification_count += 1
    np.add(dW[:,y[i]],-1 * incorrect_classification_count * X[i].T)
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  scores = X.dot(W)
  correct_class_scores = scores[[0,-1],y] 
  margin = np.subtract(scores,correct_class_scores) + 1 # note delta as 1
  margin[[0,-1],y] = 0 # set margin of correct_class 0
  
  #############################################################################
  # DONE:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  loss = np.sum(margin[margin>0])/float(num_train) + reg * np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # DONE:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  incorrect_classification_count = np.sum((margin>0),axis=1)
  
  # update gradient on incorrect_classification
  for (image_idx, class_idx) in np.ndenumerate(margin):
      if margin[image_idx][class_idx] >0 :
          dW[:,class_idx] += X[image_idx].T

  for idx in y:
    dW[:,y[idx]] += (-1 * incorrect_classification_count[idx] * X[idx]).T

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
