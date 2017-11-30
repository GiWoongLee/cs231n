import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W,X,y,reg):
	'''
	W : ndarray of shape (D,C) containing weights
	X : ndarray of shape (N,D) containing training data
	y : ndarray of shape (N,) containing training data label
	reg : regularization strength
	'''

	dW = np.zeros(W.shape) # Initialize the gradient as zero

	# compute the loss and gradient
	num_classes = W.shape[1]
	num_train = W.shape[0]
	loss = 0.0

	for i in xrange(num_train):
		scores = X[i].dot(W)
		correct_class_score = scores[y[i]]
		for j in xrange(num_classes):
			if j == y[i]:
				continue
			margin = scores[j] - correct_class_score + 1
			if margin > 0:
				loss += margin	
			
	loss /= num_train

	# Add regularization loss to loss
	loss += reg * np.sum(W*W)
	
	# TODO: compute the gradient of loss function and store it dW
	# NOTE: compute loss and gradient altogether




def svm_loss_vectorized(W,X,y,reg):
	loss = 0.0
	dW = np.zeros(W.shape)

	#TODO : implement a vectorized version of the structured SVM loss, storing the result in loss

	#TODO : implement a vectorized version of gradient for the structured SVM loss, storing the result in dW.

	return loss, dW	
