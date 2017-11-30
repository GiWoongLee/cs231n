import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W,X,y,reg):
	'''
	Softmax loss function, naive implementation

	W : ndarray of shape (D,C) containing weights
	X : ndarray of shape (N,D) 
	y : ndarray of shape (N,) containing train_labels
	reg : regularization strength
	'''

	# Initialize the loss and gradient to zero
	loss = 0.0
	dW = np.zeros_like(W)

	#TODO: compute the sofmax loss and gradient 
	#NOTE: with explicit loop, need regularization
	
	return loss, dW

def softmax_loss_vectorized(W,X,y,reg):
	'''
	Softmax loss function, vectorized version.
	Inputs and outputs are the same as softmax_loss_naive.
	'''

	# Initialize the loss and gradient to zero
	loss = 0.0
	dW = np.zeros_like(W)

	# TODO: compute the softmax loss and gradient
	# NOTE: without explicit loop, need regularization
	
	return loss, dW


