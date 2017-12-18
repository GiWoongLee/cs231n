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
        num_train = X.shape[0]


        for i in xrange(num_train):
            scores = np.exp(X[i].dot(W)) # exp on class scores of i-th image
            correct_class_scores = scores[y[i]]
            prob_dist = correct_class_scores/float(np.sum(scores))
            loss += -np.log(prob_dist) # cross_entrophy
            dscores = prob_dist
            dscores[y[i]] -= 1
            dW += (X[i].T).dot(dscores) 
            
        loss = loss/num_train + reg * np.sum(W*W)
	
	return loss, dW

def softmax_loss_vectorized(W,X,y,reg):
	'''
	Softmax loss function, vectorized version.
	Inputs and outputs are the same as softmax_loss_naive.
	'''

	# Initialize the loss and gradient to zero
	loss = 0.0
	dW = np.zeros_like(W)
        num_train = X.shape[0]

        scores = np.exp(X.dot(W)) # exp on class scores of all images
        correct_class_scores = scores[[0,-1],y]
        prob_dist = np.divide(correct_class_scores,float(np.sum(scores,axis=1)))
        loss = np.sum(-np.log(prob_dist))/float(num_train) + reg * np.sum(W*W)
        
        dscores = prob_dist
        dscores[[0,-1],y] -= 1
        dW += (X.T).dot(dscores)
        
	return loss, dW


