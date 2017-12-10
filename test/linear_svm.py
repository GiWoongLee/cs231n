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
        loss_factor = 0 #count margin > 0 to reflect on dW

	for i in xrange(num_train):
		scores = X[i].dot(W)
		correct_class_score = scores[y[i]]
                loss_factor = 0 # count (incorrect_class_score - correct_class_score > safety_margin)
		for j in xrange(num_classes):
			if j == y[i]:
                            continue
			margin = scores[j] - correct_class_score + 1
			if margin > 0:
			    loss += margin
                            np.add(dW[:,j],X[i].T) # update gradient on incorrect class
                            loss_factor += 1 # reflect on updating gradient on correct class
                np.add(dW[:,y[i]],-1 * loss_factor * X[i].T) #update gradient on correct class
	loss /= num_train

	# Add regularization loss to loss
	loss += reg * np.sum(W*W)

        return loss, dW



def svm_loss_vectorized(W,X,y,reg):
	loss = 0.0
	dW = np.zeros(W.shape)
         
        scores = X.dot(W)
        correct_class_scores = scores[[0,-1],y] # NOTE: numpy advanced indexing used
        num_train = W.shape[0]

        # NOTE: map of (incorrect_class_scores - correct_class_scores + safety_margin)
        hinge_loss_map = np.subtract(scores,correct_class_scores) + 1 # safety margin as 1
        hinge_loss_map[[0,-1],y] = 0 # set correct_class_scores to 0
        hinge_loss_map_bool = (hinge_loss_map > 0) # make boolean map with condition : (incorrect_class_scores - correct_class_scores + safety_margin >0)
        loss_factor = np.sum(hinge_loss_map_bool,axis=1) # count (incorrect_class_scores - correct_class_scores > safety_margin)
        
        loss = (hinge_loss_map * hinge_loss_map_bool).sum()/num_train + reg * np.sum(W*W)
        
        # iterator over ndarray 
        for (index, boolVal) in np.ndenumerate(hinge_loss_map_bool):
            if boolVal == True: # as correct_class_scores is 0, hinge_loss_map_bool[correct_class_scores] is false
                dW[index[1]] += X[index[0]].T # index0 indciates row, while index1 indicates column 
            else:
                pass

        # update gradient on correct class
        for image_num in xrange(num_train):
            dW[:,y[image_num]] += (-1 * X[:,image_num] * loss_factor[image_num]).T

	return loss, dW	
