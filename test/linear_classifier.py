from __future__ import print_function

import numpy as np
from linear_svm import *
from softmax import *
from past.builtins import xrange


class LinearClassifier(object):
	def __init__(self):
		# train_data(image/pixel), test_data(image/pixel)
			train_data = np.array([[10,12,13],[0,0,0],[-5,-9,-13],[5,9,10],[-1,1,0],[-4,-5,-6]])
			train_data_label = np.array([1,2,3,1,2,3])
			test_data = np.array([[7,5,3],[0,2,0],[-8,-5,-8]])
			test_label = np.array([1,2,3])
		# initialize W
			self.W = None
		# train
			# NOTE: need hyperparameter tuning(learning_rate:0.01,regularization_strength:1e-5)
			self.train(train_data,train_data_label,0.01,1e-5,100,6,True)
		# predict
			test_label_pred = self.predict(test_data)
		# accuracy calculation(print)
			print("Accuracy : %d\%"%np.sum(test_label,test_label_pred)/float(test_label.shape[0]))

	def train(self,X,y,learning_rate=1e-3,reg=1e-5,num_iters=100,batch_size=200, verbose=False):
		# training data X with shape of (N,D)
		# training data label y with shape of (N,)
		# learning rate => hyperparameter tuning
		# regularization strength => hyperparameter tuning
		# num_iters => optimization iteration
		# batch_size => stochastic gradient descent(200)
		num_train, dim = X.shape
		num_classes = max(y)+1
		if self.W is None:
			# lazily initialize W with shape of (D,K)
			# theta of class K following axis=1
			self.W = 0.001 * np.random.randn(dim,num_classes)		
		# Run stochastic gradient descent to optimize W
		loss_history = []

		for it in xrange(num_iters):
			batch_idxs = np.random.choice(np.arange(num_train),batch_size)
			X_batch = X[batch_idxs] #(dim,batch_size)
			y_batch = y[batch_idxs] #(batch_size,)

			# evaluate loss and gradient
			loss, grad = self.loss(X_batch,y_batch,reg)
			loss_history.append(loss)

			# TODO: update W with gradients and learning rate			

			if verbose and it % 100 == 0:
				print("iteration %d / %d : loss %f" %(it, num_iters,loss))

		return loss_history

	def predict(self,X):
		# self.W with shape of (D,K)
		# test data X with shape of (N,D)
		#y_pred = np.zeros(X.shape[0])

		# NOTE: if class number starts from 0 => done, else => y_pred += 1
		y_pred = np.argmax(X.dot(W),axis=1)
		return y_pred

	def loss(self,X_batch,y_batch,reg):
		# Compute the loss function and its derivative
		# subclasses will override this
		# X_batch : ndarray of shape (N,D)
		# y_batch : ndarray of shape (N,)
		# reg : regularization strength
	
		# return tuple containing : loss as single float 
		# and gradient with respect to self.W, an array of same shape of W


class LinearSVM(LinearClassifier):
	def loss(self,X_batch,y_batch,reg):
		return svm_loss_vectorized(self.W,X_batch,y_batch,reg)

class Softmax(LinearClassifier):
	def loss(self,X_batch,y_batch,reg):
		return softmax_loss_vectorized(self.W,X_batch,y_batch,reg)
