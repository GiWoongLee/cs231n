import numpy as np
from past.builtins import xrange
from collections import Counter

class Knn(object):

	def __init__(self):
		# train_image, test_image
		train_image = np.array([[10,15,13],[0,0,0],[-5,-6,-8],[5,6,10],[0,1,0],[-8,-6,-3]])
		train_image_label = np.array([1,2,3,1,2,3])
		test_image = np.array([[1,-1,0],[-8,-8,-8],[12,18,15]])
		# train knn with train_image
		self.train(train_image,train_image_label)	
                # predict test_image
                #print self.predict(test_image,2,0)
                #print self.predict(test_image,2,1)
                #print self.predict(test_image,2,2)

	def train(self,X,y):
		self.X_train = X
		self.y_train = y

	
	def predict(self,X,k=1,num_loops=0):
		if num_loops == 0:
			dists = self.compute_distances_no_loops(X)
		elif num_loops == 1:
			dists = self.compute_distances_one_loop(X)
		elif num_loops == 2:
			dists = self.compute_distances_two_loops(X)
		else:
			raise ValueError('Invalid value %d for num loops' % num_loops)
		return self.predict_labels(dists,k=k)

	def compute_distances_no_loops(self,X):
		num_test = X.shape[0]
		num_train = self.X_train.shape[0]
		image_dimension = X.shape[1]
		# expand test_image array into shape of (num_test,num_train,image_dimension)
		expanded_test = np.repeat(X[:,np.newaxis,:],num_train,axis=1)
		# expand train_image array into shape of (num_test,num_train,image_dimension)
		expanded_train = np.resize(self.X_train,(num_test,num_train,image_dimension))	
		# calculate l2 distance by ndarray
		dists = np.sqrt(np.sum(np.square(np.subtract(expanded_test,expanded_train)),axis=2))
		return dists

	def compute_distances_one_loop(self,X):
		num_test = X.shape[0]
		num_train = self.X_train.shape[0]
		dists = np.zeros((num_test,num_train))
		for i in xrange(num_test):
			# calculate l2 distance by row
			dists[i,:] = np.sqrt(np.sum(np.square(np.subtract(self.X_train,X[i,:])),axis=1)).T
		return dists

	def compute_distances_two_loops(self,X):
		num_test = X.shape[0]
		num_train = self.X_train.shape[0]
		dists = np.zeros((num_test,num_train))
		for i in xrange(num_test):
			for j in xrange(num_train):
				# calculate l2 distance by element
				dists[i,j] = np.sqrt(np.sum(np.square(X[i,:]-self.X_train[j,:])))
		return dists

	def predict_labels(self,dists,k=1):
		num_test = dists.shape[0]
		y_pred = np.zeros(num_test)
		for i in xrange(num_test):
			# get index of k-smallest elements
			closest_y = dists[i,:].argsort()[:k]
			# get labels of closest train images
			closest_y_label = map(lambda ele: self.y_train[ele],closest_y)
			# predict test image with most frequent value from closest train images
			knn_labels = Counter(closest_y_label).most_common(1)
			# Break ties by choosing the smaller label
			if len(knn_labels) > 1 :
				y_pred[i] = min(map(lambda label: label[0] , knn_labels))
			else: y_pred[i] = knn_labels[0][0]
		return y_pred

if __name__ == "__main__":
	print "Program executed"
	knn = Knn()
	print "Program finished"
