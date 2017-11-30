import numpy as np

train_data = np.array([[1,2,3],[0,0,0],[-4,-2,-5],[5,7,10],[0,1,-1],[-5,-19,-9]])
train_data_label = np.array([1,2,3,1,2,3])
batch_size = 2
train_data_idxs = np.arange(6)

batch_idxs = np.random.choice(np.arange(train_data.shape[0]),batch_size)
X_batch = train_data[batch_idxs]
y_batch = train_data_label[batch_idxs]

print X_batch
print y_batch

