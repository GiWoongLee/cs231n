import numpy as np

exam3_data = np.array([[True,True,False],[False,True,False]])
y = [[0,1]]
print exam3_data[[0,-1],y]
print np.sum(exam3_data,axis=1)
#print exam_data[[0,-1],[0,1]]



exam2_data = np.array([[1,-2],[-2,3]])
exam2_data_bool = (exam2_data > 0)
print exam2_data* exam2_data_bool

#train_data = np.array([[1,2,3],[0,0,0],[-4,-2,-5],[5,7,10],[0,1,-1],[-5,-19,-9]])
#train_data_label = np.array([1,2,3,1,2,3])
#batch_size = 2
#train_data_idxs = np.arange(6)

#batch_idxs = np.random.choice(np.arange(train_data.shape[0]),batch_size)
#X_batch = train_data[batch_idxs]
#y_batch = train_data_label[batch_idxs]

#print X_batch
#print y_batch
