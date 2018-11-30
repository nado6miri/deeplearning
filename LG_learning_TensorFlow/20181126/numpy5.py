import numpy as np 
np.random.seed(12345)
arr2 = np.arange(32)
print('\n',arr2)
arr3 = arr2.reshape((8, 4))
print('\n',arr3)
print('\n',arr3.T)


