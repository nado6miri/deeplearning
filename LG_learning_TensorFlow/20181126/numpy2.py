import numpy as np
np.random.seed(12345)

arr = np.array([[1., 2., 3.], [4., 5., 6.]])
print('\n',arr)

print('\n',arr * arr)

print('\n',arr - arr)

print('\n',1 / arr)

print('\n',arr ** 0.5)

arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
print('\n',arr2)
print('\n',arr2 > arr)
