import numpy as np

arr = np.random.randn(100)
print('\n',(arr > 0).sum())

arr = np.random.randn(5,3)
print('\n',arr)

arr.sort(1)
print('\n',arr)

arr.sort(0)
print('\n',arr)
