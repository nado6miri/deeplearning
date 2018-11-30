import numpy as np

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
print('\n',np.unique(names))
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
print('\n',np.unique(ints))

values = np.array([6, 0, 0, 3, 2, 5, 6])
print('\n',np.in1d(values, [2, 3, 6]))

print('\n',np.union1d(values, [2, 3, 6]))

print('\n',np.intersect1d(values, [2, 3, 6]))

print('\n',np.setdiff1d(values, [2, 3, 6]))
