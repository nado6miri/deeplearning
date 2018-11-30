import numpy as np

arr = np.arange(10)
print('\n',arr[5])
print('\n',arr[5:8])
arr[5:8] = 12
print('\n',arr)

arr_slice = arr[5:8]
print('\n',arr_slice)

arr_slice[1] = 12345
print('\n',arr)

arr_slice[:] = 64
print('\n',arr)


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print('\n',arr3d)


print('\n\n',arr3d[0])

old_values = arr3d[0].copy()
print('\n',old_values)
arr3d[0] = 42
print('\n',arr3d)
arr3d[0] = old_values
print('\n',arr3d)

print('\n',arr3d[1, 0])

