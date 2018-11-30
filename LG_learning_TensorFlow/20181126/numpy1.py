#import tensorflow as tf
#print(tf.__version__)

import numpy as np

numbers = np.array(range(1,11), copy=True)
print(numbers)

a = np.random.seed(12345)
print(a)

array = np.array([[1., 2., 3. ], [4., 5., 6. ]])
print('\n', array)

print('\n', array*array)

print('\n', array-array)

print('\n', 1/array)

print('\n', array**0.5)

array2 = np.array([[0., 4., 1.], [7., 2., 12.]])
print("\n", array2)
print("\n", array2 > array)


names = np.array(["Bob", "Joe", "Will", "Bob", "Joe", "Joe", "Will"])
print("\n", names)
data = np.random.randn(7,4)
print("\n", data)

print("\n", names == "Bob")
print("\n", data[names == "Bob"])

print("\n", data[names == "Bob", 2:])
print("\n", data[names == "Bob", 3])

print("\n", data[names != "Bob"])
print("\n", data[~(names == "Bob")])

mask = (names == 'Bob') | (names == 'Will')
print("\n", mask)
print("\n", data[mask])

array3 = np.arange(32)
print("\n", array3)
array4 = array3.reshape((8,4))
print("\n", array3)
print("\n", array4)
print("\n", array4.T)


array5 = np.random.randn(5,4)
print("\n", array5)

print("\nMean1:", array5.mean())
print("\nMean2:", np.mean(array5))
print("\nSum:", array5.sum())

print("\n", array5.mean(axis=1))
print("\n", array5.sum(axis=0))

array6 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
print("\n", array6.cumsum())

array7 = np.array([[0, 1, 2], [3, 4,5], [6, 7, 8]])
print("\n", array7)

print("\n", array7.cumsum(axis=0))
print("\n", array7.cumprod(axis=1))

arr = np.random.randn(100)
print("\n", arr)
print("\n", arr[2:6][:3])
print("\n", (arr > 0).sum())

arr = np.random.randn(5, 3)
print("\n", arr)

arr.sort(axis=1)  #row
print("\n", arr)

arr.sort(axis=0)  #col
print("\n", arr)


x = np.array([[1,2,3], [4,5,6]])
y = np.array([[6, 23], [-1, 7], [8, 9]])

print("\n", x)
print("\n", y)
print("\n matrix mul\n", x.dot(y))