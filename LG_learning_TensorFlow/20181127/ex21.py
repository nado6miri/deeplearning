import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Tensorflow broadcast example

x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = tf.constant([[1, 1, 1], [2, 2, 2]])

'''
subXY = tf.subtract(x,y)
sess = tf.Session()
result = sess.run(subXY)
print(result)
'''

expanded_X = tf.expand_dims(x, 0) #row
expanded_Y = tf.expand_dims(y, 1) #col
print('x =>', x.shape)
print('y =>', y.shape)
print('expaned_X =>', expanded_X.get_shape())
print('expaned_Y =>', expanded_Y.get_shape())

subXY = tf.subtract(expanded_X,expanded_Y)
sess = tf.Session()
result, _expanded_X, _expanded_Y = sess.run([subXY, expanded_X, expanded_Y])
print('\nexpaned_X =>', _expanded_X)
print('\nexpaned_Y =>', _expanded_Y)
print('\nresult =>', result)
