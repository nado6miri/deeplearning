import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

A = tf.constant([[1,2,3], [4,5,6]])
print(A.get_shape())

x = tf.constant([1,0,1])
print(x.get_shape())

x = tf.expand_dims(x, axis=1) #row
print(x.get_shape())

b = tf.matmul(A, x)

sess = tf.Session()
print("matmul result:{0}".format(sess.run(b)))

print(sess.run(x))
