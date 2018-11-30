import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(12345)
x_data = np.random.randn(5, 10)
w_data = np.random.randn(10, 1)

x = tf.placeholder(dtype=tf.float32, shape=(None, 10)) # 앞의 개수는 상관없이 받아들이겠음.
w = tf.placeholder(dtype=tf.float32, shape=(10, 1))
y = tf.matmul(x, w)
b = tf.fill((5, 1), -1.)
y_b = y + b
max = tf.reduce_max(y_b)

with tf.Session() as sess :
    _y_b, _max = sess.run([y_b, max], feed_dict={ x : x_data, w : w_data })
print("y_b = {}, \n max = {}".format(_y_b, _max))