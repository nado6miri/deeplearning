import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.constant([[1.0, 2.0, 3.0]])
w = tf.constant([[2.0], [2.0], [2.0]])
y = tf.matmul(x, w)
print(x.get_shape())

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y)
print(result)
