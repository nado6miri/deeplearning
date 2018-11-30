import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input_data = [1, 2, 3, 4, 5]
x = tf.placeholder(dtype=tf.float32)
W = tf.Variable([3], dtype=tf.float32)
y = W*x
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y, feed_dict={ x : input_data })
print(result)