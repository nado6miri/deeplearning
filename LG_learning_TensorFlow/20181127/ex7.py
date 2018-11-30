import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input_data = [1, 2, 3, 4, 5]
x = tf.placeholder(dtype=tf.float32)
y = x*2

sess = tf.Session()
result = sess.run(y, feed_dict = {x : input_data})
print(result)