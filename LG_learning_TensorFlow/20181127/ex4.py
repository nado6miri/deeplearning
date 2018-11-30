import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.constant([1,2,3], name='x_node', dtype=tf.float64)
print(x.dtype)

x = tf.cast(x, tf.int64)
print(x.dtype)


sess = tf.Session()
result = sess.run(x)
print(result)
sess.close()