import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

c = tf.constant([[1, 2, 3], [4, 5, 6 ]])
print("python list input:{0}".format(c.get_shape()))

c = tf.constant(np.array([
    [[1, 2, 3],
     [4, 5, 6 ]],
    [[1, 1, 1],
     [2, 2, 2]],
    ]))
print("3d numpy array input:{0}".format(c.get_shape()))