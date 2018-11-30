import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(10, dtype=tf.float32, name='node_a')
b = tf.constant(20, dtype=tf.float32, name='node_b')
c = tf.constant(30, dtype=tf.float32, name='node_c')

d = a*b+c

sess = tf.Session()
f = [a, b, c, d]
_a, _b, _c, _d  = sess.run(fetches=f)

print("a=", _a)
print("b=", _b)
print("c=", _c)
print("d=", _d)
