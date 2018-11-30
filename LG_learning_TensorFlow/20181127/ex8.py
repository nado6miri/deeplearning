import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.placeholder(tf.int32, shape=[2, 2], name='my_input')
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name='sum_c')

d = tf.add(b, c, name="add_d")

sess = tf.Session()
tf.summary.FileWriter("./my_graph", sess.graph)

input_dict = { a:np.array([[5,3], [4, 7]], dtype=np.int32) }
print(sess.run([b, c, d], feed_dict=input_dict))