import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([5, 3])
b = tf.reduce_sum(a, name='sum_node')
c = tf.reduce_prod(a, name='multiply_node')


sess = tf.Session()
add_result = sess.run(b)
multiply_result = sess.run(c)
print("add = {0}, multiply = {1}".format(add_result, multiply_result))
tf.summary.FileWriter('./my_graph', sess.graph)
sess.close()