import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

g = tf.Graph()
print(g)

a = tf.constant(5)
print(a.graph is g)
print(tf.get_default_graph())
print(a.graph is tf.get_default_graph())
# 아래 두 문장은 동일함..   default graph 에 그림을 그리기 때문에....
#tf.summary.FileWriter('./my_graph', sess.graph)
tf.summary.FileWriter('./my_graph', graph=tf.get_default_graph())

