import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

g = tf.Graph()
print(g)
#  g  라는 새로 만든 graph를 default로 사용해라...
with g.as_default() :
    a = tf.multiply(2, 3, name='mul_a')
    b = tf.add(a, 5, name='add_b')
    c = tf.subtract(b, 4, name='sub_c')
    print(tf.Session().run(c))

tf.summary.FileWriter('./my_graph', graph=g)
