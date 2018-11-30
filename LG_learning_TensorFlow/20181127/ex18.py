import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# name scope example....

with tf.name_scope("Scope_A") :
    a = tf.add(1, 3, name="a_add")
    b = tf.multiply(a, 3, name="a_mul")

with tf.name_scope("Scope_B") :
    c = tf.add(4, 5, name="b_add")
    d = tf.multiply(c, 6, name="b_mul")

tf.Session().run(b+d)
tf.summary.FileWriter('./my_graph', graph=tf.get_default_graph())
