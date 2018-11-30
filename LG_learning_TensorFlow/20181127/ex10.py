import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

my_var = tf.Variable(1)
my_var_times_two = my_var.assign(my_var * 2)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print("my_var = ", sess.run(my_var_times_two))
print("my_var = ", sess.run(my_var_times_two))
print("my_var = ", sess.run(my_var_times_two))
