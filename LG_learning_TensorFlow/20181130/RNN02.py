import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
x_data = np.array([[[ 1, 0, 0, 0 ],
                    [ 0, 1, 0, 0],
                    [ 0, 0, 1, 0],
                    [ 0, 0, 1, 0],
                    [ 0, 0, 0, 1]]], dtype=np.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(outputs))
