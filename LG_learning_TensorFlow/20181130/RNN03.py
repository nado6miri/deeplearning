import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#cell 과 cell 간의 전달시 오차 계산을 하는...??

y_data = tf.constant([[1, 1, 1]])

prediction = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)
weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=prediction, targets=y_data, weights=weights)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Loss = ", sess.run(sequence_loss))


