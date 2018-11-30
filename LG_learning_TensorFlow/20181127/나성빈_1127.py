import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def run_graph(in_data) :
    g = tf.Graph()
    with g.as_default() :
        with tf.name_scope("variables") :
            with tf.name_scope("input"):
                input = tf.placeholder(dtype=tf.float32, shape=(None), name="input")

            with tf.name_scope("immediate_layer"):
                sum = tf.reduce_sum(input, name='sum')
                mul = tf.reduce_prod(input, name='prod')

            with tf.name_scope("output"):
                add = tf.add(sum, mul, name='add')
                sess = tf.Session()
                _sum, _mul, _add = sess.run([sum, mul, add], feed_dict={ input : in_data })

                print("\nInput Data = {}".format(in_data))
                print("\n_sum = {}".format(_sum))
                print("\n_mul = {}".format(_mul))
                print("\n_add = {}".format(_add))

    tf.summary.FileWriter('./my_graph', graph=g)

run_graph([2, 8])
run_graph([3, 1, 3, 3])
run_graph([8, 5, 2, 2, 1, 1, 8])
run_graph([2, 3, 7])
