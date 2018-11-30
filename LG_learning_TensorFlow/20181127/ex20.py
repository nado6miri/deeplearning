import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

graph = tf.Graph()

with graph.as_default():
    global_step = tf.Variable(0, dtype=tf.int32, name='global_step')
    increment_step = global_step.assign_add(1)
    previous_value = tf.Variable(0.0, dtype=tf.float32, name='previous_value')

    with tf.name_scope("excercise_transformation"):
        with tf.name_scope("input"):
            a = tf.placeholder(tf.float32, shape=[None],
                               name="input_placeholder_a")

        with tf.name_scope("intermediate_layer"):
            b = tf.reduce_prod(a, name="product_b")
            c = tf.reduce_sum(a, name="sum_c")

        with tf.name_scope("output"):
            d = tf.add(b, c, name="add_d")
            output = tf.subtract(d, previous_value, name='output')
            update_preview = previous_value.assign(output)

    with tf.name_scope("summaries"):
        tf.summary.scalar('output', output)
        tf.summary.scalar('product of inputs', b)
        tf.summary.scalar('sum of intputs', c)

    with tf.name_scope("global_ops"):
        init = tf.global_variables_initializer();
        merged_summaries = tf.summary.merge_all();

sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter('./my_graph', graph)
sess.run(init)

def run_graph(input_tensor):
    feed_dict = {a: input_tensor}
    output, summary, step = sess.run([update_preview, merged_summaries, increment_step], feed_dict=feed_dict)
    writer.add_summary(summary, global_step=step)

run_graph([2, 8])
run_graph([3, 1, 3, 3])
run_graph([8])
run_graph([1, 2, 3])
run_graph([11, 4])
run_graph([4, 1])
run_graph([7, 3, 1])
run_graph([6, 3])
run_graph([0, 2])
run_graph([4, 5, 6])

sess.close()

