import tensorflow as tf

graph = tf.Graph()

with graph.as_default():
    with tf.name_scope("variables"):
        with tf.name_scope("input"):
            a = tf.placeholder(tf.float32, shape=[None],
                               name="input_placeholder_a")

        with tf.name_scope("intermediate_layer"):
            b = tf.reduce_prod(a, name="product_b")
            c = tf.reduce_sum(a, name="sum_c")

        with tf.name_scope("output"):
            d = tf.add(b, c, name="add_d")

sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter('./my_graph', graph)

def run_graph(input_tensor):
    feed_dict = {a: input_tensor}
    output = sess.run(d, feed_dict=feed_dict)
    print('result:',output)

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

