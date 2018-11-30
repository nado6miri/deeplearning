import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = np.array([2,3], dtype=np.int32)
b = np.array([4,5], dtype=np.int32)
c = tf.add(a, b, name='my_add_node')


sess = tf.Session()
result = sess.run(c)
print(result)
#tf.summary.FileWriter('./my_graph', sess.graph)
sess.close()