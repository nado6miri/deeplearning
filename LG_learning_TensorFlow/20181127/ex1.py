import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
hello = tf.constant("Hello Tensorflow")
print(hello)

a = tf.constant(10)
b = tf.constant(32)
c = tf.constant(2)

d = a*b+c
d = tf.multiply(a, b)
e = tf.add(d,c)
print("\nd={}".format(d))

sess = tf.Session()
print(sess.run(hello))
print(sess.run(d))
sess.close()