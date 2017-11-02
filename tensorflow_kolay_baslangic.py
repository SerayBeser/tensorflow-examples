# coding=utf-8
""" Tensorflowa Kolay Baslangic """
import tensorflow as tf

a = tf.constant([2], name="a")
b = tf.constant([3], name="b")

c = tf.add(a, b, name="c")
# veya
# c = a + b

session = tf.Session()

result = session.run(c)
print "Sonuc 1: ", result

session.close()

with tf.Session() as session:
    result = session.run(c)
print "Sonuc 2: ", result

with tf.Session() as session:
    result = session.run(c)
print "Sonuc 3: ", result
writer = tf.summary.FileWriter('./logs', session.graph)

writer.close()

# import os # os.system("tensorboard --logdir=logs --port 6006")
#  tensorboard –logdir=/path/logs
