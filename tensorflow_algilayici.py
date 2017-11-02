# coding=utf-8
""" TensorFlow: Algilayici """

import tensorflow as tf

sess = tf.InteractiveSession()

girdi_1 = tf.constant(5.0)
girdi_2 = tf.constant(6.0)

agirlik_1 = tf.constant(-1.0)
agirlik_2 = tf.constant(0.8)

toplam_1 = tf.multiply(girdi_1, agirlik_1)
toplam_2 = tf.multiply(girdi_2, agirlik_2)

agirlikli_toplam = tf.add(toplam_1, toplam_2)

toplam = agirlikli_toplam.eval()

aktivasyon_fonksiyonu = (lambda x: 1 if x > 0 else -1)

cikti = aktivasyon_fonksiyonu(toplam)

print cikti
