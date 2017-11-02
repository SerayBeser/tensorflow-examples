# coding=utf-8
""" TensorFlow: Basit Lineer Regresyon """

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

ogrenme_orani = 0.01
devir_sayisi = 70

# rastgele bir egitim seti olusturalim.
x_egitim = np.linspace(-1, 1, 51)
y_egitim = 2 * x_egitim + np.random.randn(*x_egitim.shape) * 0.25

# olusturdugumuz egitim setine bir goz atalim.
plt.scatter(x=x_egitim, y=y_egitim, marker='*', s=120, edgecolors='darkblue')
# plt.show()

# yer tutucularimizi ayarlayalim.
X_ekseni = tf.placeholder(tf.float32)
Y_ekseni = tf.placeholder(tf.float32)

# agirlik degiskeni
agirlik = tf.Variable(0.0, name="weights")

# modeli olusturup, hesaplayalim.
model = tf.multiply(X_ekseni, agirlik)

kayip = tf.reduce_mean(tf.square(Y_ekseni - model))

egitim = tf.train.GradientDescentOptimizer(ogrenme_orani).minimize(kayip)

oturum = tf.Session()
init = tf.global_variables_initializer()
oturum.run(init)

for devir in range(devir_sayisi):
    for (x, y) in zip(x_egitim, y_egitim):
        oturum.run(egitim, feed_dict={X_ekseni: x, Y_ekseni: y})

bulunan_agirlik = oturum.run(agirlik)

oturum.close()

# veri setimize uyan dogruyu cizelim.
plt.scatter(x_egitim, y_egitim, marker='*', s=120, edgecolors='darkblue')
y_tahmin = x_egitim * bulunan_agirlik
plt.plot(x_egitim, y_tahmin, 'magenta', lw=5)
plt.show()

