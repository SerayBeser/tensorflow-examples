# coding=utf-8
""" TensorFlow: Lineer Regresyon"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xlrd

import os

os.system("tensorboard --logdir=logs --port 6009")

# Adim 0: veri setimizin path'i
DATA_FILE = 'fire_theft.xls'

# Adim 1: .xls dosyasini okuyalim.
book = xlrd.open_workbook(filename=DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(sheetx=0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
fires_number = []
thefts_number = []
for i in data:
    fires_number.append(i[0])
    thefts_number.append(i[1])
n_samples = sheet.nrows - 1

# Adim 2: Girdimiz X ve etiketimiz Y icin placeholders olusturalim.
# X = yangin sayisi
# Y = hirsizlik sayisi
X = tf.placeholder(tf.float64, name='X')
Y = tf.placeholder(tf.float64, name='Y')

# Adim 3: Agirlik ve onyargiyi 0' esitleyelim.
w = tf.Variable(0.0, name='weights_1')
b = tf.Variable(0.0, name='bias')

w = tf.cast(w, tf.float64)
b = tf.cast(b, tf.float64)

# Adim 4: Modelimizi olusturalim.
# Yanginlardan hirsizliklari tahmin edecegiz.
Y_predicted = X * w + b

# Adim 5: Kayip Fonksiyonu olarak kare hatasini kullanalim.
loss = tf.square(Y - Y_predicted, name='loss')

# Adim 6: Dereceli azalmayi kullanarak, ogrenme oranimizin kaybini minimize edelim.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    # Adim 7: Gerekli degerlerimi arayalim.
    sess.run(tf.global_variables_initializer())

    # Adim 8: Modelimizi egitelim.
    for i in range(100):
        for x, y in data:
            sess.run(optimizer, feed_dict={X: x, Y: y})

    # Adim 9: w ve b nin sonucu
    w_value, b_value = sess.run([w, b])

    # Adim 10: Grafigin Cizelim.
    fig, ax = plt.subplots()
    predicted_thefths_number = []
    for i in fires_number:
        predicted_thefths_number.append(i * w_value + b_value)

    ax.scatter(fires_number, thefts_number, marker='o', c='blue', label='Real Data')
    ax.plot(fires_number, predicted_thefths_number, c='red', label='Predicted Data')
    ax.set_xlabel('Fires')
    ax.set_ylabel('Thefts')
    ax.legend()

    plt.show()

    # Adim 11: TensorBoard'da goruntuleyebilmek icin kaydedelim.
    writer = tf.summary.FileWriter('./logs', sess.graph)
    writer.close()

    sess.close()
