# coding=utf-8
""" Python Algilayici """
import numpy as np

girdiler = np.array([5, 6])

agirliklar = np.array([-1, 0.8])

agirlikli_toplam = girdiler * agirliklar

toplam = sum(agirlikli_toplam)

aktivasyon_fonksiyonu = (lambda x: 1 if x > 0 else -1)

cikti = aktivasyon_fonksiyonu(toplam)

print cikti
