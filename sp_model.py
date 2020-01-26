from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from random_samples import generate_n_squarelikes, generate_n_random


def make_model() -> models.Sequential:
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='tanh'))

    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['accuracy'])
    return model


def pretrained_model(n_good=10000, n_bad=10000, force_new=False) -> models.Sequential:
    if not force_new:
        try:
            return load_model('pretrained.h5')
        except BaseException as e:
            print(e)

    model = make_model()
    data = np.concatenate((generate_n_squarelikes(n_good), generate_n_random(n_bad)))
    print(data.shape)
    labels = [1] * n_good + [-1] * n_bad

    model.fit(data, labels, epochs=10)
    model.save("pretrained.h5")
    return model


if __name__ == "__main__":
    from optimization import generate_n_optimizations
    from drawing import strokes

    model = pretrained_model(100000, 100000)
    good = generate_n_random(1000)
    res = model.predict(good)
    opt = generate_n_optimizations(1, model)

    import matplotlib.pyplot as plt
    for res, canvas in zip(res, good):
        print(res)
        if res == 1:
            plt.imshow(canvas)
            plt.show()
