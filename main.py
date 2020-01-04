import random

import numpy as np
import config as c
from drawing import strokes
from optimization import generate_n_optimizations
from sp_model import make_model
from trainingqueue import TrainingQueue
import matplotlib.pyplot as plt


def append_second(to, what):
    for _, to_add in what:
        to.append(to_add)


if __name__ == "__main__":
    training = []
    low_history = []
    high_history = []
    mid_history = []
    model = make_model()
    tq = TrainingQueue()
    n = 0
    while True:
        print("Iteration nb: {}".format(n + 1))
        n += 1

        generated = list(np.random.uniform(0, 1, 16) for _ in range(c.n_random_samples))
        optimized = generate_n_optimizations(c.n_opt_samples, model)
        samples = generated + optimized + low_history + mid_history + high_history

        low_history = []
        high_history = []
        mid_history = []

        samples = np.array(samples, dtype='float64')
        canvases = np.zeros((samples.shape[0], 64, 64, 3))

        for i in range(samples.shape[0]):
            strokes(canvases[i], samples[i] * 64)

        evaluations = model.predict(canvases).flatten()

        evaluated_canvases = list(zip(evaluations, canvases))
        evaluated_samples = list(zip(evaluations, samples))

        evaluated_canvases = sorted(evaluated_canvases, key=lambda val: val[0])
        evaluated_samples = sorted(evaluated_samples, key=lambda val: val[0])

        plt.imshow(evaluated_canvases[-1][1])
        plt.show()

        append_second(low_history, evaluated_samples[:c.n_low_historic_samples])
        append_second(high_history, evaluated_samples[-c.n_top_historic_samples:])

        evaluated_samples = sorted(evaluated_samples, key=lambda val: val[0] ** 2)
        evaluated_canvases = sorted(evaluated_canvases, key=lambda val: val[0] ** 2)

        append_second(mid_history, evaluated_samples[:c.n_mid_historic_samples])

        for (_, left), (_, right) in \
                zip(evaluated_canvases[:c.n_mid_pairs:2], evaluated_canvases[1:c.n_mid_pairs:2]):
            tq.push_image_pair(left, right)

        evaluated_canvases = sorted(evaluated_canvases, key=lambda val: random.random())

        for (_, left), (_, right) in \
                zip(evaluated_canvases[:c.n_mid_pairs:2], evaluated_canvases[1:c.n_mid_pairs:2]):
            tq.push_image_pair(left, right)

        tq.await_data(30)

        input_data, output_data = tq.queue_to_training_data()

        model.fit(input_data, output_data, epochs=10)


