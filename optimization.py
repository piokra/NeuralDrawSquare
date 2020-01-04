import numpy as np

from drawing import stroke
from sp_model import make_model
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt


def square_drawing_loss(coords, model):
    d = coords.shape[0]
    coords = coords * 64
    assert d % 4 == 0

    coords = coords.reshape((d // 4, 4))
    canvas = np.zeros((64, 64, 3))
    for i in range(d // 4):
        stroke(canvas, *coords[i])

    canvas.shape = (1, 64, 64, 3)
    return -model.predict(canvas).flatten()[0]


def make_bounds(strokes):
    return ((0, 1),) * strokes * 4


def generate_n_optimizations(n, model):
    ret = []
    for _ in range(n):
        initial_guess = np.random.uniform(0, 1, 16)
        result = basinhopping(square_drawing_loss, initial_guess,
                              minimizer_kwargs={'args': model, 'bounds': make_bounds(4)})
        x = result.x
        ret.append(x)
    return ret


if __name__ == "__main__":

    model = make_model()
    for _ in range(10):
        result = basinhopping(square_drawing_loss, (np.random.uniform(0, 1, 16)),
                              minimizer_kwargs={'args': model, 'bounds': make_bounds(4)})

        s = result.x * 64
        canvas = np.zeros((64, 64, 3))
        print(result.fun)
        for i in range(4):
            stroke(canvas, *s[(4 * i):(4 * i + 4)])
        plt.imshow(canvas)
        plt.show()
