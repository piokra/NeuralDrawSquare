import numpy as np


def stroke(canvas, sx, sy, ex, ey):
    w, h, d = canvas.shape
    dx = ex - sx
    dy = ey - sy
    steps = 2 * np.sqrt(dx ** 2 + dy ** 2)
    xs = sx + dx * np.linspace(0, 1, steps, False)
    ys = sy + dy * np.linspace(0, 1, steps, False)

    xs = xs.astype('uint8')
    ys = ys.astype('uint8')

    xs[xs >= w] = w - 1
    ys[ys >= h] = h - 1

    canvas[xs, ys, :] = 1


def strokes(canvas, coords):
    d = len(coords)
    assert d % 4 == 0

    coords = np.reshape(coords, (d // 4, 4))
    for sx, sy, ex, ey in coords:
        stroke(canvas, sx, sy, ex, ey)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sp_model import make_model

    model = make_model()

    canvas = np.zeros((64, 64, 3))

    coords = np.random.uniform(0, 64, 16)
    strokes(canvas, coords)

    canvas_data = canvas.T.reshape((1, 64, 64, 3))
    print(model.predict(canvas_data))

    plt.imshow(canvas)
    plt.show()
