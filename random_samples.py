import numpy as np

from drawing import stroke


def generate_n_squarelikes(n):
    canvases = np.zeros(shape=(n, 64, 64, 3))

    xs, ys = np.random.uniform(0.2, 0.8, size=n), np.random.uniform(0.2, 0.8, size=n)
    fx, fy = xs.copy(), ys.copy()

    thetas = np.random.uniform(0, np.pi / 2, size=n)
    for _ in range(3):

        dx, dy = np.cos(thetas), np.sin(thetas)
        ds = np.random.uniform(0.4, 0.8, size=n)
        ds *= np.amin((np.amax((-xs / dx, (1 - xs) / dx), axis=0), np.amax((-ys / dy, (1 - ys) / dy), axis=0)), axis=0)
        ex, ey = xs + dx * ds, ys + dy * ds
        ex += np.random.normal(0, 0.1, n)
        ey += np.random.normal(0, 0.1, n)

        ex, ey = np.clip(ex, 0, 1), np.clip(ey, 0, 1)

        for i in range(n):
            stroke(canvases[i], xs[i] * 64, ys[i] * 64, ex[i] * 64, ey[i] * 64)
        thetas += np.pi / 2
        xs, ys = ex, ey

    for i in range(n):
        stroke(canvases[i], xs[i] * 64, ys[i] * 64, fx[i] * 64, fy[i] * 64)
    return canvases


def generate_n_random(n):
    canvases = np.zeros(shape=(n, 64, 64, 3))
    for _ in range(4):
        sx, sy, ex, ey = np.reshape(np.random.uniform(0, 64, size=4 * n), newshape=(4, n))
        for i in range(n):
            stroke(canvases[i], sx[i], sy[i], ex[i], ey[i])
    return canvases



if __name__ == "__main__":
    canvases = generate_n_random(100)

    import matplotlib.pyplot as plt

    for i in range(4):
        plt.imshow(canvases[i])
        plt.show()
