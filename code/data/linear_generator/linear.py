import numpy as np

class LinearTask():
    def __init__(self, parameters, x_min=-5, x_max=5, num_points=5, noise_level=0):
        self._a = parameters[0]
        self._b = parameters[1]

        self._x_train = np.random.uniform(x_min, x_max, num_points)
        self._x_test = np.random.uniform(x_min, x_max, num_points)

        self._num_points = num_points
        self._noise_level = noise_level

    @property
    def x_train(self):
        return np.array([self._x_train])

    @property
    def x_test(self):
        return np.array([self._x_test])

    @property
    def y_train(self):
        noise = self._noise_level * np.random.uniform(-1, 1, self._num_points)

        return np.array([self._a * self._x_train + self._b + noise])

    @property
    def y_test(self):
        noise = self._noise_level * np.random.uniform(-1, 1, self._num_points)

        return np.array([self._a * self._x_test + self._b + noise])

