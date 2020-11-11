import numpy as np

import torch

from torch.utils.data import Dataset
from .linear_generator import LinearTask
from .sinusoidal_generator import SinusoidalTask

task_dict = {
    "linear": LinearTask,
    "sinusoidal": SinusoidalTask
}

class TaskDataset(Dataset):
    def __init__(self, parameter_args, transform=None):
        self._transform = transform
        self._std_dev = parameter_args["std_dev"]
        self._num_tasks = parameter_args["num_tasks"]
        self._task_type = parameter_args["task"].lower()

        self.generate_parameters(parameter_args)

    def generate_parameters(self, parameter_args):
        self._parameters = []
        for parameter in parameter_args["param_centers"]:
            self._parameters.append(np.random.uniform(parameter, self._std_dev, self._num_tasks))
        self._parameters = np.array(self._parameters)

    def __len__(self):
        return self._num_tasks

    def __getitem__(self, idx):
        task = task_dict[self._task_type](parameters=self._parameters[:,idx])

        return {
            "train": (torch.from_numpy(task.x_train).float().view(-1, 1), torch.from_numpy(task.y_train).float().view(-1, 1)),
            "test": (torch.from_numpy(task.x_test).float().view(-1, 1), torch.from_numpy(task.y_test).float().view(-1, 1))
        }


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    parameter_args = {
        "task": "linear",
        "centers": [2.0, 3.0],
        "std_dev": 0.2,
        "num_tasks": 10
    }

    dataset = TaskDataset(parameter_args)

    fig = plt.figure()

    for data in dataset:
        plt.plot(data["train"][0], data["train"][1])

    plt.show()
