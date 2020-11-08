import numpy as np

import torch

from torch.utils.data import Dataset
from .linear_generator import LinearTask

task_dict = {
    "linear": LinearTask
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

    def __len__(self):
        return len(self._num_tasks)

    def __getitem__(self, idx):
        task = task_dict[self._task_type](a=self._parameters[0][idx], b=self._parameters[1][idx])

        return {
            "train": (task.x_train, task.y_train),
            "test": (task.x_test, task.y_test)
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
