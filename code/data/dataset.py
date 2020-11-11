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
        self._num_tasks = parameter_args["num_tasks"]
        self._task_type = parameter_args["task"].lower()

        self.generate_parameters(parameter_args)

    def generate_parameters(self, parameter_args):
        self._parameters = []
        if parameter_args["centers_or_ranges"] == "centers":
            for parameter, std_dev in zip(parameter_args["params"], parameter_args["std_dev"]):
                self._parameters.append(np.random.normal(parameter, std_dev, self._num_tasks))

        elif parameter_args["centers_or_ranges"] == "ranges":
            for parameter in parameter_args["params"]:
                self._parameters.append(np.random.uniform(parameter[0], parameter[1], self._num_tasks))

        else:
            print("Please choose either 'centers' or 'ranges' as an input")
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
