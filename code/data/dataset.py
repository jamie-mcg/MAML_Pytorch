import numpy as np

import torch

from torch.utils.data import Dataset
from linear_generator import LinearTask

task_dict = {
    "Linear": LinearTask
}

class TaskDataset(Dataset):
    def __init__(self, parameter_args, task_args, transform=None):
        self._transform = transform
        self._std_dev = parameter_args["std_dev"]
        self._num_tasks = parameter_args["num_tasks"]

        self._task_args = task_args

        self.generate_parameters(parameters_args)

    def generate_parameters(self, parameters):
        self._parameters = []
        for parameter in parameters_args["centers"]:
            self._parameters.append(np.random.uniform(parameter, self._std_dev, self._num_tasks))

    def __len__(self):
        return len(self._num_tasks)

    def __getitem__(self, idx):
        task = task_dict[self._task_type](args=self._parameters[:,idx], **self._task_args)

        return {
            "train": (task.x_train, task.y_train),
            "test": (task.x_test, task.y_test)
