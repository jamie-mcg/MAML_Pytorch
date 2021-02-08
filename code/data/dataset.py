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
    """
    Object for the synthetic dataset.

    - generate_parameters(): 
        Method to produce a set of parameters from either a given set of parameter centers 
        or a given range of parameter ranges.
    """
    def __init__(self, parameter_args, transform=None):
        # Assign parameters for the dataset.
        self._transform = transform
        self._num_tasks = parameter_args["num_tasks"]
        self._task_type = parameter_args["task"].lower()

        # Generate the tasks at initialisation.
        self.generate_parameters(parameter_args)

    def generate_parameters(self, parameter_args):
        """
        Method to generate and return a random subset of a parameter given by the input center.

        Inputs:
        - parameter_args: Dictionary containing the arguments used to produce the parameters 
          to define the tasks in the dataset.
        """
        self._parameters = []
        # Check if we have centers or ranges of parameters.
        if parameter_args["centers_or_ranges"] == "centers":
            # Iterate through the necessary parameters needed to define a task and 
            # assign these to a list.
            for parameter, std_dev in zip(parameter_args["params"], parameter_args["std_dev"]):
                self._parameters.append(np.random.normal(parameter, std_dev, self._num_tasks))

        elif parameter_args["centers_or_ranges"] == "ranges":
            # Iterate through the necessary parameters needed to define a task and 
            # assign these to a list.
            for parameter in parameter_args["params"]:
                self._parameters.append(np.random.uniform(parameter[0], parameter[1], self._num_tasks))
        else:
            print("Please choose either 'centers' or 'ranges' as an input")

        # Transform to an array.
        self._parameters = np.array(self._parameters)

    def __len__(self):
        return self._num_tasks

    def __getitem__(self, idx):
        # Define a task object and give it a set of already defined parameters.
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
