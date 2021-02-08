import numpy as np

class TaskCluster():
    """
    Object for a cluster of tasks containing synthetic training and testing data.

    Methods:
    - generate_parameters():
        This method takes a single parameter from the set provided for the cluster and 
        gives back a randomly generated set for this parameter for each of the tasks 
        defined in this cluster.
    """
    def __init__(self, cluster_center, std_dev, num_tasks):
        # Assign the parameters defining the cluster.
        self._cluster_center = cluster_center
        self._num_tasks = num_tasks
        self._std_dev = std_dev

        # Generate the parameters for the tasks at initialisation of the cluster.
        self._parameters = []
        for center in cluster_center:
            self._parameters.append(self.generate_parameters(center, std_dev, num_tasks))

    def generate_parameters(self, cluster_center, std_dev, num_tasks):
        """
        Method to generate and return a random subset of a parameter given by the input center.

        Inputs:
        - cluster_center: Float containing the center for one parameter defining the center 
            for this cluster.
        - std_dev: Float containing the standard deviation used to randomly generate the cluster centers.
        - num_tasks: The number of tasks that need their own parameters generated.

        Outputs:
        - y: Array containing a set of parameters generated from a normal distribution.
        """
        return np.random.normal(cluster_center, std_dev, num_tasks)

    def __len__(self):
        return len(self._parameters)

