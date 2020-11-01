import numpy as np

class TaskCluster():
    def __init__(self, cluster_center, std_dev, num_tasks):
        self._cluster_center = cluster_center
        self._num_tasks = num_tasks
        self._std_dev = std_dev

        self._parameters = []
        for center in cluster_center:
            self._parameters.append(self.generate_parameters(center, std_dev, num_tasks))

    def generate_parameters(self, cluster_center, std_dev, num_tasks):
        return np.random.normal(cluster_center, std_dev, num_tasks)

    

    def __len__(self):
        return len(self._parameters)

