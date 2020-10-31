from linear_generator import LinearData
from clusters import TaskCluster

rule_dict = {
    "linear": LinearData
}

cluster_dict = {
    "cluster": TaskCluster
}

class SyntheticData():
    def __init__(self, clusters):
        self._clusters = []

        for cluster_info in clusters:
            self._clusters.append(TaskCluster(**cluster_info))

    
    
    