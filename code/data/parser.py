

class Parser():
    def __init__(self, config):
        self._config_args = config

    @property
    def experiment_args(self):
        return self._config_args["Experiment"]

    @property
    def dataset_args(self):
        return self._config_args["Dataset"]

    @property
    def model_args(self):
        return self._config_args["Model"]

    @property
    def maml_args(self):
        return self._config_args["MAML"]

    
    def parse(self):
        return self.experiment_args, self.dataset_args, self.model_args, self.maml_args