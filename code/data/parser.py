

class Parser():
    def __init__(self, config):
        self._config_args = config

    @property
    def experiment_args(self):
        return self._config_args["Experiment"]

    @property
    def train_dataset_args(self):
        return self._config_args["Dataset - metatrain"]

    @property
    def valid_dataset_args(self):
        return self._config_args["Dataset - metatest"]

    @property
    def model_args(self):
        return self._config_args["Model"]

    @property
    def maml_args(self):
        return self._config_args["MAML"]

    @property
    def training_args(self):
        return self._config_args["training"]
    
    def parse(self):
        return self.experiment_args, self.train_dataset_args, self.valid_dataset_args, self.model_args, self.maml_args, self.training_args