

class Parser():
    def __init__(self, config):
        self._config_args = config

    @property
    def dataset_args(self):
        return self._config_args["data"]
    
    def parse(self):
        return self.data_args