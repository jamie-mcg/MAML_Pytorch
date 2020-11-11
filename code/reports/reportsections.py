from abc import ABC, abstractstaticmethod
import json

class ReportSection(ABC):
    @property
    def header(self):
        return "## " + self._header

    @property
    @abstractstaticmethod
    def body(self):
        pass

    def __str__(self):
        return self.header + "\n" + self.body + "\n"


class ReportConfig(ReportSection):
    def __init__(self, path, config_file):
        self._header = "Config"
        self._config = config_file
        self._path = path

    @property
    def body(self):
        string = json.dumps(self._config, indent=True)
        return "\n" + string + "\n"


class ReportModel(ReportSection):
    def __init__(self, path, model, model_args):
        self._header = "Model"
        self._model = model
        self._model_args = model_args

    @property
    def body(self):
        self._body = "#### " + self._model.name + "\n"
        self._body += str(self._model)
        return self._body + "\n"


class ReportTraining(ReportSection):
    def __init__(self, path, model):
        self._header = "Training"
        self._path = path

        self._meta_iterations = model.iterations
        self._train_losses = model.training_losses
        self._valid_losses = model.validation_losses

    @property
    def body(self):
        self._body = "#### Meta iterations \n"
        self._body += str(self._meta_iterations) + "\n"

        self._body += "#### Training losses \n"
        self._body += str(self._train_losses) + "\n"

        self._body += "#### Validation losses \n"
        self._body += str(self._valid_losses)
        return self._body + "\n"
