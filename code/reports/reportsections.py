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
        return self.header + self.body

class ReportConfig(ReportSection):
    def __init__(self, path, config_file):
        self._header = "Config"
        self._config = config_file
        self._path = path

    @property
    def body(self):
        string = json.dumps(self._config, indent=True)
        return "\n" + string + "\n"


# class ReportExperiment(ReportSection):



class ReportTraining(ReportSection):
    def __init__(self, train_losses, valid_losses):
        self._header = "Training"
        self._train_losses = train_losses
        self._valid_losses = valid_losses

    @property
    def body(self):
        self._body = "#### Training losses \n"
        self._body += str(self._train_losses) + "\n"

        self._body += "#### Validation losses \n"
        self._body += str(self._valid_losses)
        return "\n" + self._body + "\n"


class ReportModel(ReportSection):
    def __init__(self, model):
        self._header = "Model"
        self._model = model

    @property
    def body(self):
        self._body = self._model.name + "\n"
        self._body += str(self._model)
        return self._body

