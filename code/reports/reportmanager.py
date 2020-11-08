from abc import ABC, abstractstaticmethod

class ReportManager():
    def __init__(self, path):
        self._path = path
        self._sections = []

    def add_section(self, header, body):
        new_section = ReportSection(header, body)
        self._sections.append(new_section)

    def save(self):
        # Write to the report and delete current report
        for section in self._sections:
            print(section)
        self._sections = []


class ReportSection(ABC):
    @property
    def header(self):
        return "## " + self._header

    @property
    @abstractstaticmethod
    def body(self, text):
        pass

    def __call__(self):
        return self.header + self.body


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

