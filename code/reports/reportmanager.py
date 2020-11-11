from abc import ABC, abstractstaticmethod
from .reportsections import *

import os
from datetime import datetime

SECTIONS = {
    "config": ReportConfig,
    "model": ReportModel,
    "training": ReportTraining
}

class ReportManager():
    def __init__(self, path):
        self._path = path
        self._sections = []

        dt = datetime.now().strftime("%m%d%Y_%H%M%S")
        self._path = os.path.join(self._path, dt)

        os.mkdir(self._path, mode=0o777)

    def add_section(self, section, **kwargs):
        new_section = SECTIONS[section.lower()](path=self._path, **kwargs)
        self._sections.append(new_section)

    def save(self):
        with open(os.path.join(self._path, "report.md"), "a") as report_file:
            for section in self._sections:
                print(str(section))
                report_file.write(str(section))
        self._sections = []
