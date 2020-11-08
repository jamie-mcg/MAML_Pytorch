from abc import ABC, abstractstaticmethod
from .reportsections import ReportConfig

SECTIONS = {
    "config": ReportConfig
}

class ReportManager():
    def __init__(self, path):
        self._path = path
        self._sections = []

    def add_section(self, section, config):
        new_section = SECTIONS[section.lower()](config)
        self._sections.append(new_section)

    def save(self):
        # Write to the report and delete current report
        for section in self._sections:
            print(section)
        self._sections = []
