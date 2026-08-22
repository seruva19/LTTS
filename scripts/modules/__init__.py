from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LTTSModuleData:
    target: str
    id: str
    title: str
    input_struct: dict
    output_struct: dict
    binding: str


class LTTSModule(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def target(self):
        pass

    @property
    @abstractmethod
    def id(self):
        pass

    @property
    @abstractmethod
    def title(self):
        pass

    @property
    @abstractmethod
    def input_struct(self):
        pass

    @property
    @abstractmethod
    def output_struct(self):
        pass

    @property
    @abstractmethod
    def binding(self):
        pass
