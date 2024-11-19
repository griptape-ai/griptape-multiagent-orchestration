from abc import ABC, abstractmethod
from pathlib import Path
from attrs import define, field


@define()
class BaseParser(ABC):
    file_path: Path = field()

    @abstractmethod
    def parse(self) -> dict: ...
