from typing import Protocol

from nlplab.core.schema import Entity


class NERRecognizer(Protocol):
    name: str

    def recognize(self, text: str) -> list[Entity]: ...
