from typing import Protocol


class TextClusterer(Protocol):
    def fit_predict(self, texts: list[str]) -> list[int]: ...
