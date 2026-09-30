"""复用 NLPService 的 NER，生成可追溯的实体与关系结果。"""

from nlplab.core.schema import Entity
from nlplab.information_extraction import extract_relations
from nlplab.services.nlp_service import NLPService


class InformationExtractionService:
    def __init__(self, nlp: NLPService) -> None:
        self.nlp = nlp

    def analyze(self, text: str) -> dict:
        ner = self.nlp.analyze(text, ["ner"])["ner"]
        if ner["status"] != "ok":
            return {
                "text": text,
                "status": "unavailable",
                "ner_model": ner["model"],
                "error": ner["error"],
                "entities": [],
                "relations": [],
            }
        entities = [Entity(**item) for item in ner["entities"]]
        return {
            "text": text,
            "status": "ok",
            "ner_model": ner["model"],
            "entities": ner["entities"],
            "relations": extract_relations(text, entities),
            "method": "NER + explicit typed trigger rules",
        }
