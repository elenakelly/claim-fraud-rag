"""Retrieval of similar past cases (the "R" in RAG).

Uses TF-IDF vectors by default so it runs offline and in CI with no model
downloads. Swap in dense embeddings (e.g. Azure OpenAI or sentence-transformers)
by implementing the same `search` method.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_CASES = Path(__file__).resolve().parent.parent / "data" / "past_cases.jsonl"


@dataclass
class Case:
    case_id: str
    label: str
    description: str


def load_cases(path: Path = DEFAULT_CASES) -> list[Case]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                cases.append(Case(row["case_id"], row["label"], row["description"]))
    return cases


class CaseRetriever:
    def __init__(self, cases: list[Case]):
        if not cases:
            raise ValueError("The case base is empty")
        self.cases = cases
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(c.description for c in cases)

    def search(self, text: str, k: int = 3) -> list[tuple[Case, float]]:
        query = self.vectorizer.transform([text])
        scores = cosine_similarity(query, self.matrix)[0]
        ranked = scores.argsort()[::-1][:k]
        return [(self.cases[i], float(scores[i])) for i in ranked if scores[i] > 0]
