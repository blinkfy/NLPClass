"""显式字符二元组：非负 Count → 训练标签 χ² → TF-IDF → L2。"""

import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2

from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json

ANALYZER_RULE = {
    "unicode_normalization": "NFKC",
    "lowercase": True,
    "ngram_length": 2,
    "content": "Unicode Han ideographs, ASCII a-z, Unicode decimal digits (Nd)",
    "boundaries": "all other codepoints; no crossing",
    "stopwords": [],
    "overlapping": True,
    "uses_tokenizer": False,
}
COUNT_CONFIG = {"min_df": 2, "max_df": 1.0, "max_features": None}
TFIDF_CONFIG = {"sublinear_tf": True, "smooth_idf": True, "use_idf": True, "norm": "l2"}


def content_character(char):
    return (
        "a" <= char <= "z"
        or unicodedata.category(char) == "Nd"
        or unicodedata.name(char, "").startswith(
            ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")
        )
    )


def char_bigrams(text):
    if not isinstance(text, str):
        raise DataValidationError("字符 analyzer 只接受正文字符串")
    result, previous = [], None
    for char in unicodedata.normalize("NFKC", text).lower():
        if content_character(char):
            if previous is not None:
                result.append(previous + char)
            previous = char
        else:
            previous = None
    return result


def sparse_stats(matrix):
    return {
        "shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
        "sparsity": float(1 - matrix.nnz / np.prod(matrix.shape)),
        "zero_vector_count": int(np.sum(matrix.getnnz(axis=1) == 0)),
    }


class StableChi2Selector(SelectKBest):
    """同 χ² score 时取词表字典序较前项；输出列保持原始词表顺序。"""

    def __init__(self, k="all"):
        super().__init__(score_func=chi2, k=k)

    def _get_support_mask(self):
        if self.k == "all":
            return np.ones(len(self.scores_), dtype=bool)
        scores = np.nan_to_num(self.scores_, nan=-np.inf)
        ranked = np.lexsort((np.arange(len(scores)), -scores))
        mask = np.zeros(len(scores), dtype=bool)
        mask[ranked[: self.k]] = True
        return mask


class CharacterCounts:
    def __init__(self):
        self.vectorizer = CountVectorizer(
            analyzer=char_bigrams, lowercase=False, dtype=np.int64, **COUNT_CONFIG
        )

    def fit_transform(self, texts):
        try:
            matrix = self.vectorizer.fit_transform(texts).tocsr()
        except ValueError as error:
            raise DataValidationError(f"无可用字符 bigram: {error}") from error
        self.df = np.asarray(matrix.getnnz(axis=0)).ravel()
        self.observed_df = Counter(feature for text in texts for feature in set(char_bigrams(text)))
        self.training_stats = {
            "raw_count_vocabulary_size": matrix.shape[1],
            "observed_bigram_types_before_min_df": len(self.observed_df),
            "singleton_types_removed": sum(df == 1 for df in self.observed_df.values()),
            "df_histogram_before_min_df": dict(sorted(Counter(self.observed_df.values()).items())),
            "max_retained_df": int(self.df.max()),
            "count": sparse_stats(matrix),
        }
        return matrix

    def transform(self, texts):
        return self.vectorizer.transform(texts).tocsr()

    @property
    def terms(self):
        return self.vectorizer.get_feature_names_out()


class CharacterChi2TFIDF:
    def __init__(self, k="all"):
        self.k = k

    def fit_from_counts(self, counts, labels, raw):
        if self.k != "all" and (not isinstance(self.k, int) or not 1 <= self.k <= counts.shape[1]):
            raise DataValidationError("k 超过可用特征数；不能静默裁剪")
        if np.any(counts.data < 0):
            raise DataValidationError("χ² 仅接受非负 Count")
        self.raw = raw
        self.selector = StableChi2Selector(self.k)
        selected = self.selector.fit_transform(counts, labels).tocsr()
        self.selected_indices = self.selector.get_support(indices=True)
        self.chi2_scores = self.selector.scores_.copy()
        self.tfidf = TfidfTransformer(**TFIDF_CONFIG)
        vectors = self.tfidf.fit_transform(selected).tocsr()
        self.fit_stats = {
            "raw": raw.training_stats,
            "selected_counts": sparse_stats(selected),
            "tfidf": sparse_stats(vectors),
        }
        return vectors

    def fit_transform(self, texts, labels):
        raw = CharacterCounts()
        counts = raw.fit_transform(texts)
        return self.fit_from_counts(counts, labels, raw)

    def transform_counts(self, counts):
        selected = self.selector.transform(counts).tocsr()
        return self.tfidf.transform(selected).tocsr(), {
            "before_selection": sparse_stats(counts),
            "after_selection": sparse_stats(selected),
        }

    def transform(self, texts):
        return self.transform_counts(self.raw.transform(texts))[0]

    @property
    def terms(self):
        return self.raw.terms[self.selected_indices]

    @property
    def selected_scores(self):
        return self.chi2_scores[self.selected_indices]

    def top_chi2(self, n=30):
        terms = self.raw.terms
        order = sorted(
            range(len(self.chi2_scores)), key=lambda i: (-self.chi2_scores[i], terms[i])
        )[:n]
        selected = set(self.selected_indices.tolist())
        return [
            {
                "term": str(terms[i]),
                "chi2_score": float(self.chi2_scores[i]),
                "df": int(self.raw.df[i]),
                "selected": i in selected,
            }
            for i in order
        ]

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=False)
        write_json(
            path / "representation.json",
            {
                "k": self.k,
                "analyzer": ANALYZER_RULE,
                "count_config": COUNT_CONFIG,
                "tfidf_config": TFIDF_CONFIG,
                "selector_tie_break": "score descending, term ascending",
                "vocabulary": {term: int(i) for term, i in self.raw.vectorizer.vocabulary_.items()},
                "fit_stats": self.fit_stats,
            },
        )
        np.savez_compressed(
            path / "parameters.npz",
            selected_indices=self.selected_indices,
            chi2_scores=self.chi2_scores,
            idf=self.tfidf.idf_,
            df=self.raw.df,
        )
        write_json(path / "observed_df.json", dict(sorted(self.raw.observed_df.items())))
        write_json(
            path / "selected_features.json",
            [
                {"column": i, "term": str(term), "chi2_score": float(score)}
                for i, (term, score) in enumerate(zip(self.terms, self.selected_scores))
            ],
        )

    @classmethod
    def load(cls, path: Path):
        meta = json.loads((path / "representation.json").read_text("utf-8"))
        result = cls(meta["k"])
        result.raw = CharacterCounts()
        result.raw.vectorizer.set_params(vocabulary=meta["vocabulary"])
        with np.load(path / "parameters.npz", allow_pickle=False) as arrays:
            result.selected_indices = arrays["selected_indices"]
            result.chi2_scores = arrays["chi2_scores"]
            result.selector = StableChi2Selector(meta["k"])
            result.selector.scores_ = result.chi2_scores
            result.selector.n_features_in_ = len(result.chi2_scores)
            result.raw.df = arrays["df"]
            result.tfidf = TfidfTransformer(**meta["tfidf_config"])
            result.tfidf.idf_ = arrays["idf"]
        result.fit_stats = meta["fit_stats"]
        result.raw.training_stats = meta["fit_stats"]["raw"]
        result.raw.observed_df = Counter(json.loads((path / "observed_df.json").read_text("utf-8")))
        return result


def feature_coverage(features, vocabulary):
    """抽取候选的 occurrence / distinct-type 中，命中给定 fitted vocabulary 的比例。"""
    unique = set(features)
    return {
        "candidate_occurrences": len(features),
        "hit_occurrences": sum(f in vocabulary for f in features),
        "occurrence_ratio": sum(f in vocabulary for f in features) / len(features)
        if features
        else None,
        "candidate_unique_types": len(unique),
        "hit_unique_types": len(unique & vocabulary),
        "unique_type_ratio": len(unique & vocabulary) / len(unique) if unique else None,
    }
