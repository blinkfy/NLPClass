"""Read-only arithmetic traces from frozen tokens, IDF, vectors and LSA bases."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits

from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.data.course import file_sha256


def term_counts(tokens, ngram_range):
    return Counter(
        " ".join(tokens[index : index + width])
        for width in range(ngram_range[0], ngram_range[1] + 1)
        for index in range(len(tokens) - width + 1)
    )


def tfidf_trace(counts, row, terms, idf, df, config):
    columns = row.indices
    frequencies = np.array([counts[terms[column]] for column in columns], dtype=float)
    if np.any(frequencies <= 0):
        raise DataValidationError("Frozen nonzero term is absent from token counts")
    tf = 1 + np.log(frequencies) if config["sublinear_tf"] else frequencies
    raw = tf * idf[columns]
    norm = float(np.linalg.norm(raw))
    if not norm or not np.allclose(raw / norm, row.data, atol=1e-10, rtol=1e-9):
        raise DataValidationError("TF-IDF arithmetic does not match frozen word vectors")
    selected = sorted(range(len(columns)), key=lambda i: (-row.data[i], terms[columns[i]]))[:5]
    return {
        "norm": norm,
        "nonzero_terms": len(columns),
        "rows": [
            {
                "term": terms[columns[i]],
                "count": int(frequencies[i]),
                "df": int(df[columns[i]]),
                "tf": float(tf[i]),
                "idf": float(idf[columns[i]]),
                "raw_weight": float(raw[i]),
                "weight": float(row.data[i]),
            }
            for i in selected
        ],
    }


def lsa_trace(row, components, actual, terms):
    with threadpool_limits(limits=1):
        latent = np.asarray(row @ components.T).ravel()
    norm = float(np.linalg.norm(latent))
    if not norm:
        raise DataValidationError("Frozen LSA maps to a zero vector")
    normalized = latent / norm
    if not norm or not np.allclose(normalized, actual, atol=1e-10, rtol=1e-9):
        raise DataValidationError("LSA arithmetic does not match frozen clustering vectors")
    products = row.data * components[0, row.indices]
    selected = sorted(range(len(products)), key=lambda i: -abs(products[i]))[:5]
    return {
        "dimension": 1,
        "rows": [
            {
                "term": terms[row.indices[i]],
                "weight": float(row.data[i]),
                "loading": float(components[0, row.indices[i]]),
                "product": float(products[i]),
            }
            for i in selected
        ],
        "remaining_sum": float(latent[0] - sum(products[i] for i in selected)),
        "remaining_terms": len(products) - len(selected),
        "raw_value": float(latent[0]),
        "norm": norm,
        "raw_preview": latent[:3].tolist(),
        "remaining_squared": float(np.square(latent[3:]).sum()),
        "normalized_preview": normalized[:3].tolist(),
    }


def save_calculation_cache(settings, run_id):
    from nlplab.services.clustering_service import ClusteringService

    service = ClusteringService(settings)
    directory = service.store.resolve(run_id)
    service.verify_artifacts(directory.name)
    model = settings.models / "exp2" / directory.name
    sources = {}

    def track(path):
        sources[str(path.relative_to(settings.root))] = file_sha256(path)
        return path

    def read(path):
        return json.loads(track(path).read_text(encoding="utf-8"))

    ids = read(directory / "input_ids.json")
    tokens = [
        json.loads(line)
        for line in track(directory / "tokens.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if [item["id"] for item in tokens] != ids:
        raise DataValidationError("Calculation token/ID order mismatch")
    vocabulary = read(model / "tfidf/vocabulary.json")
    config = vocabulary["config"]
    config.setdefault("ngram_range", [1, 1])
    terms = [term for term, _ in sorted(vocabulary["vocabulary"].items(), key=lambda pair: pair[1])]
    idf = np.load(track(model / "tfidf/idf.npy"), allow_pickle=False)
    counts = [term_counts(item["tokens"], config["ngram_range"]) for item in tokens]
    corpus_df = Counter(term for item in counts for term in item)
    df = np.array([corpus_df[term] for term in terms])
    smooth = int(config["smooth_idf"])
    if not np.allclose(np.log((len(ids) + smooth) / (df + smooth)) + 1, idf):
        raise DataValidationError("Corpus document frequency and frozen IDF disagree")
    has_lsa = (model / "lsa/lsa.npz").is_file()
    words = load_npz(track(model / ("word_vectors.npz" if has_lsa else "document_vectors.npz")))
    if has_lsa:
        with np.load(track(model / "lsa/lsa.npz"), allow_pickle=False) as data:
            components = data["components"]
        actual = np.load(track(model / "document_vectors.npy"), allow_pickle=False)
    documents = {}
    for index, document_id in enumerate(ids):
        row = words.getrow(index)
        item = {"tfidf": tfidf_trace(counts[index], row, terms, idf, df, config)}
        if has_lsa:
            item["lsa"] = lsa_trace(row, components, actual[index], terms)
        documents[document_id] = item
    value = {
        "run_id": directory.name,
        "document_count": len(ids),
        "config": config,
        "documents": documents,
        "provenance": {
            "source_files": sources,
            "builder_sha256": file_sha256(Path(__file__)),
            "formal_fit_executed": False,
            "labels_used": False,
            "method": "frozen arithmetic reconstruction; not fit",
        },
    }
    target = settings.outputs / "exp2/visualizations" / directory.name / "calculation.json"
    if target.exists():
        raise DataValidationError("Calculation cache already exists; refusing to overwrite")
    write_json(target, value)
    write_json(target.with_name("calculation_manifest.json"), {"sha256": file_sha256(target)})
    service.verify_artifacts(directory.name)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(save_calculation_cache(Settings(args.root), args.run_id))
