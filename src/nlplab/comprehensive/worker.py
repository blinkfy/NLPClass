"""Reuse original adapters/algorithms in existing offline runtimes."""

import argparse
import contextlib
import json
import os
import socket
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from nlplab.core.config import Settings
from nlplab.core.exceptions import NLPLabError

TOKENIZER_RUN = "20260926_130612_32a5e168_smooth"


def lexical(root, payload):
    from nlplab.ner.hanlp_adapter import HanLPRecognizer
    from nlplab.tagging.hanlp_pos import HanLPPosTagger
    text, tokens = payload["text"], payload["tokens"]
    if "".join(tokens) != text:
        raise ValueError("词项与原文不一致")
    output = {"pos": [], "entities": [], "modules": {}, "provenance": {}}
    started = perf_counter()
    try:
        tagger = HanLPPosTagger.from_settings(Settings(root))
        # Application scheduling, not a new tokenizer: split only at existing
        # HMM token boundaries, preserve all whitespace and original offsets.
        chunks, current, length = [], [], 0
        for token in tokens:
            if current and length + len(token) > 200:
                chunks.append(current)
                current, length = [], 0
            current.append(token)
            length += len(token)
        if current:
            chunks.append(current)
        offset = 0
        for chunk in chunks:
            value = "".join(chunk)
            result = tagger.tag(value, chunk)
            output["pos"].extend({**row, "start": row["start"] + offset,
                                  "end": row["end"] + offset} for row in result["items"])
            offset += len(value)
        output["modules"]["pos"] = {"status": "ok", "elapsed_ms": round((perf_counter()-started)*1000, 2)}
        output["provenance"]["pos"] = {"model": tagger.name, "setup": "artifacts/cache/pos_setup.json",
                                         "input_protocol": "HMM word tokens; token-boundary batches ≤200 chars (oversize token passed intact)",
                                         "chunk_count": len(chunks)}
    except (NLPLabError, ValueError, TypeError) as error:
        output["modules"]["pos"] = {"status": "unavailable", "error": str(error), "elapsed_ms": round((perf_counter()-started)*1000, 2)}
    started = perf_counter()
    try:
        recognizer = HanLPRecognizer.from_settings(Settings(root))
        output["entities"] = [asdict(entity) for entity in recognizer.recognize(text)]
        output["modules"]["ner"] = {"status": "ok", "elapsed_ms": round((perf_counter()-started)*1000, 2)}
        output["provenance"]["ner"] = {"model": recognizer.name,
                                         "setup": "artifacts/cache/ner_setup.json",
                                         "input_protocol": "original characters; existing native WordPiece-aware chunks",
                                         "label_map": recognizer.label_map}
    except (NLPLabError, ValueError, TypeError) as error:
        output["modules"]["ner"] = {"status": "unavailable", "error": str(error), "elapsed_ms": round((perf_counter()-started)*1000, 2)}
    return output


def custom(root, payload):
    from nlplab.clustering.experiment import execute_unsupervised
    from nlplab.data.clustering_input import ClusteringDocument
    from nlplab.services.nlp_service import NLPService
    nlp = NLPService(Settings(root))
    segment = lambda text: nlp.analyze(text, ["segmentation"], TOKENIZER_RUN)["segmentation"]["tokens"]
    k = payload["k"]
    config = {"k_values": [k], "reference_k": k, "stability_seeds": [42, 43],
              "kmeans": {"n_clusters": k, "random_state": 42},
              "preprocessing": {}, "representation": {"min_df": 1, "max_df": 1.0, "ngram_range": [1, 2]},
              "projection": {"n_components": 2, "n_iter": 7, "random_state": 42}}
    docs = [ClusteringDocument(row["id"], row["text"]) for row in payload["documents"]]
    result = execute_unsupervised(docs, segment, config)
    saved = result["results"][k]
    term_vectors = {}
    terms = result["representation"].terms
    for index, doc in enumerate(docs):
        row = result["word_vectors"].getrow(index)
        term_vectors[doc.id] = [{"term": str(terms[j]), "weight": float(weight)} for j, weight in zip(row.indices, row.data)]
    return {"term_vectors": term_vectors, "assignments": saved["assignments"], "clusters": saved["clusters"],
            "projection": result["projection"], "representation": result["stats"],
            "provenance": {"algorithm": "existing execute_unsupervised / KMeansClusterer",
                             "tokenizer_run": TOKENIZER_RUN, "config": config,
                             "temporary_analysis": True, "formal_run_written": False}}


def terms(root, payload):
    import numpy as np
    from scipy.sparse import load_npz
    from nlplab.services.clustering_service import ClusteringService
    service = ClusteringService(Settings(root))
    run_id = payload["run_id"]
    if run_id != "20260930_102533_9c28e666":
        raise ValueError("只读关键词需要冻结R2来源")
    service.verify_artifacts(run_id)
    directory = service.store.resolve(run_id)
    ids = json.loads((directory / "input_ids.json").read_text(encoding="utf-8"))
    positions = {doc_id: index for index, doc_id in enumerate(ids)}
    selected = payload["document_ids"]
    if not selected or any(doc_id not in positions for doc_id in selected):
        raise ValueError("关键词选区ID无效")
    model = root / "artifacts/models/exp2" / run_id
    vocabulary = json.loads((model / "tfidf/vocabulary.json").read_text(encoding="utf-8"))["vocabulary"]
    names = [term for term, index in sorted(vocabulary.items(), key=lambda pair: pair[1])]
    vectors = load_npz(model / "word_vectors.npz")
    weights = np.asarray(vectors[[positions[doc_id] for doc_id in selected]].mean(axis=0)).ravel()
    order = sorted(range(len(names)), key=lambda index: (-weights[index], names[index]))[:12]
    return {"document_ids": selected, "terms": [{"term": names[i], "weight": float(weights[i])} for i in order if weights[i] > 0],
            "source": "mean selected rows of frozen R2 word_vectors.npz; not classification evidence",
            "run_id": run_id, "fit_executed": False, "raw_text_read": False}


def classify(root, payload):
    """One safe frozen load, transform only; no labels, fit or evaluation."""
    from collections import Counter
    from nlplab.classification.c2 import C2LinearClassifier
    from nlplab.data.course import file_sha256
    from nlplab.features.char_chi2 import CharacterChi2TFIDF, char_bigrams, feature_coverage
    from nlplab.comprehensive.classification_application import attach_evidence
    from nlplab.comprehensive.contracts import document_id, validate_text
    run_id = payload["run_id"]
    if run_id != "20261001_133829_68ee50d7":
        raise ValueError("09分类索引只允许冻结C2")
    directory = root / "outputs/exp3" / run_id
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))["files"]
    # Verify only files loaded by this application inference, not Test results.
    for name in ("final/C2/model.json", "final/C2/parameters.npz", "final/representation/representation.json",
                 "final/representation/parameters.npz", "final/representation/observed_df.json"):
        if file_sha256(directory / name) != manifest[name]:
            raise ValueError("冻结C2推理文件指纹不符")
    representation = CharacterChi2TFIDF.load(directory / "final/representation")
    model = C2LinearClassifier.load(directory / "final/C2")
    docs = payload["documents"]
    if not docs or len(docs) > 510 or len({d["id"] for d in docs}) != len(docs):
        raise ValueError("应用分类批次ID无效")
    for doc in docs:
        validate_text(doc["text"])
    started = perf_counter()
    vectors = representation.transform([d["text"] for d in docs])
    terms = representation.terms
    columns = {str(term): i for i, term in enumerate(terms)}
    selected = representation.selected_indices
    records = []
    for i, doc in enumerate(docs):
        row = vectors.getrow(i)
        result = model.explain(row, terms)
        features = char_bigrams(doc["text"])
        counts = Counter(features)
        metadata = {}
        for feature in result["top_relative_features"]:
            term = feature["term"]
            j = columns[term]
            raw = int(selected[j])
            metadata[term] = {"column": j, "raw_column": raw, "count": counts[term],
                              "chi2_score": float(representation.chi2_scores[raw]),
                              "df": int(representation.raw.df[raw]), "idf": float(representation.tfidf.idf_[j]),
                              "retained": True}
        result.update(run_id=run_id, model_id="C2", token_count=len(features), known_feature_count=row.nnz,
                      coverage=feature_coverage(features, set(terms)),
                      notice="LinearSVC decision score / margin 不是概率；仅适用于五类新闻。")
        result = attach_evidence(doc["text"], result, metadata)
        records.append({"id": doc["id"], "text_id": document_id(doc["text"]), "classification": result})
    return {"records": records, "run_id": run_id, "model_id": "C2", "fit_executed": False,
            "ner_executed": False, "gold_labels_used": False, "formal_evaluation_executed": False,
            "elapsed_ms": round((perf_counter()-started)*1000, 2)}


def information_extraction(root, payload):
    from nlplab.ner.hanlp_adapter import HanLPRecognizer
    from nlplab.services.nlp_service import NLPService
    from nlplab.services.information_extraction_service import InformationExtractionService
    nlp = NLPService(Settings(root), ner=HanLPRecognizer.from_settings(Settings(root)))
    return InformationExtractionService(nlp).analyze(payload["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(sys.stdin.read())
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      HANLP_HOME=str(args.root / "artifacts/cache/hanlp"))
    original = socket.socket.connect
    def offline(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError("Only prepared local NLP resources are allowed")
        return original(sock, address)
    socket.socket.connect = offline
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if payload["kind"] not in ("custom", "terms", "classify"):
                import torch
                torch.set_num_threads(4)
                torch.set_num_interop_threads(1)
            function = {"lexical": lexical, "custom": custom, "information_extraction": information_extraction, "terms": terms, "classify": classify}[payload["kind"]]
            result = function(args.root.resolve(), payload)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except (NLPLabError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
