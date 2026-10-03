"""纯无监督执行路径：此模块不导入标签 loader，不接收 gold。"""

from dataclasses import replace
from itertools import combinations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from nlplab.clustering.analysis import snippet, summarize_clusters
from nlplab.clustering.kmeans import KMeansClusterer, KMeansConfig
from nlplab.core.exceptions import DataValidationError
from nlplab.data.clustering_input import ClusteringDocument
from nlplab.evaluation.clustering import internal_metrics
from nlplab.features.lsa import LSAConfig, TextLSA
from nlplab.features.text_tfidf import (
    ChinesePreprocessor,
    PreprocessingConfig,
    TextTFIDF,
    TFIDFConfig,
)


def execute_unsupervised(
    documents, segment, config, progress=lambda message: None, *, tokenized=None
):
    if not documents or any(type(doc) is not ClusteringDocument for doc in documents):
        raise DataValidationError("无监督路径只接受严格的 ClusteringDocument(id,text)")
    if len({doc.id for doc in documents}) != len(documents):
        raise DataValidationError("输入 id 重复")
    k_values, seeds = config["k_values"], config["stability_seeds"]
    primary_seed = config["kmeans"]["random_state"]
    if (
        not k_values
        or len(set(k_values)) != len(k_values)
        or config["reference_k"] not in k_values
        or len(set(seeds)) != len(seeds)
        or len(seeds) < 2
        or primary_seed not in seeds
    ):
        raise DataValidationError("sweep/reference/stability 配置无效")
    if config["projection"].get("n_components") != 2:
        raise DataValidationError("可视化要求二维 SVD")
    pre_config = dict(config["preprocessing"])
    pre_config["stopwords"] = tuple(pre_config.get("stopwords", []))
    processor = ChinesePreprocessor(segment, PreprocessingConfig(**pre_config))
    if tokenized is None:
        tokenized = [processor.tokens(doc.text) for doc in documents]
    elif len(tokenized) != len(documents) or any(
        not isinstance(words, list) or any(not isinstance(word, str) for word in words)
        for words in tokenized
    ):
        raise DataValidationError("缓存 token 必须与无标签输入逐条对齐")
    progress(f"tokenized {len(documents)} documents")
    representation = TextTFIDF(TFIDFConfig(**config["representation"]))
    word_vectors = representation.fit_transform(tokenized)
    lsa = None
    if config.get("lsa") is not None:
        lsa = TextLSA(LSAConfig(**config["lsa"]))
        vectors = lsa.fit_transform(word_vectors)
    else:
        vectors = word_vectors
    dense = vectors.toarray() if lsa is None else vectors
    # 内部指标与代表距离都在实际 K-means 输入空间，不在绘图投影中。
    terms = representation.terms
    example_vector = word_vectors.getrow(0)
    example_weights = dict(zip(example_vector.indices, example_vector.data))
    example_indices = sorted(example_vector.indices, key=lambda i: (-example_weights[i], terms[i]))[
        :12
    ]
    stats = {
        "document_count": len(documents),
        "vocabulary_size": len(terms),
        "dimension": vectors.shape[1],
        "nnz": int(word_vectors.nnz) if lsa is None else int(np.count_nonzero(vectors)),
        "sparsity": float(1 - word_vectors.nnz / np.prod(word_vectors.shape))
        if lsa is None
        else None,
        "feature_dimension": word_vectors.shape[1],
        "representation_dimension": vectors.shape[1],
        "word_space_sparsity": float(1 - word_vectors.nnz / np.prod(word_vectors.shape)),
        "space": "lsa_l2" if lsa is not None else "tfidf_l2",
        "lsa": lsa.metadata() if lsa is not None else None,
        "top_terms_method": "mean TF-IDF of cluster members"
        if lsa is not None
        else "TF-IDF centroid weights",
        "zero_vector_count": 0,
        "token_count": sum(map(len, tokenized)),
        "example": {
            "id": documents[0].id,
            "snippet": snippet(documents[0].text),
            "tokens": tokenized[0][:100],
            "top_features": [
                {"term": str(terms[i]), "weight": float(example_weights[i])}
                for i in example_indices
            ],
        },
    }
    base = KMeansConfig(**config["kmeans"])
    models, results, sweep, stability_records = {}, {}, [], []
    for k in k_values:
        trials = {}
        for seed in seeds:
            model = KMeansClusterer(replace(base, n_clusters=k, random_state=seed))
            labels = model.fit_predict(vectors)
            trials[seed] = labels
            stability_records.append(
                {
                    "k": k,
                    "seed": seed,
                    "inertia": model.inertia,
                    "n_iter": model.n_iter,
                    "cluster_sizes": np.bincount(labels, minlength=k).tolist(),
                    "cluster_ids": labels.tolist(),
                }
            )
            if seed == primary_seed:
                models[k] = model
        labels, model = trials[primary_seed], models[k]
        with threadpool_limits(limits=1):
            metrics = internal_metrics(vectors, dense, labels, model)
        term_centers = None
        if lsa is not None:
            term_centers = np.vstack(
                [
                    np.asarray(word_vectors[labels == cluster_id].mean(axis=0)).ravel()
                    for cluster_id in range(k)
                ]
            )
        assignments, clusters = summarize_clusters(
            documents, vectors, labels, model.centers, terms, term_centers=term_centers
        )
        results[k] = {
            "k": k,
            "role": "sweep",
            "internal": metrics,
            "assignments": assignments,
            "clusters": clusters,
            "clustering_space": stats["space"],
            "top_terms_method": stats["top_terms_method"],
        }
        pairwise = [
            {"seeds": [a, b], "ari": float(adjusted_rand_score(trials[a], trials[b]))}
            for a, b in combinations(seeds, 2)
        ]
        sizes = metrics["cluster_sizes"]
        sweep.append(
            {
                "k": k,
                **metrics,
                "stability": {
                    "seeds": seeds,
                    "mean_ari": float(np.mean([x["ari"] for x in pairwise])),
                    "min_ari": min(x["ari"] for x in pairwise),
                    "pairs": pairwise,
                },
                "min_cluster_size": min(sizes),
                "max_cluster_size": max(sizes),
                "imbalance_ratio": max(sizes) / min(sizes),
            }
        )
        progress(f"K={k}: inertia={model.inertia:.4f}, silhouette={metrics['silhouette']:.4f}")
    projection_config = config["projection"]
    svd = TruncatedSVD(
        n_components=2,
        n_iter=projection_config["n_iter"],
        random_state=projection_config["random_state"],
    )
    with threadpool_limits(limits=1):
        coords = svd.fit_transform(vectors)
    projection = {
        "method": "TruncatedSVD",
        "purpose": "visualization_only",
        "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
        "points": [
            {"id": doc.id, "x": float(x), "y": float(y)} for doc, (x, y) in zip(documents, coords)
        ],
    }
    return {
        "representation": representation,
        "vectors": vectors,
        "word_vectors": word_vectors,
        "lsa": lsa,
        "tokenized": tokenized,
        "stats": stats,
        "models": models,
        "results": results,
        "sweep": sweep,
        "stability_records": stability_records,
        "projection": projection,
        "svd": svd,
    }
