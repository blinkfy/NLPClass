"""Presentation offsets for existing C2 explanations, not a feature extractor."""

import copy
import unicodedata


def evidence_occurrences(text, feature):
    # A conservative normalization map. Composition across source code points
    # cannot be assigned independently, so fail closed rather than invent spans.
    normalized, offsets = "", []
    for position, char in enumerate(text):
        part = unicodedata.normalize("NFKC", char).lower()
        normalized += part
        offsets.extend([(position, position + 1)] * len(part))
    if normalized != unicodedata.normalize("NFKC", text).lower() or len(feature) != 2:
        return []
    rows, seen, position = [], set(), 0
    while True:
        position = normalized.find(feature, position)
        if position < 0:
            break
        start, end = offsets[position][0], offsets[position + len(feature) - 1][1]
        original = text[start:end]
        if (start, end) not in seen and unicodedata.normalize("NFKC", original).lower() == feature:
            rows.append({"start": start, "end": end, "text": original})
            seen.add((start, end))
        position += 1
    return rows


def attach_evidence(text, prediction, metadata=None):
    result = copy.deepcopy(prediction)
    direct = {row["term"]: row for row in result.get("top_contributing_features", [])}
    evidence = []
    for row in result.get("top_relative_features", []):
        feature = row["term"]
        item = {"feature": feature, "tfidf": row["tfidf"],
                "relative_contribution": row["contribution"],
                "coefficient_difference": row["coefficient_difference"],
                "occurrences": evidence_occurrences(text, feature)}
        if feature in direct:
            item.update({key: direct[feature][key] for key in ("weight", "direct_contribution")
                         if key in direct[feature]})
        item.update((metadata or {}).get(feature, {}))
        evidence.append(item)
    result.update(evidence=evidence, evidence_method="existing C2 top_relative_features; x_j*(w_pred-w_runner)",
                  offset_space="original_text Unicode code-point", status="ok", domain="五类中文新闻")
    return result
