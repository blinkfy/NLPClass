"""Train-derived span cleaning and boundary-preserving feature inputs (no labels)."""

import re
import unicodedata

import numpy as np

from nlplab.features.word_sequence import retain_head_tail


def normalize(text):
    return unicodedata.normalize("NFKC", text).lower()


def find_spans(text, rules):
    """Offsets refer to the NFKC/lowercase text, identical for all conditions."""
    normalized = normalize(text)
    found = []
    for rule in rules["rules"]:
        for match in re.finditer(rule["pattern"], normalized, re.MULTILINE):
            found.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "kind": rule["kind"],
                    "rule": rule["id"],
                    "text": match.group(),
                }
            )
    return normalized, sorted(found, key=lambda s: (s["start"], s["end"], s["rule"]))


def merge_intervals(spans):
    merged = []
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        a, b = span["start"], span["end"]
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def clean_view(text, rules, kinds):
    normalized, spans = find_spans(text, rules)
    removed = [s for s in spans if s["kind"] in kinds]
    intervals = merge_intervals(removed)
    segments, cursor = [], 0
    for a, b in intervals:
        if a > cursor:
            segments.append({"text": normalized[cursor:a], "start": cursor, "end": a})
        cursor = b
    if cursor < len(normalized):
        segments.append({"text": normalized[cursor:], "start": cursor, "end": len(normalized)})
    # Newline boundaries protect char bigrams. Word/CNN use explicit segment IDs.
    return {
        "text": normalized,
        "segments": segments,
        "removed_spans": removed,
        "char_text": "\n".join(s["text"] for s in segments),
        "removed_characters": sum(b - a for a, b in intervals),
    }


def tokenize_view(processor, view):
    """Reuse frozen HMM and filtering, recording exact normalized character offsets."""
    chunks, tokens, offsets, segment_ids = [], [], [], []
    for segment_id, segment in enumerate(view["segments"]):
        section = []
        for match in processor.pattern.finditer(segment["text"]):
            chunk = match.group()
            words = [chunk] if chunk.isascii() else processor.segment(chunk)
            cursor = 0
            for word in words:
                local = chunk.find(word, cursor)
                if local < 0:
                    raise ValueError("Tokenizer output cannot map to input characters")
                cursor = local + len(word)
                if len(word) < processor.config.min_token_length or word in processor.stopwords:
                    continue
                section.append(word)
                tokens.append(word)
                start = segment["start"] + match.start() + local
                offsets.append([start, start + len(word)])
                segment_ids.append(segment_id)
        chunks.append(section)
    return {"sections": chunks, "tokens": tokens, "offsets": offsets, "segment_ids": segment_ids}


def segmented_word_features(sections):
    """Same frozen uni+bi features, excluding adjacency across deletion intervals."""
    return [
        feature
        for tokens in sections
        for feature in (list(tokens) + [a + " " + b for a, b in zip(tokens, tokens[1:])])
    ]


def encode_sections(vocabulary, token_rows):
    encoded = vocabulary.encode([row["tokens"] for row in token_rows])
    segments = np.full(encoded["ids"].shape, -1, dtype=np.int64)
    retained_offsets = []
    for i, row in enumerate(token_rows):
        _, positions = retain_head_tail(row["tokens"], vocabulary.max_len)
        segments[i, : len(positions)] = [row["segment_ids"][p] for p in positions]
        retained_offsets.append([row["offsets"][p] for p in positions])
    encoded["segment_ids"] = segments
    encoded["character_offsets"] = retained_offsets
    return encoded


def overlap(offsets, spans):
    return any(a < s["end"] and b > s["start"] for a, b in offsets for s in spans)


def prevalence(documents, views):
    from nlplab.data.thucnews import CATEGORIES

    result = []
    for label in CATEGORIES:
        selected = [(d, v) for d, v in zip(documents, views) if d.label == label]
        n = len(selected)
        source = sum(any(s["kind"] == "source" for s in v["removed_spans"]) for _, v in selected)
        template = sum(
            any(s["kind"] == "template" for s in v["removed_spans"]) for _, v in selected
        )
        characters = sum(len(v["text"]) for _, v in selected)
        removed = sum(v["removed_characters"] for _, v in selected)
        result.append(
            {
                "label": label,
                "count": n,
                "source_documents": source,
                "template_documents": template,
                "source_rate": source / n,
                "template_rate": template / n,
                "removed_characters": removed,
                "original_characters": characters,
                "removed_character_ratio": removed / characters if characters else 0,
            }
        )
    return result
