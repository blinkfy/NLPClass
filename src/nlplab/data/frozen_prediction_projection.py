"""Root-field projection from frozen JSONL; irrelevant values are skipped without decoding."""

import hashlib
import json

from nlplab.core.exceptions import DataValidationError

FIELDS = frozenset({"document_id", "gold", "prediction"})


def end_string(data, position):
    if data[position] != 34:
        raise DataValidationError("Expected JSON string")
    position += 1
    while position < len(data):
        char = data[position]
        if char == 92:
            position += 2
        elif char == 34:
            return position + 1
        else:
            position += 1
    raise DataValidationError("Unterminated JSON string")


def skip_value(data, position):
    if data[position] == 34:
        return end_string(data, position)
    if data[position] in (91, 123):
        stack = [data[position]]
        position += 1
        while position < len(data):
            char = data[position]
            if char == 34:
                position = end_string(data, position)
                continue
            if char in (91, 123):
                stack.append(char)
            elif char in (93, 125):
                if not stack or (stack[-1], char) not in ((91, 93), (123, 125)):
                    raise DataValidationError("Malformed JSON container")
                stack.pop()
                if not stack:
                    return position + 1
            position += 1
        raise DataValidationError("Unterminated JSON container")
    while position < len(data) and data[position] not in b",}\r\n \t":
        position += 1
    return position


def project_record(data):
    position, result = 0, {}

    def whitespace(pos):
        while pos < len(data) and data[pos] in b" \t\r\n":
            pos += 1
        return pos

    position = whitespace(position)
    if data[position] != 123:
        raise DataValidationError("Prediction records must be JSON objects")
    position += 1
    while True:
        position = whitespace(position)
        if position >= len(data):
            raise DataValidationError("Incomplete prediction object")
        if data[position] == 125:
            break
        stop = end_string(data, position)
        key = json.loads(data[position:stop])
        position = whitespace(stop)
        if data[position] != 58:
            raise DataValidationError("Expected JSON colon")
        position = whitespace(position + 1)
        stop = skip_value(data, position)
        if key in FIELDS:
            if key in result or data[position] != 34:
                raise DataValidationError("Projected fields must be unique strings")
            result[key] = json.loads(data[position:stop])
        position = whitespace(stop)
        if data[position] == 125:
            break
        if data[position] != 44:
            raise DataValidationError("Expected JSON separator")
        position += 1
    if set(result) != FIELDS:
        raise DataValidationError("Missing frozen prediction fields")
    return result


def load_projection(path, expected_sha256):
    digest, rows = hashlib.sha256(), []
    with path.open("rb") as stream:
        for data in stream:
            digest.update(data)
            if data.strip():
                rows.append(project_record(data))
    if digest.hexdigest() != expected_sha256:
        raise DataValidationError("Frozen prediction container hash mismatch; stop")
    return rows
