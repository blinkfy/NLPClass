"""Export the actual frozen HanLP POS transformer; never fit or substitute weights."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export_transformers(destination):
    output = Path(destination) / "models/transformers"
    output.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / "artifacts/runtime/hanlp-venv/Scripts/python.exe"
    converter = ROOT / "build/onnx-converter"
    if not runtime.is_file() or not converter.is_dir():
        return {"unavailable": {"pos": "ONNX export runtime is not installed"}}
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(converter), str(ROOT / "src"))))
    subprocess.run([str(runtime), __file__, "--pos", str(output)], check=True, cwd=ROOT, env=env)
    subprocess.run([str(runtime), __file__, "--ner", str(output)], check=True, cwd=ROOT, env=env)
    subprocess.run(
        [str(runtime), __file__, "--dependency", str(output)], check=True, cwd=ROOT, env=env
    )
    subprocess.run([str(runtime), __file__, "--bert", str(output)], check=True, cwd=ROOT, env=env)
    return {
        key: {"path": f"models/transformers/{key}.json"}
        for key in ("pos", "ner", "dependency", "bert")
    }


def export_bert(output):
    import math
    import tempfile

    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    temporary = ROOT / "build/onnx-temp"
    temporary.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temporary)
    snapshots = (
        Path(os.environ["USERPROFILE"])
        / ".cache/huggingface/hub/models--bert-base-chinese/snapshots"
    )
    source = snapshots / "8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    reference = json.loads(
        (ROOT / "artifacts/course_models/bert-reference.json").read_text(encoding="utf-8")
    )
    expected = reference["files"]["model.safetensors"]["sha256"]
    assert hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest() == expected
    model = AutoModelForMaskedLM.from_pretrained(
        str(source), local_files_only=True, use_safetensors=True, attn_implementation="eager"
    ).eval()

    class BertTraceGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            output = self.model(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_attentions=True,
                output_hidden_states=True,
            )
            scores, norms, previews, context_norms = [], [], [], []
            for i, layer in enumerate(self.model.bert.encoder.layer):
                attention = layer.attention.self
                hidden = output.hidden_states[i]
                q, k, v = [
                    attention.transpose_for_scores(fn(hidden))
                    for fn in (attention.query, attention.key, attention.value)
                ]
                scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
                    attention.attention_head_size
                )
                context = torch.matmul(output.attentions[i], v)
                scores.append(scaled[0])
                norms.append(torch.stack([x.norm(dim=-1)[0] for x in (q, k, v)]))
                previews.append(torch.stack([x[0, :, :, :8] for x in (q, k, v, context)]))
                context_norms.append(context.norm(dim=-1)[0])
            return (
                output.logits[0],
                torch.stack([a[0] for a in output.attentions]),
                torch.stack(scores),
                torch.stack(norms),
                torch.stack(previews),
                torch.stack(context_norms),
            )

    graph = BertTraceGraph().eval()
    output.mkdir(parents=True, exist_ok=True)
    floating = ROOT / "build/bert-float.onnx"
    file = output / "bert.onnx"
    names = ["logits", "attention", "scaled", "norms", "previews", "context_norms"]
    axes = {
        "input_ids": {1: "tokens"},
        "logits": {0: "tokens"},
        "attention": {2: "tokens", 3: "tokens"},
        "scaled": {2: "tokens", 3: "tokens"},
        "norms": {3: "tokens"},
        "previews": {3: "tokens"},
        "context_norms": {2: "tokens"},
    }
    torch.onnx.export(
        graph,
        (torch.tensor([[101, 704, 1744, 103, 102]]),),
        str(floating),
        dynamo=False,
        opset_version=17,
        input_names=["input_ids"],
        output_names=names,
        dynamic_axes=axes,
    )
    import shutil

    shutil.copyfile(floating, file)
    # The legacy exporter folds the tied MLM decoder into a second transposed
    # copy of the 65 MB embedding. Keep one exact copy in the published assets.
    import onnx
    from onnx import helper, numpy_helper

    serialized = onnx.load(str(file))
    embedding = next(t for t in serialized.graph.initializer if list(t.dims) == [21128, 768])
    duplicates = [t for t in serialized.graph.initializer if list(t.dims) == [768, 21128]]
    for duplicate in duplicates:
        if np.array_equal(numpy_helper.to_array(embedding).T, numpy_helper.to_array(duplicate)):
            serialized.graph.initializer.remove(duplicate)
            serialized.graph.node.insert(
                0, helper.make_node("Transpose", [embedding.name], [duplicate.name], perm=[1, 0])
            )
    onnx.save(serialized, str(file))
    external = split_external_weights(file)
    session = ort.InferenceSession(str(file), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, use_fast=True)
    fixtures = []
    for text in [
        "中国的首都是[MASK]。",
        "人工智能帮助学生学习。",
        "AI研究2026年算法😊",
        "[PAD]中国[MASK]。",
    ]:
        encoded = tokenizer(text, return_offsets_mapping=True)
        ids = np.array([encoded["input_ids"]], dtype=np.int64)
        actual = session.run(None, {"input_ids": ids})
        with torch.inference_mode():
            native = graph(torch.from_numpy(ids))
        attention, scaled = actual[1], actual[2]
        p = np.exp(scaled - scaled.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        error = float(np.max(np.abs(p - attention)))
        assert error < 1e-5
        masks = []
        for i, token in enumerate(ids[0]):
            if token == 103:
                order = np.argsort(-actual[0][i], kind="stable")[:8].tolist()
                masks.append({"position": i, "top8": order})
                assert order[0] == int(native[0][i].argmax())
        fixtures.append(
            {
                "text": text,
                "ids": ids[0].tolist(),
                "offsets": encoded["offset_mapping"],
                "masks": masks,
                "max_softmax_error": error,
            }
        )
    (ROOT / "build/bert-parity.json").write_text(
        json.dumps(fixtures, ensure_ascii=False), encoding="utf-8"
    )
    descriptor = {
        "model": "bert-base-chinese",
        "graph": file.name,
        "bytes": file.stat().st_size,
        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "external": external,
        "source_sha256": expected,
        "revision": source.name,
        "precision": "original float32 weights; no quantization",
        "vocabulary": [
            token for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
        ],
        "max_subwords": 64,
        "lowercase": bool(tokenizer.do_lower_case),
        "verification": {
            "synthetic_cases": len(fixtures),
            "max_softmax_error": max(f["max_softmax_error"] for f in fixtures),
        },
    }
    (output / "bert.json").write_text(
        json.dumps(descriptor, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(json.dumps(descriptor["verification"]))


def export_dependency(output):
    os.environ["HANLP_HOME"] = str(ROOT / "artifacts/cache/hanlp")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import hanlp
    import numpy as np
    import onnxruntime as ort
    import torch

    source = (
        ROOT
        / "artifacts/course_models/hanlp-ctb9-dep-electra-small/extracted-dc421a93ba02"
        / "ctb9_dep_electra_small_20220216_100306"
    )
    parser = hanlp.load(str(source), devices=-1)
    parser.model.eval()
    assert parser.config.tree and parser.config.proj
    assert not parser.model.encoder.transformer.average_subwords
    assert not hasattr(parser.model.encoder, "lstm")
    assert not hasattr(parser.model.encoder, "feat_embed")

    class DependencyGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = parser.model.encoder.transformer.transformer
            self.decoder = parser.model.biaffine_decoder

        def forward(self, input_ids, token_span):
            hidden = self.encoder(input_ids, attention_mask=input_ids.ne(0))[0]
            words = hidden[:, token_span[:, 0], :]
            return self.decoder(words)

    graph = DependencyGraph().eval()
    output.mkdir(parents=True, exist_ok=True)
    file = output / "dependency.onnx"
    ids = torch.tensor([[101, 2769, 1599, 1266, 776, 102]])
    spans = torch.tensor([[0], [1], [2], [3]])
    torch.onnx.export(
        graph,
        (ids, spans),
        str(file),
        dynamo=False,
        opset_version=17,
        input_names=["input_ids", "token_span"],
        output_names=["arc", "relation"],
        dynamic_axes={
            "input_ids": {1: "subwords"},
            "token_span": {0: "words"},
            "arc": {1: "words", 2: "words"},
            "relation": {1: "words", 2: "words"},
        },
    )
    tokenizer = parser.transformer_tokenizer
    normalization = json.loads(
        (ROOT / "artifacts/cache/hanlp/utils/char_table_20210602_202632.json").read_text(
            encoding="utf-8"
        )
    )
    session = ort.InferenceSession(str(file), providers=["CPUExecutionProvider"])
    fixtures, errors = [], []
    for words in [
        ["我", "爱", "中国", "。"],
        ["北京", "大学", "的", "学生", "学习", "人工智能", "。"],
        ["AI", "研究", "2026", "年", "算法", "😊"],
    ]:
        ids, spans = [101], [[0]]
        for word in words:
            mapped = "".join(normalization.get(c, c) for c in word)
            pieces = tokenizer(mapped, add_special_tokens=False)["input_ids"] or [100]
            spans.append([len(ids)])
            ids.extend(pieces)
        ids.append(102)
        inputs = {
            "input_ids": np.array([ids], dtype=np.int64),
            "token_span": np.array(spans, dtype=np.int64),
        }
        actual = session.run(None, inputs)
        captured = []
        hook = parser.model.register_forward_hook(
            lambda module, args, result: captured.append(
                tuple(t.detach().cpu().numpy() for t in result)
            )
        )
        try:
            with torch.inference_mode():
                parsed = parser(words)
        finally:
            hook.remove()
        for a, b in zip(actual, captured[0]):
            errors.append(float(np.max(np.abs(a - b))))
            assert np.max(np.abs(a - b)) < 5e-4
        fixtures.append(
            {
                "words": words,
                "ids": ids,
                "spans": spans,
                "heads": [int(w.head) for w in parsed],
                "relations": [w.deprel for w in parsed],
                "arc": actual[0][0].tolist(),
                "relation": actual[1][0].tolist(),
            }
        )
    (ROOT / "build/dependency-parity.json").write_text(
        json.dumps(fixtures, ensure_ascii=False), encoding="utf-8"
    )
    descriptor = {
        "model": "HanLP CTB9 ELECTRA-small biaffine",
        "graph": file.name,
        "bytes": file.stat().st_size,
        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256((source / "model.pt").read_bytes()).hexdigest(),
        "normalization": normalization,
        "labels": parser.vocabs["rel"].idx_to_token,
        "vocabulary": [
            token for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
        ],
        "max_subwords": 512,
        "lowercase": bool(tokenizer.do_lower_case),
        "verification": {"synthetic_cases": len(fixtures), "max_logit_error": max(errors)},
    }
    (output / "dependency.json").write_text(
        json.dumps(descriptor, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(json.dumps(descriptor["verification"]))


def split_external_weights(file):
    """Keep every GitHub file below 100 MB, with relative ONNX external-data names."""
    import onnx
    from onnx.external_data_helper import set_external_data

    graph = onnx.load(str(file))
    for stale in file.parent.glob(f"{file.stem}-weights-*.bin"):
        if stale.is_file() and not stale.is_symlink():
            stale.unlink()
    files, chunk, stream, offset = [], -1, None, 0
    try:
        for tensor in graph.graph.initializer:
            if not tensor.raw_data or len(tensor.raw_data) < 1024:
                continue
            size = len(tensor.raw_data)
            if size > 90_000_000:
                raise ValueError("Tensor exceeds the static single-file budget")
            if stream is None or offset + size > 90_000_000:
                if stream:
                    stream.close()
                chunk += 1
                name = f"{file.stem}-weights-{chunk}.bin"
                files.append(name)
                stream = (file.parent / name).open("wb")
                offset = 0
            stream.write(tensor.raw_data)
            set_external_data(tensor, location=files[-1], offset=offset, length=size)
            tensor.ClearField("raw_data")
            offset += size
    finally:
        if stream:
            stream.close()
    # Serializing directly avoids ONNX's writer resetting our deliberate offsets.
    file.write_bytes(graph.SerializeToString())
    return [
        {
            "file": name,
            "bytes": (file.parent / name).stat().st_size,
            "sha256": hashlib.sha256((file.parent / name).read_bytes()).hexdigest(),
        }
        for name in files
    ]


def export_ner(output):
    import tempfile

    import h5py
    import numpy as np
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer, BertConfig, BertModel

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    temporary = ROOT / "build/onnx-temp"
    temporary.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temporary)
    source = ROOT / "artifacts/cache/hanlp/ner/ner_bert_base_msra_20211227_114712"
    # Reuse only architecture/tokenizer configuration, never the reference BERT weights.
    snapshots = (
        Path(os.environ["USERPROFILE"])
        / ".cache/huggingface/hub/models--bert-base-chinese/snapshots"
    )
    backbone = next(path for path in snapshots.iterdir() if (path / "config.json").is_file())
    config = BertConfig.from_pretrained(str(backbone), local_files_only=True)
    config._attn_implementation = "eager"
    encoder = BertModel(config).eval()
    state = {}
    with h5py.File(source / "model.h5") as h5:
        prefix = "tf_bert_model/tf_bert_model/bert/"
        for name, tensor in encoder.state_dict().items():
            stem, kind = name.rsplit(".", 1)
            import re

            key = re.sub(r"encoder/layer/(\d+)", r"encoder/layer_._\1", stem.replace(".", "/"))
            if "LayerNorm" in stem:
                suffix = "gamma" if kind == "weight" else "beta"
            elif stem == "embeddings.word_embeddings":
                suffix = "weight"
            elif stem in ("embeddings.position_embeddings", "embeddings.token_type_embeddings"):
                suffix = "embeddings"
            else:
                suffix = "kernel" if kind == "weight" else "bias"
            value = h5[prefix + key + "/" + suffix + ":0"][()]
            if suffix == "kernel":
                value = value.T
            assert tuple(value.shape) == tuple(tensor.shape), name
            state[name] = torch.from_numpy(value.copy())
        encoder.load_state_dict(state, strict=True)
        head = torch.nn.Linear(768, 14)
        head.weight.data.copy_(torch.from_numpy(h5["dense/dense/kernel:0"][()].T.copy()))
        head.bias.data.copy_(torch.from_numpy(h5["dense/dense/bias:0"][()]))

    class NerGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.head = encoder, head

        def forward(self, input_ids):
            return self.head(self.encoder(input_ids, attention_mask=input_ids.ne(0))[0])

    graph = NerGraph().eval()
    output.mkdir(parents=True, exist_ok=True)
    floating = ROOT / "build/ner-float.onnx"
    file = output / "ner.onnx"
    torch.onnx.export(
        graph,
        (torch.tensor([[101, 2769, 1599, 1266, 776, 102]]),),
        str(floating),
        dynamo=False,
        opset_version=17,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {1: "subwords"}, "logits": {1: "subwords"}},
    )
    quantize_dynamic(
        str(floating), str(file), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"]
    )
    external = split_external_weights(file)
    import onnxruntime as ort

    session = ort.InferenceSession(str(file), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(str(backbone), local_files_only=True, use_fast=False)
    labels = json.loads((source / "vocabs.json").read_text(encoding="utf-8"))["tag_vocab"][
        "idx_to_token"
    ]
    examples = [
        "张三在北京大学学习。",
        "微软公司位于美国。",
        "李明去上海参加会议。",
        "😊AI研究2026年新算法。",
    ]
    fixtures, errors = [], []
    for text in examples:
        ids, positions = [101], []
        for c in text:
            pieces = tokenizer.encode(c, add_special_tokens=False) or [100]
            positions.append(len(ids))
            ids.extend(pieces)
        ids.append(102)
        array = np.array([ids], dtype=np.int64)
        with torch.inference_mode():
            expected = graph(torch.from_numpy(array)).numpy()[0]
        actual = session.run(None, {"input_ids": array})[0][0]
        expected[:, 0] = actual[:, 0] = -np.inf
        original_tags = [labels[i] for i in expected[positions].argmax(1)]
        tags = [labels[i] for i in actual[positions].argmax(1)]
        assert tags == original_tags, text
        errors.append(float(np.max(np.abs(expected[:, 1:] - actual[:, 1:]))))
        fixtures.append(
            {
                "text": text,
                "ids": ids,
                "positions": positions,
                "tags": tags,
                "float_logits": expected[:, 1:].tolist(),
            }
        )
    (ROOT / "build/ner-parity.json").write_text(
        json.dumps(fixtures, ensure_ascii=False), encoding="utf-8"
    )
    descriptor = {
        "model": "hanlp-msra-bert",
        "graph": file.name,
        "bytes": file.stat().st_size,
        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "external": external,
        "precision": "dynamic-int8 MatMul; original float32 embeddings",
        "source_sha256": hashlib.sha256((source / "model.h5").read_bytes()).hexdigest(),
        "labels": labels,
        "label_map": {"NR": "PER", "NS": "LOC", "NT": "ORG"},
        "vocabulary": [
            token for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
        ],
        "max_subwords": 128,
        "lowercase": bool(tokenizer.do_lower_case),
        "verification": {
            "synthetic_cases": len(examples),
            "max_quantized_logit_error": max(errors),
        },
    }
    (output / "ner.json").write_text(
        json.dumps(descriptor, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(json.dumps(descriptor["verification"]))


def export_pos(output):
    os.environ["HANLP_HOME"] = str(ROOT / "artifacts/cache/hanlp")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import hanlp
    import torch

    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "artifacts/cache/hanlp/pos/pos_ctb_electra_small_20220215_111944"
    component = hanlp.load(str(source), devices=-1)
    component.model.eval()

    class PosGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = component.model.encoder.transformer
            self.classifier = component.model.classifier

        def forward(self, input_ids, token_span):
            hidden = self.encoder(input_ids, attention_mask=input_ids.ne(0))[0]
            selected = hidden[0][token_span]
            mask = token_span.ne(0).unsqueeze(-1)
            pooled = (selected * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.classifier(pooled)

    graph = PosGraph().eval()
    ids = torch.tensor([[101, 2769, 1599, 1266, 776, 102]], dtype=torch.long)
    spans = torch.tensor([[1, 0], [2, 0], [3, 4]], dtype=torch.long)
    file = output / "pos.onnx"
    torch.onnx.export(
        graph,
        (ids, spans),
        str(file),
        dynamo=False,
        opset_version=17,
        input_names=["input_ids", "token_span"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {1: "subwords"},
            "token_span": {0: "words", 1: "pieces"},
            "logits": {0: "words"},
        },
    )
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(file), providers=["CPUExecutionProvider"])
    tests = [
        ["我", "爱", "中国"],
        ["GitHub", "AI", "2026", "😊"],
        ["北京", "大学", "的", "学生", "。"],
    ]
    errors = []
    tokenizer = component.transformer_tokenizer
    for words in tests:
        mapped = json.loads(
            (ROOT / "artifacts/cache/hanlp/utils/char_table_20210602_202632.json").read_text(
                encoding="utf-8"
            )
        )
        normalized = ["".join(mapped.get(c, c) for c in w) for w in words]
        pieces = [tokenizer(w, add_special_tokens=False)["input_ids"] or [100] for w in normalized]
        ids = [101]
        spans = []
        for row in pieces:
            spans.append(list(range(len(ids), len(ids) + len(row))))
            ids.extend(row)
        ids.append(102)
        width = max(map(len, spans))
        spans = [row + [0] * (width - len(row)) for row in spans]
        inputs = {
            "input_ids": np.array([ids], dtype=np.int64),
            "token_span": np.array(spans, dtype=np.int64),
        }
        with torch.inference_mode():
            expected = graph(*(torch.from_numpy(inputs[k]) for k in inputs)).numpy()
        actual = session.run(None, inputs)[0]
        errors.append(float(np.max(np.abs(expected - actual))))
        labels = component.vocabs["tag"].idx_to_token
        assert [labels[i] for i in actual.argmax(1)] == component(words)
        assert np.max(np.abs(expected - actual)) < 2e-4
    rules = json.loads(
        (ROOT / "artifacts/cache/hanlp/utils/char_table_20210602_202632.json").read_text(
            encoding="utf-8"
        )
    )
    descriptor = {
        "model": "hanlp-ctb9-pos-electra-small",
        "graph": file.name,
        "bytes": file.stat().st_size,
        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256((source / "model.pt").read_bytes()).hexdigest(),
        "vocabulary": [
            token for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
        ],
        "labels": labels,
        "normalization": rules,
        "max_subwords": 510,
        "lowercase": bool(tokenizer.do_lower_case),
        "verification": {"synthetic_cases": len(tests), "max_logit_error": max(errors)},
    }
    (output / "pos.json").write_text(
        json.dumps(descriptor, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(json.dumps(descriptor["verification"]))


if __name__ == "__main__":
    if sys.argv[1] == "--pos":
        export_pos(Path(sys.argv[2]))
    elif sys.argv[1] == "--ner":
        export_ner(Path(sys.argv[2]))
    elif sys.argv[1] == "--dependency":
        export_dependency(Path(sys.argv[2]))
    elif sys.argv[1] == "--bert":
        export_bert(Path(sys.argv[2]))
