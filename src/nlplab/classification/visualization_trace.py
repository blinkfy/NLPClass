"""One eval-only CPU forward of a frozen saved encoding; no tokenizer or fit."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from nlplab.classification.textcnn import TextCNN, set_seed
from nlplab.core.exceptions import DataValidationError
from nlplab.data.thucnews import CATEGORIES


def trace_model(model, ids, length, saved):
    tracks = {}
    hooks = []
    for convolution in model.convs:
        kernel = convolution.kernel_size[0]
        def capture(module, inputs, output, k=kernel):
            tracks[str(k)] = torch.relu(output[0])[:, :max(0, length-k+1)].detach().cpu().numpy().tolist()
        hooks.append(convolution.register_forward_hook(capture))
    try:
        model.eval()
        with torch.inference_mode():
            logits, pooled, positions = model(torch.as_tensor(ids[None, :], dtype=torch.long), torch.tensor([length]), True)
    finally:
        for hook in hooks:
            hook.remove()
    logits, pooled, positions = logits[0].numpy(), pooled[0].numpy(), positions[0].numpy()
    expected = np.asarray(saved["raw_logits"])
    if not np.allclose(logits, expected, atol=2e-5, rtol=2e-5) or CATEGORIES[int(logits.argmax())] != saved["prediction"]:
        raise DataValidationError("diagnostic logits/prediction 与冻结结果不一致")
    channels = sorted(saved["all_channels"], key=lambda r: r["channel"])
    if len(channels) != 384 or not np.allclose(pooled, [r["max_activation"] for r in channels], atol=2e-5, rtol=2e-5) or not np.array_equal(positions, [r["argmax_position"] for r in channels]):
        raise DataValidationError("diagnostic pooled/argmax 与冻结结果不一致")
    weights = model.classifier.weight.detach().numpy()
    bias = model.classifier.bias.detach().numpy()
    decomposition = bias.astype(np.float64) + weights.astype(np.float64) @ pooled.astype(np.float64)
    if not np.allclose(decomposition, logits, atol=2e-5, rtol=2e-5):
        raise DataValidationError("diagnostic 通道贡献无法还原 logits")
    return {"activations": tracks, "pooled": pooled.tolist(), "argmax": positions.tolist(), "weights": weights.tolist(), "bias": bias.tolist(), "logits": logits.tolist(), "frozen_logits": expected.tolist(), "label_order": list(CATEGORIES),
            "prediction": saved["prediction"], "runner_up": saved["runner_up"],
            "verification": {"logits_match": True, "prediction_match": True, "pooled_match": True, "argmax_match": True, "max_logit_abs_error": float(np.max(np.abs(logits-expected))), "atol": 2e-5, "rtol": 2e-5},
            "diagnostic_forward_only": True, "formal_evaluation_executed": False, "fit_executed": False, "device": "cpu", "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--directory", required=True); parser.add_argument("--document-id", required=True)
    args = parser.parse_args(); directory = Path(args.directory)
    encoding = json.loads((directory / "test_encoding.json").read_text("utf-8"))
    if args.document_id not in encoding["document_ids"]:
        raise DataValidationError("非冻结 Test document ID")
    index = encoding["document_ids"].index(args.document_id)
    with (directory / "C3_test_predictions.jsonl").open(encoding="utf-8") as stream:
        saved = next((r for line in stream if (r := json.loads(line))["document_id"] == args.document_id), None)
    if saved is None:
        raise DataValidationError("冻结预测不存在")
    with np.load(directory / "test_encoding.npz", allow_pickle=False) as arrays:
        ids, length = arrays["ids"][index].copy(), int(arrays["lengths"][index])
    set_seed(42)
    trace = trace_model(TextCNN.load(directory / "final/model"), ids, length, saved)
    trace.update({"document_id": args.document_id, "retained_length": length, "padded_shape": [1, len(ids)], "embedding_shape": [1, len(ids), 128], "max_len": len(ids), "diagnostic_forward_count": 1})
    print(json.dumps(trace, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
