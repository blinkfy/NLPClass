"""隔离神经 runtime 的数值 I/O CLI；不读取 THUCNews split 或原始标签文件。"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from nlplab.classification.textcnn import TextCNN, explain_batch, set_seed
from nlplab.classification.textcnn_training import TRAINING, train_textcnn
from nlplab.core.io import write_json
from nlplab.features.word_sequence import WordSequenceVocabulary, read_json


def load_encoding(path):
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    meta = read_json(path.with_suffix(".json"))
    return {**arrays, **meta}


def run_job(job):
    # 所有路径由离线 controller 给定；final 模式明确拒绝 val/test 输入。
    mode = job["mode"]
    if mode in ("validation", "final"):
        if "test" in job or (mode == "final" and "validation" in job):
            raise ValueError("训练 job 禁止 Test，final 禁止 Validation")
        train = load_encoding(Path(job["train"]))
        val = load_encoding(Path(job["validation"])) if mode == "validation" else None
        output = Path(job["output"])
        output.mkdir(parents=True, exist_ok=False)

        def progress(epoch, row):
            val_note = f" val_f1={row['validation_macro_f1']:.4f}" if val is not None else ""
            print(
                f"{mode} lr={job['learning_rate']} seed={job['seed']} epoch={epoch}"
                f" loss={row['train_loss']:.4f}{val_note}",
                flush=True,
            )

        model, result = train_textcnn(
            train,
            job["vocabulary_size"],
            job["learning_rate"],
            job["seed"],
            TRAINING,
            val,
            job.get("final_epochs"),
            progress,
        )
        model.save(
            output / "model",
            {
                "learning_rate": job["learning_rate"],
                "seed": job["seed"],
                "epochs": result["best_epoch"],
                "mode": mode,
            },
        )
        write_json(output / "training_result.json", result)
    elif mode == "infer":
        set_seed(42)
        vocab = WordSequenceVocabulary.load(Path(job["vocabulary"]))
        encoded = load_encoding(Path(job["encoding"]))
        model = TextCNN.load(Path(job["model"]))
        results = []
        start_time = perf_counter()
        for start in range(0, len(encoded["ids"]), 32):
            batch = {
                key: value[start : start + 32]
                for key, value in encoded.items()
                if key in ("ids", "lengths", "tokens", "original_positions")
            }
            results.extend(explain_batch(model, batch, vocab))
        if "runtime_output" in job:
            write_json(
                Path(job["runtime_output"]),
                {
                    "inference_and_explanation_seconds": perf_counter() - start_time,
                    "batch_size": 32,
                    "document_count": len(results),
                },
            )
        write_json(Path(job["output"]), results)
    else:
        raise ValueError("未知神经 job")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    run_job(read_json(args.job))


if __name__ == "__main__":
    main()
