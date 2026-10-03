"""纯 PyTorch 训练进程；不读原文/split，不依赖 sklearn 或 Web。"""

import copy
from time import perf_counter

import numpy as np
import torch

from nlplab.classification.stage3_config import TRAINING
from nlplab.classification.textcnn import TextCNN, set_seed


def validation_macro_f1(gold, predicted):
    """仅 early stopping 标量；完整指标由原 classification_metrics 统一计算并核验。"""
    matrix = np.bincount(np.asarray(gold) * 5 + np.asarray(predicted), minlength=25).reshape(5, 5)
    denom = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(
        np.mean(
            np.divide(2 * np.diag(matrix), denom, out=np.zeros(5, dtype=float), where=denom != 0)
        )
    )


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_score, self.best_epoch, self.bad_epochs = -1.0, 0, 0

    def update(self, score, epoch):
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_epoch, self.bad_epochs = score, epoch, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def epoch_train(model, x, lengths, y, optimizer, generator, config):
    model.train()
    loss_sum = correct = 0
    order = torch.randperm(len(y), generator=generator)
    for indices in order.split(config["batch_size"]):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x[indices], lengths[indices])
        loss = torch.nn.functional.cross_entropy(logits, y[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
        optimizer.step()
        loss_sum += float(loss.detach()) * len(indices)
        correct += int((logits.argmax(1) == y[indices]).sum())
    return loss_sum / len(y), correct / len(y)


def epoch_evaluate(model, x, lengths, y, batch_size):
    model.eval()
    loss_sum, predicted = 0.0, []
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            stop = start + batch_size
            logits = model(x[start:stop], lengths[start:stop])
            loss_sum += float(
                torch.nn.functional.cross_entropy(logits, y[start:stop], reduction="sum")
            )
            predicted.extend(logits.argmax(1).tolist())
    return loss_sum / len(y), predicted


def train_textcnn(
    train,
    vocabulary_size,
    learning_rate,
    seed,
    config=TRAINING,
    validation=None,
    final_epochs=None,
    progress=None,
):
    if (validation is None) == (final_epochs is None):
        raise ValueError("仅允许有 Validation 的选参训练或无 Validation 的定长 final fit")
    set_seed(seed, config["threads"])
    model = TextCNN(vocabulary_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config["weight_decay"]
    )
    x, lengths, y = (torch.as_tensor(train[name]) for name in ("ids", "lengths", "labels"))
    generator = torch.Generator().manual_seed(seed)
    history, stopping, best_state, best_prediction = (
        [],
        EarlyStopping(config["patience"]),
        None,
        None,
    )
    start = perf_counter()
    limit = config["max_epochs"] if validation is not None else final_epochs
    if validation is not None:
        vx, vl, vy = (torch.as_tensor(validation[n]) for n in ("ids", "lengths", "labels"))
    for epoch in range(1, limit + 1):
        epoch_start = perf_counter()
        loss, accuracy = epoch_train(model, x, lengths, y, optimizer, generator, config)
        row = {"epoch": epoch, "train_loss": loss, "train_accuracy": accuracy}
        if validation is not None:
            val_loss, prediction = epoch_evaluate(model, vx, vl, vy, config["batch_size"])
            macro = validation_macro_f1(vy.numpy(), prediction)
            row.update(
                validation_loss=val_loss,
                validation_macro_f1=macro,
                validation_prediction_indices=prediction,
            )
            improved, stop = stopping.update(macro, epoch)
            if improved:
                best_state, best_prediction = copy.deepcopy(model.state_dict()), prediction
        else:
            stop = False
        row["runtime_seconds"] = perf_counter() - epoch_start
        history.append(row)
        if progress:
            progress(epoch, row)
        if stop:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {
        "learning_rate": learning_rate,
        "seed": seed,
        "epochs_executed": len(history),
        "best_epoch": stopping.best_epoch if validation is not None else final_epochs,
        "best_validation_macro_f1": stopping.best_score if validation is not None else None,
        "best_prediction_indices": best_prediction,
        "history": history,
        "runtime_seconds": perf_counter() - start,
        "early_stopping_used": validation is not None,
        "parameter_count": model.parameter_count,
    }
