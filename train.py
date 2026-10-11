"""Train on local annotated TSV; no raw text is exported."""

from pathlib import Path
import argparse, ast, copy, hashlib, json, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ner import (
    validate_spans,
    strict_metrics,
    token_labels,
    decode,
    regex_spans,
    merge_spans,
)


def encode(tokenizer, text):
    return tokenizer(
        text,
        max_length=256,
        truncation=True,
        stride=64,
        padding="max_length",
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
    )


def infer(model, tokenizer, texts):
    model.eval()
    result = []
    with torch.inference_mode():
        for text in texts:
            e = encode(tokenizer, text)
            spans = []
            for start in range(0, len(e["input_ids"]), 16):
                batch = {
                    k: torch.tensor(e[k][start : start + 16])
                    for k in ["input_ids", "attention_mask"]
                }
                ids = model(**batch).logits.argmax(-1).tolist()
                for j, tags in enumerate(ids):
                    spans.extend(
                        decode(
                            e["offset_mapping"][start + j],
                            [model.config.id2label[x] for x in tags],
                        )
                    )
            result.append(merge_spans(spans))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("results"))
    ap.add_argument("--model", default="cointegrated/rubert-tiny2")
    ap.add_argument("--revision", default="e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    df = pd.read_csv(args.data, sep="\t")
    records = {}
    conflicts = set()
    for row in df.itertuples():
        spans = validate_spans(row.text, ast.literal_eval(row.target))
        key = hashlib.sha256(row.text.encode()).hexdigest()
        if key in records and records[key]["spans"] != spans:
            conflicts.add(key)
        records[key] = {"text": row.text, "spans": spans, "sha256": key}
    rows = [r for k, r in sorted(records.items()) if k not in conflicts]
    rng = np.random.default_rng(42)
    rng.shuffle(rows)
    a = int(0.7 * len(rows))
    b = int(0.85 * len(rows))
    train, valid, test = rows[:a], rows[a:b], rows[b:]
    labels = sorted({c for r in train for _, _, c in r["spans"]})
    label2id = {"O": 0}
    for c in labels:
        for prefix in ["B-", "I-"]:
            label2id[prefix + c] = len(label2id)
    manifest = []
    for part, rs in [("train", train), ("validation", valid), ("test", test)]:
        manifest.extend(
            {"sha256": r["sha256"], "split": part, "entities": len(r["spans"])}
            for r in rs
        )
    pd.DataFrame(manifest).to_csv(out / "split_manifest.csv", index=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        revision=args.revision,
        num_labels=len(label2id),
        label2id=label2id,
        id2label={i: s for s, i in label2id.items()},
    )
    windows = []
    overlap_docs = 0
    for r in train:
        spans = r["spans"]
        if any(y[0] < x[1] for x, y in zip(spans, spans[1:])):
            overlap_docs += 1
            continue
        e = encode(tokenizer, r["text"])
        for i, offsets in enumerate(e["offset_mapping"]):
            windows.append(
                {k: torch.tensor(e[k][i]) for k in ["input_ids", "attention_mask"]}
                | {"labels": torch.tensor(token_labels(offsets, spans, label2id))}
            )
    protocol = {
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "model": args.model,
        "model_revision": args.revision,
        "seed": 42,
        "epochs": args.epochs,
        "learning_rate": 5e-5,
        "batch_size": 16,
        "max_length": 256,
        "stride": 64,
        "split": "70/15/15 by unique exact-text hash, shuffled seed 42",
        "conflicting_documents_excluded": len(conflicts),
        "overlapping_training_documents_excluded": overlap_docs,
        "train_windows": len(windows),
        "labels": labels,
        "selection": "strict validation entity micro-F1",
        "raw_text_exported": False,
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    loader = DataLoader(windows, batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    best = -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            opt.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        preds = infer(model, tokenizer, [r["text"] for r in valid])
        m = strict_metrics([r["spans"] for r in valid], preds)["micro"]
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), **m))
        if m["f1"] > best:
            best = m["f1"]
            model.save_pretrained(out / "model")
            tokenizer.save_pretrained(out / "model")
        print(json.dumps(history[-1]), flush=True)
    model = AutoModelForTokenClassification.from_pretrained(out / "model")
    pred = infer(model, tokenizer, [r["text"] for r in test])
    gold = [r["spans"] for r in test]
    rules = [regex_spans(r["text"]) for r in test]
    result = {
        name: strict_metrics(gold, p)
        for name, p in [
            ("rubert", pred),
            ("regex_email_phone", rules),
            ("hybrid", [merge_spans(x + y) for x, y in zip(pred, rules)]),
        ]
    }
    (out / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "learning_curve.json").write_text(json.dumps(history, indent=2))
    (out / "test_predictions.json").write_text(
        json.dumps(
            [
                {"sha256": r["sha256"], "gold": r["spans"], "predicted": p}
                for r, p in zip(test, pred)
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: v["micro"] for k, v in result.items()}), flush=True)


if __name__ == "__main__":
    main()
