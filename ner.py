"""Character-span NER, strict evaluation and local masking."""

import re
from collections import Counter


def validate_spans(text, spans):
    out = sorted(set((int(a), int(b), str(c)) for a, b, c in spans))
    if any(not 0 <= a < b <= len(text) for a, b, _ in out):
        raise ValueError("Invalid character span")
    return out


def strict_metrics(gold, predicted):
    if len(gold) != len(predicted):
        raise ValueError("Mismatched document counts")
    counts = Counter()
    by_type = {}
    for gs, ps in zip(gold, predicted):
        g, p = set(map(tuple, gs)), set(map(tuple, ps))
        for kind in {s[2] for s in g | p}:
            gt = {s for s in g if s[2] == kind}
            pr = {s for s in p if s[2] == kind}
            c = by_type.setdefault(kind, Counter())
            c.update(tp=len(gt & pr), fp=len(pr - gt), fn=len(gt - pr))
        counts.update(tp=len(g & p), fp=len(p - g), fn=len(g - p))

    def scores(c):
        p = c["tp"] / max(c["tp"] + c["fp"], 1)
        r = c["tp"] / max(c["tp"] + c["fn"], 1)
        return dict(c, precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-15))

    return {
        "micro": scores(counts),
        "per_type": {k: scores(v) for k, v in sorted(by_type.items())},
    }


def decode(offsets, tags):
    spans = []
    current = None
    for (a, b), tag in zip(offsets, tags):
        if a == b:
            continue
        if tag == "O":
            if current:
                spans.append(tuple(current))
                current = None
            continue
        prefix, kind = tag.split("-", 1)
        if current and prefix == "I" and current[2] == kind:
            current[1] = b
        else:
            if current:
                spans.append(tuple(current))
            current = [a, b, kind]
    if current:
        spans.append(tuple(current))
    return spans


def token_labels(offsets, spans, label2id):
    labels = []
    previous = None
    for a, b in offsets:
        if a == b:
            labels.append(-100)
            previous = None
            continue
        match = next((i for i, (s, e, c) in enumerate(spans) if s <= a < b <= e), None)
        if match is None:
            labels.append(0)
        else:
            prefix = "I-" if previous == match else "B-"
            labels.append(label2id[prefix + spans[match][2]])
        previous = match
    return labels


def regex_spans(text):
    patterns = [
        (r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])", "Email"),
        (
            r"(?<!\d)(?:\+7|8)[ ()-]*\d{3}[ ()-]*\d{3}[ -]*\d{2}[ -]*\d{2}(?!\d)",
            "Номер телефона",
        ),
    ]
    return sorted(
        (m.start(), m.end(), label)
        for pattern, label in patterns
        for m in re.finditer(pattern, text)
    )


def merge_spans(spans):
    selected = []
    for s in sorted(set(map(tuple, spans)), key=lambda x: (-(x[1] - x[0]), x[0], x[2])):
        if all(s[1] <= t[0] or t[1] <= s[0] for t in selected):
            selected.append(s)
    return sorted(selected)


def mask(text, spans):
    # Union intervals so nested or crossing spans cannot expose a covered fragment.
    intervals = []
    for a, b, _ in validate_spans(text, spans):
        if intervals and a <= intervals[-1][1]:
            intervals[-1][1] = max(b, intervals[-1][1])
        else:
            intervals.append([a, b])
    for a, b in reversed(intervals):
        text = text[:a] + "[REDACTED]" + text[b:]
    return text
