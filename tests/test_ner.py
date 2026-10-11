import pytest
from ner import strict_metrics, decode, token_labels, mask, validate_spans


def test_strict_offsets_and_type():
    m = strict_metrics([[(0, 4, "PERSON")]], [[(0, 3, "PERSON")]])["micro"]
    assert m["tp"] == 0 and m["fp"] == 1 and m["fn"] == 1


def test_outside_tokens_do_not_inflate_score():
    assert strict_metrics([[], [(0, 3, "X")]], [[], []])["micro"]["f1"] == 0


def test_adjacent_same_type_entities_remain_separate():
    assert decode([(0, 2), (3, 5)], ["B-X", "B-X"]) == [(0, 2, "X"), (3, 5, "X")]


def test_subword_labels_continue_entity():
    assert token_labels(
        [(0, 0), (0, 2), (2, 5)], [(0, 5, "X")], {"O": 0, "B-X": 1, "I-X": 2}
    ) == [-100, 1, 2]


def test_union_masking():
    assert mask("abcdefgh", [(1, 4, "X"), (3, 7, "Y")]) == "a[REDACTED]h"


def test_invalid_span_rejected():
    with pytest.raises(ValueError):
        validate_spans("abc", [(0, 4, "X")])
