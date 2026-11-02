from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from kvstat.errors import ConfigError
from kvstat.ingest.tokenizer import load_decoder

VOCAB = {"[UNK]": 0, "Where": 1, "is": 2, "my": 3, "order": 4}
TEMPLATE_START = 5


def write_tokenizer(path: Path) -> Path:
    tokenizer = Tokenizer(WordLevel(VOCAB, unk_token="[UNK]"))  # noqa: S106 - a vocabulary entry, not a secret
    tokenizer.add_special_tokens(["<|im_start|>"])
    tokenizer.save(str(path))
    return path


def test_a_tokenizer_file_decodes_without_special_tokens(tmp_path):
    decode = load_decoder(str(write_tokenizer(tmp_path / "tokenizer.json")))
    assert decode((TEMPLATE_START, 1, 2, 3, 4)) == "Where is my order"


def test_a_directory_with_a_tokenizer_file_loads(tmp_path):
    write_tokenizer(tmp_path / "tokenizer.json")
    assert load_decoder(str(tmp_path))((1, 2)) == "Where is"


def test_a_tokenizer_that_cannot_load_says_which_one(tmp_path):
    broken = tmp_path / "tokenizer.json"
    broken.write_text("{}")
    with pytest.raises(ConfigError, match=f"cannot load tokenizer {broken}"):
        load_decoder(str(broken))


def test_a_missing_tokenizer_file_is_named_as_a_missing_file(tmp_path):
    missing = tmp_path / "tokenizer.json"
    with pytest.raises(ConfigError, match="No such file"):
        load_decoder(str(missing))
