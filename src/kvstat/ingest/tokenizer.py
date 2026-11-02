from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from kvstat.errors import ConfigError


def load_decoder(name: str) -> Callable[[tuple[int, ...]], str]:
    """Load a model's tokenizer and return a function that turns token ids into text.

    The function drops special tokens such as chat template markers. Only tokenizer.json is
    read, so no code from the model repository runs.

    Args:
        name: A tokenizer.json file, a directory that holds one, or a Hugging Face model id.

    Raises:
        ConfigError: The tokenizer cannot be loaded.
    """
    # Imported here so that kvstat without --tokenizer starts as fast as before.
    from tokenizers import Tokenizer

    path = Path(name)
    try:
        if path.is_dir():
            tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        elif path.is_file() or path.suffix == ".json":
            tokenizer = Tokenizer.from_file(name)
        else:
            tokenizer = Tokenizer.from_pretrained(name)
    except Exception as exc:  # tokenizers raises a bare Exception
        raise ConfigError(f"cannot load tokenizer {name}: {exc}") from exc

    def decode(ids: tuple[int, ...]) -> str:
        return tokenizer.decode(list(ids), skip_special_tokens=True)

    return decode
