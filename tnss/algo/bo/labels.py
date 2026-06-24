"""
labels.py — readable run identities for BOSS.

A run's label is `{kind}-{acqf}-{word}`: the surrogate kind ('clas' / 'reg'), the
acquisition name ('ei' / 'lcb' / 'tmse' / 'cucb' / 'sur' / 'gsur'), and a
docker-style adjective-noun word derived deterministically from the seed. So runs
that share a seed share a word (easy to spot as comparable), and the same config is
always reproducible — replacing opaque hex ids with something you can say out loud,
e.g. `reg-cucb-white-monkey`.

The kind/name are read off the surrogate/acquisition (a `kind` / `name` class
attribute), so each component names itself; this module only assembles them.
"""
from __future__ import annotations

import hashlib

ADJECTIVES = (
    "amber", "azure", "bold", "brave", "bright", "brisk", "calm", "clever",
    "cosmic", "crisp", "dapper", "eager", "frosty", "fuzzy", "gentle", "golden",
    "jolly", "keen", "lively", "lucky", "mellow", "merry", "nimble", "plucky",
    "quiet", "rapid", "rustic", "sage", "sandy", "sleek", "snowy", "solar",
    "spry", "stout", "sunny", "swift", "tidy", "vivid", "witty", "zesty",
)
NOUNS = (
    "almond", "badger", "banana", "cactus", "cedar", "cobra", "comet", "dragon",
    "ember", "falcon", "ferret", "fjord", "garnet", "gecko", "harbor", "heron",
    "ibis", "jaguar", "juniper", "kelp", "koala", "lemur", "lynx", "maple",
    "marmot", "monkey", "narwhal", "ocelot", "otter", "panda", "quokka", "raven",
    "salmon", "tapir", "urchin", "viper", "walnut", "walrus", "yak", "zebra",
)


def _pick(items: tuple[str, ...], key: str, salt: str) -> str:
    """Deterministic choice from `items` keyed by `key` (process-stable, unlike the
    built-in `hash`). `salt` decorrelates the adjective and noun draws."""
    digest = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    return items[int(digest, 16) % len(items)]


def word(seed: int) -> str:
    """Deterministic adjective-noun word for a seed, e.g. word(1) -> 'brave-otter'."""
    return f"{_pick(ADJECTIVES, str(seed), 'adj')}-{_pick(NOUNS, str(seed), 'noun')}"


def make_label(surrogate, acquisition, seed: int) -> str:
    """`{kind}-{acqf}-{word}` for a (surrogate, acquisition, seed) composition.

    surrogate, acquisition : the composed components; their `kind` / `name` class
        attributes name them (falling back to the lowercased class name).
    seed : the run seed — sets the word.
    """
    kind = getattr(surrogate, "kind", type(surrogate).__name__.lower())
    name = getattr(acquisition, "name", type(acquisition).__name__.lower())
    return f"{kind}-{name}-{word(seed)}"
