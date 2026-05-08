"""Mapping of equipment names to parser classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dataset.parsers.base import BaseParser

PARSER_REGISTRY: dict[str, type[BaseParser]] = {}


def register(*names: str):
    """Class decorator that registers a parser under one or more equipment names."""

    def decorator(cls: type[BaseParser]) -> type[BaseParser]:
        for name in names:
            PARSER_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_parser(equipment: str) -> BaseParser:
    """Instantiate the parser registered for *equipment*."""
    key = equipment.strip().lower()
    if key not in PARSER_REGISTRY:
        available = ", ".join(sorted(PARSER_REGISTRY))
        raise ValueError(
            f"No parser registered for equipment '{equipment}'. "
            f"Available: [{available}]"
        )
    return PARSER_REGISTRY[key]()


# Import concrete parsers so their @register decorators execute.
import src.dataset.parsers.sunrise as _sunrise
import src.dataset.parsers.biospa as _biospa
