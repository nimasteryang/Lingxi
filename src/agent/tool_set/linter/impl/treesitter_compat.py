"""Compatibility layer for tree-sitter 0.24.0."""

import importlib

from tree_sitter import Language, Parser

# Cache of loaded languages
_language_cache = {}


def get_parser(language):
    """Get a Parser object for the given language name."""
    if language not in _language_cache:
        # Try to import the language module
        module_name = f'tree_sitter_{language}'
        try:
            module = importlib.import_module(module_name)
            _language_cache[language] = Language(module.language())
        except ImportError:
            raise ValueError(
                f'Language {language} is not supported. Please install {module_name} package.'
            )

    return Parser(_language_cache[language])
