"""Linter module for OpenHands ACI.

Part of this Linter module is adapted from Aider (Apache 2.0 License, [original code](https://github.com/paul-gauthier/aider/blob/main/aider/linter.py)). Please see the [original repository](https://github.com/paul-gauthier/aider) for more information.
"""

from .base import LintResult
from .linter import DefaultLinter

__all__ = ['DefaultLinter', 'LintResult']
