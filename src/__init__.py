"""src package.

Keep this module import-safe: avoid importing submodules with side effects
at package import time (Azure Functions indexes by importing the app).
"""

__all__ = []
