"""Engine package: re-exports ``Tabularium`` and ``ToolError``."""

from tabularium.engine._errors import ToolError
from tabularium.engine.core import Tabularium

__all__ = ["Tabularium", "ToolError"]
