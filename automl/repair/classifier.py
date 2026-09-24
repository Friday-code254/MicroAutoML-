"""
MicroAutoML-Agent — Failure Classifier
Deterministically maps exceptions and tracebacks to known FailureTypes.
"""

from __future__ import annotations

import re

from automl.core.enums import FailureType


class FailureClassifier:
    """Classifies runtime errors into actionable FailureType enums."""

    def __init__(self) -> None:
        self.rules = [
            # Memory
            (r"(?i)memoryerror|out of memory|allocat(e|ion).*failed", FailureType.MEMORY),
            
            # Shape / Dimension mismatch / Bad features
            (r"(?i)shape|dimension.*mismatch|found array with dim|expected \d+d.*got \d+d|column names are not columns", FailureType.SHAPE),
            
            # Type / Categorical handling
            (r"(?i)could not convert string to float|unsupported.*type.*string|unseen label", FailureType.TYPE),
            
            # Numerical (NaN/Inf)
            (r"(?i)infinity or a value too large|input contains nan|inf|infinity", FailureType.NUMERICAL),
            
            # Syntax / Import (User or LLM error in dynamic code)
            (r"(?i)syntaxerror|indentationerror|nameerror|not defined", FailureType.SYNTAX),
            (r"(?i)importerror|modulenotfounderror|no module named", FailureType.IMPORT),
            
            # Timeout
            (r"(?i)time out|timed out|took too long|TimeoutError", FailureType.TIMEOUT),
            
            # Generalization
            (r"(?i)all targets are constant|contains only one class", FailureType.GENERALIZATION),
        ]

    def classify(self, exception_type: str, error_message: str, traceback_str: str) -> FailureType:
        """
        Given the error details, applies regex rules to determine the FailureType.
        """
        # Combine everything for regex scanning
        full_text = f"{exception_type}: {error_message}\n{traceback_str}"
        
        # Exact match on exception type first
        if exception_type == "MemoryError":
            return FailureType.MEMORY
        elif exception_type == "TimeoutError":
            return FailureType.TIMEOUT
        elif exception_type in ("SyntaxError", "IndentationError", "NameError"):
            return FailureType.SYNTAX
        elif exception_type in ("ImportError", "ModuleNotFoundError"):
            return FailureType.IMPORT
            
        # Regex scanning
        for pattern, failure_type in self.rules:
            if re.search(pattern, full_text):
                return failure_type
                
        # Default
        return FailureType.UNKNOWN
