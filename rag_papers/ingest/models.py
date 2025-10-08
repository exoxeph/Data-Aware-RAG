from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class Chunk:
    content: str
    metadata: Dict[str, object]
