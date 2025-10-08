"""
Memory subsystem for long-term conversation recall.

Provides:
- Conversation summarization
- Memory note storage (session/corpus/global scope)
- Semantic recall and injection into RAG context
- Write policies (conservative/aggressive)
"""

from .schemas import MemoryNote, MemoryQuery, MemoryScope, MemoryWritePolicy, MemoryMetadata
from .store import MemoryStore
from .recall import MemoryRecall
from .summarizer import MemorySummarizer
from .policy import WritePolicy
from .integrate import MemoryIntegration, create_memory_integration

__all__ = [
    "MemoryNote",
    "MemoryQuery",
    "MemoryScope",
    "MemoryWritePolicy",
    "MemoryMetadata",
    "MemoryStore",
    "MemoryRecall",
    "MemorySummarizer",
    "WritePolicy",
    "MemoryIntegration",
    "create_memory_integration",
]
