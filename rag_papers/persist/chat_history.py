"""
Chat history persistence for multi-turn conversations.

Stores conversation messages as JSON with timestamps for session replay.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ChatMessage:
    """Single chat message."""
    
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        """Create from dict."""
        return cls(**data)


@dataclass
class ChatHistory:
    """
    Manages chat history for a session.
    
    Persists messages to JSON file for durability and replay.
    """
    
    path: Path
    messages: list[ChatMessage] = field(default_factory=list)
    
    def __post_init__(self):
        """Load existing history if file exists."""
        if self.path.exists():
            self.load()
    
    def add(self, role: str, content: str, metadata: dict = None) -> ChatMessage:
        """
        Add a message to history and persist.
        
        Args:
            role: "user" or "assistant"
            content: Message text
            metadata: Optional metadata (sources, cache info, etc.)
        
        Returns:
            The created ChatMessage
        """
        msg = ChatMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.messages.append(msg)
        self.save()
        return msg
    
    def save(self) -> None:
        """Persist history to JSON file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "messages": [msg.to_dict() for msg in self.messages],
            "updated_at": time.time()
        }
        
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self) -> list[ChatMessage]:
        """Load history from JSON file."""
        if not self.path.exists():
            self.messages = []
            return self.messages
        
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.messages = [
                ChatMessage.from_dict(msg_data)
                for msg_data in data.get("messages", [])
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Corrupted file - start fresh
            self.messages = []
        
        return self.messages
    
    def clear(self) -> None:
        """Clear all messages and delete file."""
        self.messages = []
        if self.path.exists():
            self.path.unlink()
    
    def get_context(self, max_messages: int = 10) -> str:
        """
        Get recent messages as context string.
        
        Args:
            max_messages: Maximum number of recent messages to include
        
        Returns:
            Formatted conversation history
        """
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        
        lines = []
        for msg in recent:
            role_label = msg.role.capitalize()
            lines.append(f"{role_label}: {msg.content}")
        
        return "\n".join(lines)
    
    def get_recent_messages(self, count: int = 5) -> list[ChatMessage]:
        """Get N most recent messages."""
        return self.messages[-count:] if count < len(self.messages) else self.messages
    
    def __len__(self) -> int:
        """Return number of messages."""
        return len(self.messages)


def create_session_history(session_id: str, base_dir: Path = None) -> ChatHistory:
    """
    Create a ChatHistory for a session.
    
    Args:
        session_id: Unique session identifier
        base_dir: Base directory for chat history files (defaults to runs/)
    
    Returns:
        ChatHistory instance
    """
    if base_dir is None:
        base_dir = Path("runs")
    
    history_path = base_dir / f"session_{session_id}" / "chat_history.json"
    return ChatHistory(path=history_path)
