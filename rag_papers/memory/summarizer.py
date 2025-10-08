"""
Memory summarization using LLM or mock generator.

Converts conversation turns into concise memory notes.
"""

from typing import List, Dict, Optional
from rag_papers.generation.generator import BaseGenerator, GenerationConfig
from .schemas import MemoryNote, MemoryScope
from .policy import WritePolicy
import uuid
import time


class MemorySummarizer:
    """
    Summarizes conversation turns into memory notes.
    
    Uses LLM (or mock) to create factual, concise memories.
    """
    
    def __init__(self, generator: BaseGenerator, policy: Optional[WritePolicy] = None):
        """
        Initialize summarizer.
        
        Args:
            generator: LLM generator for summarization
            policy: Write policy (for tag extraction)
        """
        self.generator = generator
        self.policy = policy or WritePolicy()
    
    def build_summary(
        self,
        history: List[Dict[str, str]],
        target_chars: int = 800
    ) -> str:
        """
        Summarize conversation turns into compact text.
        
        Args:
            history: List of {role, content} dicts
            target_chars: Target character count
        
        Returns:
            Summary text (bullet points)
        """
        if not history:
            return ""
        
        # Build context from history
        context = "\n".join([
            f"{turn.get('role', 'user').capitalize()}: {turn.get('content', '')}"
            for turn in history[-10:]  # Last 10 turns max
        ])
        
        # Create summarization prompt
        prompt = f"""You are a note-taker. Summarize the user–assistant conversation into factual, verifiable notes helpful for future questions about the same topic.

Requirements:
- 3–6 bullet points, neutral tone
- Keep to ≤ {target_chars} characters total
- Do NOT include speculation or personal info
- Prefer definitions, resolved conclusions, selected parameters, and key entities

Conversation:
{context}

Summary (bullet points):"""
        
        # Generate summary
        config = GenerationConfig(
            temperature=0.1,  # Low temperature for factual output
            max_new_tokens=min(200, target_chars // 3)  # Conservative token limit
        )
        
        try:
            response = self.generator.generate(prompt, config)
            summary = response.text.strip()
            
            # Truncate if needed
            if len(summary) > target_chars:
                summary = summary[:target_chars-3] + "..."
            
            return summary
            
        except Exception as e:
            # Fallback: extract key phrases from turns
            return self._extract_key_phrases(history, target_chars)
    
    def _extract_key_phrases(
        self,
        history: List[Dict[str, str]],
        target_chars: int
    ) -> str:
        """
        Fallback: Extract key phrases when LLM unavailable.
        
        Args:
            history: Conversation turns
            target_chars: Target length
        
        Returns:
            Pseudo-summary from key terms
        """
        phrases = []
        
        for turn in history[-5:]:  # Last 5 turns
            content = turn.get("content", "")
            role = turn.get("role", "user")
            
            # Extract sentences with key terms
            sentences = content.split(".")
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                
                # Look for definition patterns
                if any(pattern in sent.lower() for pattern in ["is a", "refers to", "means", "defined as"]):
                    phrases.append(f"• {sent}")
                
                # Look for preference/setting patterns
                elif any(pattern in sent.lower() for pattern in ["prefer", "use", "set to", "configured"]):
                    if role == "user":
                        phrases.append(f"• {sent}")
            
            # Limit total length
            current_length = sum(len(p) for p in phrases)
            if current_length > target_chars:
                break
        
        summary = "\n".join(phrases[:6])  # Max 6 bullets
        
        if len(summary) > target_chars:
            summary = summary[:target_chars-3] + "..."
        
        return summary if summary else "• Discussed technical concepts."
    
    def to_memory(
        self,
        turn_data: Dict[str, str],
        history: List[Dict[str, str]],
        scope: MemoryScope = "session",
        scope_key: str = "default"
    ) -> Optional[MemoryNote]:
        """
        Convert a turn (or history) into a memory note.
        
        Args:
            turn_data: Current turn {query, answer, sources, ...}
            history: Previous turns for context
            scope: Memory scope
            scope_key: Scope identifier
        
        Returns:
            MemoryNote or None
        """
        # Extract key info
        query = turn_data.get("query", "")
        answer = turn_data.get("answer", "")
        
        # Build memory text from recent context
        if len(history) > 3:
            # Use summary for longer conversations
            memory_text = self.build_summary(history + [{"role": "user", "content": query}], target_chars=300)
        else:
            # For short exchanges, extract directly
            memory_text = self._extract_memory_from_turn(query, answer)
        
        if not memory_text or len(memory_text) < 10:
            return None
        
        # Extract tags
        tags = self.policy.extract_tags(memory_text)
        
        # Add source info to tags
        sources = turn_data.get("sources", [])
        if sources:
            tags.append(f"sources:{len(sources)}")
        
        # Create memory note
        note = MemoryNote(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            scope=scope,
            scope_key=scope_key,
            text=memory_text,
            tags=tags,
            source="chat",
            created_at=time.time(),
            last_used_at=time.time(),
            uses=0,
            meta={
                "query": query[:100],
                "source_count": len(sources),
                "turn_timestamp": time.time()
            }
        )
        
        return note
    
    def _extract_memory_from_turn(self, query: str, answer: str) -> str:
        """
        Extract memory-worthy content from a single Q&A turn.
        
        Args:
            query: User query
            answer: Assistant answer
        
        Returns:
            Concise memory text
        """
        # Look for definition pattern in answer
        answer_lower = answer.lower()
        
        # Extract first sentence if it's a definition
        sentences = answer.split(".")
        if sentences and len(sentences[0]) < 200:
            first_sent = sentences[0].strip()
            if any(pattern in first_sent.lower() for pattern in ["is a", "refers to", "means"]):
                return f"• {first_sent}."
        
        # Extract key fact (first 150 chars of answer)
        if len(answer) > 150:
            return f"• {answer[:147]}..."
        
        return f"• {answer}"
    
    def summarize_with_context(
        self,
        history: List[Dict[str, str]],
        scope: MemoryScope,
        scope_key: str,
        target_chars: int = 800
    ) -> List[MemoryNote]:
        """
        Summarize entire conversation into multiple memory notes.
        
        Args:
            history: Full conversation history
            scope: Memory scope
            scope_key: Scope identifier
            target_chars: Target chars per note
        
        Returns:
            List of MemoryNote objects
        """
        if len(history) < 2:
            return []
        
        # Build summary
        summary_text = self.build_summary(history, target_chars)
        
        if not summary_text:
            return []
        
        # Split into bullet points
        bullets = [line.strip() for line in summary_text.split("\n") if line.strip()]
        
        notes = []
        for bullet in bullets:
            # Remove bullet marker
            text = bullet.lstrip("•-*").strip()
            
            if len(text) < 10:
                continue
            
            # Extract tags
            tags = self.policy.extract_tags(text)
            tags.append("summary")
            
            note = MemoryNote(
                id=f"mem_{uuid.uuid4().hex[:12]}",
                scope=scope,
                scope_key=scope_key,
                text=text,
                tags=tags,
                source="chat",
                created_at=time.time(),
                last_used_at=time.time(),
                uses=0,
                meta={
                    "history_length": len(history),
                    "summary_timestamp": time.time()
                }
            )
            
            notes.append(note)
        
        return notes
