"""
Memory write policies.

Determines when to create memory notes based on conversation quality.
"""

from typing import Dict, Any, List
from .schemas import MemoryWritePolicy


class WritePolicy:
    """
    Policy engine for memory note creation.
    
    Policies:
    - never: Disabled, no automatic writes
    - conservative: Write only for high-quality, verified answers
    - aggressive: Write for all accepted answers
    """
    
    def __init__(self, policy: MemoryWritePolicy = "conservative"):
        """
        Initialize write policy.
        
        Args:
            policy: Policy mode
        """
        self.policy = policy
    
    def should_write(
        self,
        turn_data: Dict[str, Any],
        verify_score: float,
        cfg: Any = None
    ) -> bool:
        """
        Determine if a memory should be written.
        
        Args:
            turn_data: Turn information (query, answer, sources, etc.)
            verify_score: Verification score from verifier
            cfg: Configuration object (for thresholds)
        
        Returns:
            True if memory should be written
        """
        if self.policy == "never":
            return False
        
        # Extract verification info
        accepted = turn_data.get("accepted", False)
        if not accepted:
            return False  # Never write rejected answers
        
        # Get threshold from config
        threshold = getattr(cfg, "accept_threshold", 0.72) if cfg else 0.72
        
        if self.policy == "aggressive":
            # Write for any accepted answer
            return verify_score >= threshold
        
        elif self.policy == "conservative":
            # Additional checks for conservative mode
            
            # Must be well above threshold
            if verify_score < threshold + 0.05:
                return False
            
            # Must cite at least one source
            sources = turn_data.get("sources", [])
            if not sources or len(sources) < 1:
                return False
            
            # Check for explicit memory trigger phrases
            query = turn_data.get("query", "").lower()
            answer = turn_data.get("answer", "").lower()
            
            memory_triggers = [
                "remember",
                "note that",
                "keep in mind",
                "for future reference",
                "recall",
                "don't forget"
            ]
            
            has_trigger = any(trigger in query or trigger in answer for trigger in memory_triggers)
            
            # Conservative: needs either trigger OR very high score
            if has_trigger or verify_score >= threshold + 0.15:
                return True
            
            return False
        
        return False
    
    def should_summarize(
        self,
        history_length: int,
        summarize_every: int = 4
    ) -> bool:
        """
        Determine if conversation should be summarized.
        
        Args:
            history_length: Number of turns in history
            summarize_every: Threshold for summarization
        
        Returns:
            True if summarization should occur
        """
        if self.policy == "never":
            return False
        
        return history_length > summarize_every
    
    def extract_tags(self, text: str) -> List[str]:
        """
        Extract tags from text based on content patterns.
        
        Args:
            text: Memory or query text
        
        Returns:
            List of suggested tags
        """
        tags = []
        text_lower = text.lower()
        
        # Concept tags
        concept_patterns = {
            "dropout": "concept:dropout",
            "attention": "concept:attention",
            "transformer": "concept:transformer",
            "optimization": "concept:optimization",
            "regularization": "concept:regularization",
            "fine-tuning": "concept:fine-tuning",
            "transfer learning": "concept:transfer-learning",
        }
        
        for pattern, tag in concept_patterns.items():
            if pattern in text_lower:
                tags.append(tag)
        
        # Entity tags (models, techniques)
        if "llama" in text_lower or "gpt" in text_lower or "bert" in text_lower:
            tags.append("entity:model")
        
        if "temperature" in text_lower or "top_k" in text_lower or "top_p" in text_lower:
            tags.append("setting:generation")
        
        # Task tags
        task_patterns = {
            "setup": "task:setup",
            "configure": "task:configuration",
            "train": "task:training",
            "evaluate": "task:evaluation",
            "test": "task:testing",
        }
        
        for pattern, tag in task_patterns.items():
            if pattern in text_lower:
                tags.append(tag)
        
        # Preference tags
        if any(word in text_lower for word in ["prefer", "like", "want", "need"]):
            tags.append("preference:user")
        
        return list(set(tags))  # Remove duplicates
