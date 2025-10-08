"""
Enhanced Prompt Templates for Advanced Query Processing

This module extends the basic prompt engineering with domain-specific templates,
few-shot examples, and chain-of-thought prompting for improved response quality.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from .prompts import ProcessedQuery, QueryIntent


class PromptTemplate(Enum):
    """Advanced prompt template types."""
    BASIC = "basic"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    DOMAIN_SPECIFIC = "domain_specific"


@dataclass
class EnhancedPromptConfig:
    """Configuration for enhanced prompt generation."""
    template_type: PromptTemplate = PromptTemplate.BASIC
    include_examples: bool = True
    max_context_length: int = 2000
    domain: Optional[str] = None
    reasoning_steps: bool = False


class AdvancedPromptEngine:
    """Advanced prompt engineering with domain-specific templates and examples."""
    
    def __init__(self, config: Optional[EnhancedPromptConfig] = None):
        self.config = config or EnhancedPromptConfig()
        self.domain_templates = self._load_domain_templates()
        self.few_shot_examples = self._load_few_shot_examples()
    
    def _load_domain_templates(self) -> Dict[str, Dict[str, str]]:
        """Load domain-specific prompt templates."""
        return {
            "ai_ml": {
                "definition": """You are an AI/ML expert. Provide a clear, technical definition of {concept}.
                Include:
                1. Core concept explanation
                2. Key characteristics
                3. Common applications
                4. Relationship to related concepts
                
                Context: {context}
                Question: {query}
                
                Provide a comprehensive but accessible explanation:""",
                
                "explanation": """As an AI/ML expert, explain {concept} step by step.
                Break down:
                1. How it works (methodology)
                2. Why it's important (significance)
                3. When to use it (applications)
                4. Common challenges and solutions
                
                Context: {context}
                Question: {query}
                
                Detailed explanation:""",
                
                "comparison": """Compare and contrast {concepts} from an AI/ML perspective.
                Structure your response:
                1. Brief overview of each concept
                2. Key similarities
                3. Key differences
                4. Use case scenarios for each
                5. Recommendation guidelines
                
                Context: {context}
                Question: {query}
                
                Comparative analysis:"""
            },
            
            "research": {
                "definition": """As a research expert, define {concept} with academic rigor.
                Include:
                1. Formal definition
                2. Historical context
                3. Current state of research
                4. Key researchers/papers
                
                Context: {context}
                Question: {query}
                
                Academic definition:""",
                
                "summarization": """Summarize the research on {topic} based on the provided context.
                Structure:
                1. Key findings
                2. Methodologies used
                3. Current gaps
                4. Future directions
                
                Context: {context}
                Question: {query}
                
                Research summary:"""
            }
        }
    
    def _load_few_shot_examples(self) -> Dict[str, List[Dict[str, str]]]:
        """Load few-shot examples for different query types."""
        return {
            "definition": [
                {
                    "query": "What is machine learning?",
                    "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions based on those patterns."
                },
                {
                    "query": "What is deep learning?",
                    "response": "Deep learning is a specialized form of machine learning that uses artificial neural networks with multiple layers to learn complex patterns in data. It mimics the way human brains process information and has been particularly successful in tasks like image recognition, natural language processing, and speech recognition."
                }
            ],
            
            "explanation": [
                {
                    "query": "How does gradient descent work?",
                    "response": "Gradient descent works by iteratively adjusting model parameters to minimize a cost function. It calculates the gradient (slope) of the cost function with respect to each parameter, then moves in the opposite direction of the gradient. This process continues until the algorithm converges to a minimum, optimizing the model's performance."
                }
            ],
            
            "comparison": [
                {
                    "query": "Compare supervised vs unsupervised learning",
                    "response": "Supervised learning uses labeled training data to learn mappings from inputs to outputs, enabling predictions on new data. Examples include classification and regression. Unsupervised learning finds hidden patterns in unlabeled data without predefined outcomes. Examples include clustering and dimensionality reduction. Supervised learning is goal-directed, while unsupervised learning is exploratory."
                }
            ]
        }
    
    def create_enhanced_prompt(
        self, 
        context: str, 
        query: str, 
        processed_query: ProcessedQuery
    ) -> str:
        """Create an enhanced prompt using advanced techniques."""
        
        # Determine domain and template
        domain = self._detect_domain(processed_query)
        template_type = self.config.template_type
        
        if template_type == PromptTemplate.FEW_SHOT:
            return self._create_few_shot_prompt(context, query, processed_query)
        elif template_type == PromptTemplate.CHAIN_OF_THOUGHT:
            return self._create_cot_prompt(context, query, processed_query)
        elif template_type == PromptTemplate.DOMAIN_SPECIFIC:
            return self._create_domain_prompt(context, query, processed_query, domain)
        else:
            return self._create_basic_enhanced_prompt(context, query, processed_query)
    
    def _detect_domain(self, processed_query: ProcessedQuery) -> str:
        """Detect the domain of the query for specialized prompts."""
        ml_keywords = {"learning", "neural", "algorithm", "model", "training", "ai", "artificial"}
        research_keywords = {"study", "research", "paper", "findings", "methodology"}
        
        query_words = set(processed_query.keywords + processed_query.entities)
        
        if query_words & ml_keywords:
            return "ai_ml"
        elif query_words & research_keywords:
            return "research"
        else:
            return "general"
    
    def _create_few_shot_prompt(
        self, 
        context: str, 
        query: str, 
        processed_query: ProcessedQuery
    ) -> str:
        """Create a few-shot prompt with examples."""
        intent_str = processed_query.intent.value
        examples = self.few_shot_examples.get(intent_str, [])
        
        prompt_parts = [
            "You are an expert assistant. Answer questions accurately based on the provided context.",
            "Here are some examples of good responses:\\n"
        ]
        
        # Add examples
        for i, example in enumerate(examples[:2]):  # Limit to 2 examples
            prompt_parts.append(f"Example {i+1}:")
            prompt_parts.append(f"Q: {example['query']}")
            prompt_parts.append(f"A: {example['response']}\\n")
        
        # Add current query
        prompt_parts.extend([
            "Now answer this question using the provided context:",
            f"Context: {context}",
            f"Question: {query}",
            "Answer:"
        ])
        
        return "\\n".join(prompt_parts)
    
    def _create_cot_prompt(
        self, 
        context: str, 
        query: str, 
        processed_query: ProcessedQuery
    ) -> str:
        """Create a chain-of-thought prompt for step-by-step reasoning."""
        return f"""You are an expert assistant. Answer the question step by step, showing your reasoning.

Context: {context}

Question: {query}

Let me think through this step by step:

Step 1: First, I'll identify the key concepts in the question
Step 2: Then, I'll find relevant information in the context
Step 3: Next, I'll organize the information logically
Step 4: Finally, I'll provide a comprehensive answer

Let me work through this:"""
    
    def _create_domain_prompt(
        self, 
        context: str, 
        query: str, 
        processed_query: ProcessedQuery,
        domain: str
    ) -> str:
        """Create a domain-specific prompt."""
        templates = self.domain_templates.get(domain, {})
        intent_template = templates.get(processed_query.intent.value)
        
        if intent_template:
            # Extract key concepts for template
            concepts = " and ".join(processed_query.entities[:3]) if processed_query.entities else "the topic"
            
            return intent_template.format(
                concept=concepts,
                concepts=concepts,
                topic=concepts,
                context=context,
                query=query
            )
        else:
            return self._create_basic_enhanced_prompt(context, query, processed_query)
    
    def _create_basic_enhanced_prompt(
        self, 
        context: str, 
        query: str, 
        processed_query: ProcessedQuery
    ) -> str:
        """Create an enhanced version of the basic prompt."""
        intent_instructions = {
            QueryIntent.DEFINITION: "Provide a clear, comprehensive definition with examples.",
            QueryIntent.EXPLANATION: "Explain thoroughly with step-by-step reasoning.",
            QueryIntent.COMPARISON: "Compare systematically, highlighting key differences and similarities.",
            QueryIntent.HOW_TO: "Provide step-by-step instructions with practical examples.",
            QueryIntent.SUMMARIZATION: "Summarize the key points concisely and accurately.",
            QueryIntent.UNKNOWN: "Answer the question accurately and comprehensively."
        }
        
        instruction = intent_instructions.get(processed_query.intent, intent_instructions[QueryIntent.UNKNOWN])
        
        return f"""You are a knowledgeable assistant. {instruction}

Use the following context to inform your response:

Context:
{context}

Question: {query}

Key concepts to address: {', '.join(processed_query.entities[:5])}

Provide a detailed, accurate response:"""


# Backward compatibility and integration
def create_enhanced_prompt(
    context: str, 
    query: str, 
    processed_query: ProcessedQuery,
    config: Optional[EnhancedPromptConfig] = None
) -> str:
    """Create an enhanced prompt with advanced techniques."""
    engine = AdvancedPromptEngine(config)
    return engine.create_enhanced_prompt(context, query, processed_query)