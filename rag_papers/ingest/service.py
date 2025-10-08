from __future__ import annotations
from typing import List
from dataclasses import dataclass
from pathlib import Path
import os

from .models import Chunk
from .parsers import MarkerParser, PlumberParser, Parser
from .quality import is_low_quality
from .chunker import normalize, chunk_text


class IngestionError(Exception):
    """Exception raised when PDF ingestion fails."""
    pass


@dataclass
class IngestionConfig:
    """Configuration for the ingestion service."""
    prefer: tuple[str, ...] = ("marker", "plumber")
    max_words: int = 512
    quality_threshold: float = 0.25
    marker_timeout_s: int = 60


class IngestionService:
    """Service for ingesting PDF files into chunks."""
    
    def __init__(
        self,
        cfg: IngestionConfig | None = None,
        marker: Parser | None = None,
        plumber: Parser | None = None
    ):
        self.cfg = cfg or IngestionConfig()
        self.marker = marker or MarkerParser(timeout_s=self.cfg.marker_timeout_s)
        self.plumber = plumber or PlumberParser()
    
    def ingest_pdf(self, pdf_path: Path | str) -> List[Chunk]:
        """
        Ingest a PDF file and return a list of chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of text chunks with metadata
            
        Raises:
            IngestionError: If the PDF cannot be processed
        """
        pdf_path = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path
        
        # Validate file exists
        if not pdf_path.exists() or not pdf_path.is_file():
            raise IngestionError(f"Not a file: {pdf_path}")
        
        # Try parsers in preferred order
        pages = None
        errors = []
        parser_used = None
        
        for name in self.cfg.prefer:
            parser = self.marker if name == "marker" else self.plumber
            
            try:
                pages = parser.parse(pdf_path)
                
                # Basic validation
                if not isinstance(pages, list):
                    raise ValueError("parser returned non-list")
                
                # For marker parser, check quality and fallback if poor
                if name == "marker":
                    all_text = "\n".join(p.get("text", "") or "" for p in pages)
                    if is_low_quality(all_text, self.cfg.quality_threshold):
                        # Mark as low quality and continue to next parser
                        errors.append((name, "Low quality text detected, falling back"))
                        continue
                
                parser_used = name
                break
                
            except Exception as e:
                errors.append((name, str(e)))
                pages = None
        
        if not pages:
            raise IngestionError(f"Could not parse PDF {pdf_path}; tried {errors}")
        
        # Build chunks
        file_name = pdf_path.name
        chunks: List[Chunk] = []
        chunk_id = 0
        
        for page in pages:
            page_no = int(page.get("page", 0)) or 1
            text = normalize(page.get("text", "") or "")
            
            if not text:
                continue
            
            # Split text into chunks
            for piece in chunk_text(text, max_words=self.cfg.max_words):
                metadata = {
                    "file": str(pdf_path),
                    "page": page_no,
                    "chunk_id": chunk_id
                }
                
                # Add parser info if fallback was used
                if parser_used == "plumber":
                    metadata["parser"] = "plumber"
                
                chunks.append(Chunk(
                    content=piece,
                    metadata=metadata
                ))
                chunk_id += 1
        
        return chunks
