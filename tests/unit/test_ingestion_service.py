"""
Tests for the ingestion service layer using TDD approach.
"""
import base64
import pytest
from pathlib import Path
from typing import List

# Import types and classes that will be needed
from rag_papers.ingest.models import Chunk
from rag_papers.ingest.service import IngestionService, IngestionError
from rag_papers.ingest.parsers import MarkerParser, PlumberParser


# Base64 encoded minimal PDF with "Hello RAG" text
SAMPLE_PDF_BASE64 = """JVBERi0xLjQKMSAwIG9iago8PC9QYWdlcyAyIDAgUi9UeXBlL0NhdGFsb2c+PgplbmRvYmoKMiAwIG9iago8PC9UeXBlL1BhZ2VzL0tpZHNbMyAwIFJdL0NvdW50IDE+PgplbmRvYmoKMyAwIG9iago8PC9UeXBlL1BhZ2UvUGFyZW50IDIgMCBSL01lZGlhQm94WzAgMCA1OTUuMjggODQxLjg5XS9Db250ZW50cyA0IDAgUi9SZXNvdXJjZXM8PC9Gb250PDwvRjEgNSAwIFI+Pj4+PgplbmRvYmoKNCAwIG9iago8PC9MZW5ndGggNjQ+PnN0cmVhbQpCVC9GIDEyIFRmCjEwMCA3MDAgVGQKKChIZWxsbyBSQUcpKSBUagplbmRzdHJlYW0KZW5kb2JqCjUgMCBvYmoKPDwvVHlwZS9Gb250L1N1YnR5cGUvVHlwZTEvTmFtZS9GMS9CYXNlRm9udC9IZWx2ZXRpY2EvRW5jb2RpbmcvV2luQW5zaUVuY29kaW5nPj4KZW5kb2JqCnhyZWYKMCA2CjAwMDAwMDAwMDAgNjU1MzUgZgowMDAwMDAwMDkzIDAwMDAwIG4KMDAwMDAwMDE2OCAwMDAwMCBuCjAwMDAwMDAyNjUgMDAwMDAgbgowMDAwMDAwMzk4IDAwMDAwIG4KMDAwMDAwMDUyMyAwMDAwMCBuCnRyYWlsZXIKPDwvUm9vdCAxIDAgUi9TaXplIDY+PgpzdGFydHhyZWYKNTQ3CiUlRU9G"""


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a sample PDF file from base64 data for testing."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_data = base64.b64decode(SAMPLE_PDF_BASE64)
    pdf_path.write_bytes(pdf_data)
    return pdf_path


@pytest.fixture
def ingestion_service() -> IngestionService:
    """Create an IngestionService instance for testing."""
    return IngestionService()


def test_ingestion_marker_happy_path(sample_pdf_path: Path, ingestion_service: IngestionService, monkeypatch):
    """Test successful PDF ingestion using marker parser."""
    # Monkeypatch MarkerParser.parse() to return mock result with enough text to pass quality check
    def mock_marker_parse(self, pdf_path: Path) -> List[dict]:
        # Create text with enough content to pass quality threshold (>= 375 chars for 0.25 score)
        good_text = "Hello RAG text content that is long enough to pass the quality threshold. " * 10
        return [{"page": 1, "text": good_text}]
    
    monkeypatch.setattr(MarkerParser, "parse", mock_marker_parse)
    
    # Call the ingestion service
    chunks = ingestion_service.ingest_pdf(sample_pdf_path)
    
    # Assertions
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
    assert "Hello RAG" in chunk.content
    assert chunk.metadata["file"] == str(sample_pdf_path)
    assert chunk.metadata["page"] == 1
    assert "chunk_id" in chunk.metadata


def test_ingestion_fallback_to_plumber_when_quality_low(
    sample_pdf_path: Path, ingestion_service: IngestionService, monkeypatch
):
    """Test fallback to plumber parser when marker quality is low."""
    # Monkeypatch MarkerParser to return low quality (very short) text
    def mock_marker_parse_low_quality(self, pdf_path: Path) -> List[dict]:
        return [{"page": 1, "text": "x"}]  # Very short/garbage text
    
    # Monkeypatch PlumberParser to return good fallback text
    def mock_plumber_parse(self, pdf_path: Path) -> List[dict]:
        return [{"page": 1, "text": "Fallback RAG text"}]
    
    monkeypatch.setattr(MarkerParser, "parse", mock_marker_parse_low_quality)
    monkeypatch.setattr(PlumberParser, "parse", mock_plumber_parse)
    
    # Call the ingestion service
    chunks = ingestion_service.ingest_pdf(sample_pdf_path)
    
    # Assertions - should use plumber result
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "Fallback RAG text" in chunk.content
    assert chunk.metadata["parser"] == "plumber"  # Should indicate fallback was used


def test_chunking_max_words(sample_pdf_path: Path, ingestion_service: IngestionService, monkeypatch):
    """Test that long text gets split into multiple chunks respecting max_words limit."""
    # Create a very long text (3000 words)
    long_text = " ".join([f"word{i}" for i in range(3000)])
    
    # Monkeypatch MarkerParser to return long text
    def mock_marker_parse_long(self, pdf_path: Path) -> List[dict]:
        return [{"page": 1, "text": long_text}]
    
    monkeypatch.setattr(MarkerParser, "parse", mock_marker_parse_long)
    
    # Call the ingestion service
    chunks = ingestion_service.ingest_pdf(sample_pdf_path)
    
    # Assertions
    assert len(chunks) > 1  # Should be split into multiple chunks
    
    # Check each chunk respects max_words limit (assuming 512 words max)
    for chunk in chunks:
        word_count = len(chunk.content.split())
        assert word_count <= 512
        
        # Check metadata is preserved across chunks
        assert chunk.metadata["file"] == str(sample_pdf_path)
        assert chunk.metadata["page"] == 1
        assert "chunk_id" in chunk.metadata
    
    # Verify all chunks together contain the original content
    combined_content = " ".join(chunk.content for chunk in chunks)
    # Should contain most of the original words (allowing for some splitting variations)
    assert len(combined_content.split()) >= 2900


def test_corrupted_pdf_raises_clean_error(tmp_path: Path, ingestion_service: IngestionService):
    """Test that corrupted/non-PDF files raise appropriate errors."""
    # Create a non-PDF file
    corrupted_file = tmp_path / "not_a_pdf.txt"
    corrupted_file.write_text("This is not a PDF file")
    
    # Attempt to ingest the corrupted file
    with pytest.raises(IngestionError) as exc_info:
        ingestion_service.ingest_pdf(corrupted_file)
    
    # Verify the error message is helpful
    error_message = str(exc_info.value)
    assert "PDF" in error_message or "corrupted" in error_message or "invalid" in error_message
    assert str(corrupted_file) in error_message