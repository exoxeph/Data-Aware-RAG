"""Unit test for marker command building."""

from rag_papers.ingest.marker_run import build_marker_cmd


def test_build_marker_cmd_with_pagination():
    """Test that marker command includes pagination when enabled."""
    cmd = build_marker_cmd(
        pdf_path="/path/to/paper.pdf",
        out_dir="/output",
        fmt="markdown",
        paginate=True
    )
    
    assert "marker_single" in cmd
    assert "/path/to/paper.pdf" in cmd
    assert "--output_format" in cmd
    assert "markdown" in cmd
    assert "--output_dir" in cmd
    assert "/output" in cmd
    assert "--paginate_output" in cmd


def test_build_marker_cmd_without_pagination():
    """Test that marker command excludes pagination when disabled."""
    cmd = build_marker_cmd(
        pdf_path="/path/to/paper.pdf",
        out_dir="/output",
        fmt="json",
        paginate=False
    )
    
    assert "marker_single" in cmd
    assert "/path/to/paper.pdf" in cmd
    assert "--output_format" in cmd
    assert "json" in cmd
    assert "--output_dir" in cmd
    assert "/output" in cmd
    assert "--paginate_output" not in cmd


def test_build_marker_cmd_default_pagination():
    """Test that pagination defaults to True."""
    cmd = build_marker_cmd(
        pdf_path="/path/to/paper.pdf",
        out_dir="/output",
        fmt="markdown"
    )
    
    assert "--paginate_output" in cmd