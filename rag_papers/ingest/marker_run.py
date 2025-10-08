"""Marker-pdf integration and command building."""


def build_marker_cmd(pdf_path: str, out_dir: str, fmt: str, paginate: bool = True) -> list[str]:
    """Build marker command for PDF processing.
    
    Args:
        pdf_path: Path to input PDF file
        out_dir: Output directory for processed files
        fmt: Output format (markdown, json, etc.)
        paginate: Whether to enable pagination in output
        
    Returns:
        List of command arguments for marker_single
    """
    cmd = ["marker_single", pdf_path, "--output_format", fmt, "--output_dir", out_dir]
    if paginate:
        cmd.append("--paginate_output")
    return cmd