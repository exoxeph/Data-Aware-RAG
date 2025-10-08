"""Query planning and routing DAG for multi-modal retrieval."""

from dataclasses import dataclass

# Keywords that suggest table/numeric data is needed
CUE_TABLE = ("auc", "f1", "accuracy", "95%", "n=", "p<", "confidence interval")

# Keywords that suggest figures/diagrams are needed  
CUE_FIG = ("architecture", "workflow", "pipeline", "framework", "diagram", "schema", 
           "encoder", "decoder", "module")


@dataclass
class Plan:
    """Query execution plan specifying which modalities to search."""
    use_text: bool = True
    use_tables: bool = False
    use_figures: bool = False


def plan_query(q: str) -> Plan:
    """Analyze query and create execution plan for retrieval.
    
    Args:
        q: User query string
        
    Returns:
        Plan object specifying which modalities to search
    """
    ql = q.lower()
    
    return Plan(
        use_text=True,  # Always search text
        use_tables=any(c in ql for c in CUE_TABLE),
        use_figures=any(c in ql for c in CUE_FIG)
    )