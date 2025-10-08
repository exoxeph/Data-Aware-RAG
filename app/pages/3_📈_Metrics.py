"""
Metrics Page - View evaluation telemetry and reports.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import glob

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.state import get_state
from app.components import metric_card


def find_latest_run() -> Path | None:
    """Find the most recent run directory."""
    run_dirs = glob.glob("runs/run_*")
    if not run_dirs:
        return None
    
    # Sort by modification time
    run_dirs.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return Path(run_dirs[0])


def load_run_summary(run_dir: Path) -> dict:
    """Load summary.md and extract metrics."""
    summary_file = run_dir / "summary.md"
    if not summary_file.exists():
        return {}
    
    metrics = {}
    with open(summary_file, "r") as f:
        content = f.read()
        
        # Parse metrics table (simple extraction)
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "Accept@1" in line and i + 1 < len(lines):
                # Extract from table format
                parts = lines[i].split("|")
                if len(parts) >= 3:
                    try:
                        metrics["accept_at_1"] = parts[2].strip()
                    except:
                        pass
            elif "Avg Score" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        metrics["avg_score"] = parts[2].strip()
                    except:
                        pass
            elif "Repair Rate" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        metrics["repair_rate"] = parts[2].strip()
                    except:
                        pass
            elif "Total Queries" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        metrics["total_queries"] = parts[2].strip()
                    except:
                        pass
            elif "P50" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        metrics["p50_latency"] = parts[2].strip()
                    except:
                        pass
            elif "P95" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        metrics["p95_latency"] = parts[2].strip()
                    except:
                        pass
            elif "Avg Context Chars" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        metrics["avg_context_chars"] = parts[2].strip()
                    except:
                        pass
    
    return metrics


def load_results_df(run_dir: Path) -> pd.DataFrame | None:
    """Load results from CSV or Parquet."""
    csv_file = run_dir / "results.csv"
    parquet_file = run_dir / "results.parquet"
    
    if parquet_file.exists():
        try:
            return pd.read_parquet(parquet_file)
        except Exception as e:
            st.warning(f"Could not load parquet: {e}")
    
    if csv_file.exists():
        try:
            return pd.read_csv(csv_file)
        except Exception as e:
            st.warning(f"Could not load CSV: {e}")
    
    return None


def main():
    st.title("📈 Metrics & Telemetry")
    st.markdown("View evaluation results and performance metrics.")
    
    state = get_state()
    
    # Run selector
    st.markdown("### 📁 Select Run")
    
    all_runs = sorted(glob.glob("runs/run_*"), key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    if not all_runs:
        st.info("No evaluation runs found. Run `poetry run rag-eval` to create evaluation data.")
        st.stop()
    
    run_names = [Path(r).name for r in all_runs]
    selected_run = st.selectbox(
        "Run",
        options=run_names,
        index=0,
        help="Select an evaluation run to view"
    )
    
    run_dir = Path("runs") / selected_run
    
    # Load data
    with st.spinner("Loading run data..."):
        metrics = load_run_summary(run_dir)
        df = load_results_df(run_dir)
    
    # Display metrics
    if metrics:
        st.markdown("### 📊 Summary Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Accept@1",
                metrics.get("accept_at_1", "N/A"),
                help="Fraction of queries with accepted answers"
            )
        
        with col2:
            st.metric(
                "Avg Score",
                metrics.get("avg_score", "N/A"),
                help="Mean verification score"
            )
        
        with col3:
            st.metric(
                "Repair Rate",
                metrics.get("repair_rate", "N/A"),
                help="Fraction of queries that triggered repair"
            )
        
        with col4:
            st.metric(
                "Total Queries",
                metrics.get("total_queries", "N/A"),
                help="Number of queries evaluated"
            )
        
        st.markdown("### ⚡ Latency")
        
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.metric("P50", metrics.get("p50_latency", "N/A"))
        
        with col6:
            st.metric("P95", metrics.get("p95_latency", "N/A"))
        
        with col7:
            st.metric("Avg Context", metrics.get("avg_context_chars", "N/A"))
    else:
        st.warning("Could not parse metrics from summary.md")
    
    # Results table
    if df is not None:
        st.markdown("---")
        st.markdown("### 📋 Detailed Results")
        
        # Show basic stats
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            if "accepted" in df.columns:
                accepted_count = df["accepted"].sum()
                st.metric("Accepted", f"{accepted_count}/{len(df)}")
        
        with col_s2:
            if "verify_score" in df.columns:
                avg_score = df["verify_score"].mean()
                st.metric("Avg Score", f"{avg_score:.3f}")
        
        with col_s3:
            if "repair_used" in df.columns:
                repair_count = df["repair_used"].sum()
                st.metric("Repairs", f"{repair_count}/{len(df)}")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["All Results", "Worst Cases", "Best Cases"])
        
        with tab1:
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        with tab2:
            if "verify_score" in df.columns:
                worst = df.nsmallest(3, "verify_score")
                st.markdown("**Bottom 3 by Verify Score:**")
                for idx, row in worst.iterrows():
                    with st.expander(f"Score: {row['verify_score']:.3f} - {row.get('query', 'N/A')[:60]}"):
                        st.markdown(f"**Query:** {row.get('query', 'N/A')}")
                        st.markdown(f"**Answer:** {row.get('answer', 'N/A')[:300]}...")
                        st.markdown(f"**Intent:** {row.get('intent', 'N/A')}")
                        st.markdown(f"**Accepted:** {row.get('accepted', False)}")
                        st.markdown(f"**Repair Used:** {row.get('repair_used', False)}")
            else:
                st.info("verify_score column not found")
        
        with tab3:
            if "verify_score" in df.columns:
                best = df.nlargest(3, "verify_score")
                st.markdown("**Top 3 by Verify Score:**")
                for idx, row in best.iterrows():
                    with st.expander(f"Score: {row['verify_score']:.3f} - {row.get('query', 'N/A')[:60]}"):
                        st.markdown(f"**Query:** {row.get('query', 'N/A')}")
                        st.markdown(f"**Answer:** {row.get('answer', 'N/A')[:300]}...")
                        st.markdown(f"**Intent:** {row.get('intent', 'N/A')}")
                        st.markdown(f"**Accepted:** {row.get('accepted', True)}")
                        st.markdown(f"**Repair Used:** {row.get('repair_used', False)}")
            else:
                st.info("verify_score column not found")
    else:
        st.info("No results CSV/Parquet found for this run")
    
    # DuckDB section (optional)
    st.markdown("---")
    st.markdown("### 🦆 DuckDB Telemetry")
    
    duckdb_path = Path("data/telemetry.duckdb")
    if duckdb_path.exists():
        try:
            import duckdb
            
            conn = duckdb.connect(str(duckdb_path), read_only=True)
            
            # Query for run list
            try:
                runs_df = conn.execute("""
                    SELECT DISTINCT run_id, COUNT(*) as queries
                    FROM eval_results
                    GROUP BY run_id
                    ORDER BY run_id DESC
                    LIMIT 10
                """).fetchdf()
                
                st.markdown("**Recent Runs in DuckDB:**")
                st.dataframe(runs_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.warning(f"Could not query DuckDB: {e}")
            
            conn.close()
            
        except Exception as e:
            st.error(f"DuckDB error: {e}")
    else:
        st.info(
            "DuckDB telemetry not found. "
            "Run evaluations with `--duckdb data/telemetry.duckdb` to enable."
        )
    
    # Raw files view
    with st.expander("📄 Raw Files"):
        st.markdown(f"**Run Directory:** `{run_dir}`")
        
        files = list(run_dir.glob("*"))
        if files:
            for file in files:
                st.text(f"• {file.name} ({file.stat().st_size:,} bytes)")
        else:
            st.info("No files found")


if __name__ == "__main__":
    main()
