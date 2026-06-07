import streamlit as st
import numpy as np
from backend import rag_chain, METRICS, TRACE_LOGS

st.set_page_config(page_title="RAG Dashboard", layout="wide")

st.title("RAG Observability Dashboard")

query = st.text_input("Enter your query", "What is Machine Learning?")

if st.button("Run Pipeline"):

    with st.spinner("Running RAG pipeline..."):
        ans = rag_chain(query)

    st.success("Done!")

    # ---------------- METRICS ----------------
    p50 = np.percentile(METRICS["latencies"], 50) if METRICS["latencies"] else 0
    p95 = np.percentile(METRICS["latencies"], 95) if METRICS["latencies"] else 0
    avg_cost = np.mean(METRICS["costs"]) if METRICS["costs"] else 0
    avg_quality = np.mean(METRICS["quality"]) if METRICS["quality"] else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("p50 Latency", f"{p50:.3f}s")
    col2.metric("p95 Latency", f"{p95:.3f}s")
    col3.metric("Avg Cost", f"${avg_cost:.6f}")
    col4.metric("Quality", f"{avg_quality*100:.1f}%")

    # ---------------- ANSWER ----------------
    st.subheader("Answer")
    st.write(ans)

    # ---------------- TRACE ----------------
    st.subheader("Latest Trace")
    if TRACE_LOGS:
        st.json(TRACE_LOGS[-1])