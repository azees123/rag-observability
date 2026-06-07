import streamlit as st
import numpy as np
import pandas as pd
from backend import rag_chain, METRICS, TRACE_LOGS

# ---------------- CONFIGURATION ----------------
st.set_page_config(page_title="RAG Dashboard", layout="wide")
st.title("📊 RAG Observability Dashboard")

if "user_history" not in st.session_state:
    st.session_state["user_history"] = []
if "session_latencies" not in st.session_state:
    st.session_state["session_latencies"] = []
if "session_costs" not in st.session_state:
    st.session_state["session_costs"] = []
if "session_quality" not in st.session_state:
    st.session_state["session_quality"] = []

# ---------------- USER INPUT ----------------
query = st.text_input("Enter your query", placeholder="What is Machine Learning?")

if st.button("Run Pipeline", type="primary"):
    if query.strip():
        with st.spinner("Running RAG pipeline..."):
            ans = rag_chain(query)
            latest_global_trace = TRACE_LOGS[-1]
            
            st.session_state["session_latencies"].append(latest_global_trace["latency"])
            st.session_state["session_costs"].append(latest_global_trace["cost"])
            st.session_state["session_quality"].append(latest_global_trace["quality"])
            
            st.session_state["user_history"].append({
                "query": query,
                "answer": ans,
                "trace": latest_global_trace
            })
        st.success("Execution complete!")

# ---------------- INTERFACE TABS ----------------
tab_session, tab_global, tab_traces = st.tabs([
    "👤 Your Active Session", 
    "🌐 Global Telemetry (All Users)", 
    "🕵️‍♂️ Detailed Step Traces"
])

# ---------------- SESSION WORKING TAB ----------------
with tab_session:
    if not st.session_state["user_history"]:
        st.info("Run a query above to populate your session statistics.")
    else:
        l_arr = st.session_state["session_latencies"]
        p50 = np.percentile(l_arr, 50)
        p95 = np.percentile(l_arr, 95)
        avg_cost = np.mean(st.session_state["session_costs"])
        avg_quality = np.mean(st.session_state["session_quality"])

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Your p50 Latency", f"{p50:.3f}s")
        m_col2.metric("Your p95 Latency", f"{p95:.3f}s")
        m_col3.metric("Your Avg Cost", f"${avg_cost:.6f}")
        m_col4.metric("Your Avg Quality", f"{avg_quality*100:.1f}%")

        st.markdown("---")
        st.subheader("💬 Chat Context Timeline")
        for chat in reversed(st.session_state["user_history"]):
            with st.chat_message("user"):
                st.write(chat["query"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])

# ---------------- GLOBAL METRICS TAB ----------------
with tab_global:
    if not TRACE_LOGS:
        st.info("No global analytics compiled yet.")
    else:
        st.subheader("System Performance Across All Active Sessions")
        
        g_p50 = np.percentile(METRICS["latencies"], 50)
        g_p95 = np.percentile(METRICS["latencies"], 95)
        g_cost = np.sum(METRICS["costs"])
        g_qual = np.mean(METRICS["quality"])

        g_col1, g_col2, g_col3, g_col4 = st.columns(4)
        g_col1.metric("Global p50", f"{g_p50:.3f}s")
        g_col2.metric("Global p95", f"{g_p95:.3f}s")
        g_col3.metric("Total Cumulative Cost", f"${g_cost:.5f}")
        g_col4.metric("Global Avg Quality", f"{g_qual*100:.1f}%")

        st.markdown("---")
        st.subheader("📉 Production Latency Trend")
        chart_data = pd.DataFrame({"Latency (seconds)": METRICS["latencies"]})
        st.line_chart(chart_data)

# ---------------- LOWER-LEVEL TRACES TAB ----------------
with tab_traces:
    st.subheader("Deep-Dive Execution Tree Inspection")
    if not st.session_state["user_history"]:
        st.info("Execute pipelines to analyze low-level trace JSON keys.")
    else:
        for i, item in enumerate(reversed(st.session_state["user_history"])):
            run_idx = len(st.session_state["user_history"]) - i
            trace_payload = item["trace"]
            
            with st.expander(f"🔍 Run #{run_idx} | Trace ID: {trace_payload.get('trace_id', 'N/A')}"):
                st.json(trace_payload)
