import streamlit as st
import re
import time
import sys
import os

# ---------------------------
# Add main.py folder to sys.path
# ---------------------------
sys.path.append(os.path.dirname(__file__))

# Import workflow objects from main.py
from main import HRState, graph, END  # make sure main.py defines these

# ============================================
# ⚙️ Streamlit Config
# ============================================
st.set_page_config(
    page_title="HR Match-O-Matic 🪩",
    page_icon="💼",
    layout="wide",
)

# ============================================
# 🌌 Custom Dark Aesthetic
# ============================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #0D0C1D 0%, #1A1A2E 50%, #16213E 100%);
    color: #EAEAEA;
}
[data-testid="stSidebar"] {
    background: rgba(20, 20, 35, 0.75);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}
textarea {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #EAEAEA !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}
.stButton>button {
    background: linear-gradient(90deg, #8E2DE2, #4A00E0);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    box-shadow: 0 0 12px rgba(138, 43, 226, 0.4);
    transition: 0.25s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(138, 43, 226, 0.8);
}
h1, h2, h3 {
    font-family: "Poppins", sans-serif;
    color: #E0E0FF;
}
div[data-testid="stMetricValue"] {
    color: #00FFFF;
    font-weight: 700;
}
.log-box {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 6px;
    border-left: 3px solid #8E2DE2;
}
.agent-status {
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 6px 10px;
    margin: 3px 0;
    font-size: 0.9rem;
}
.agent-active { color: #00FFFF; }
.agent-done { color: #8E2DE2; }
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background: #4A00E0; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ============================================
# 🧠 Sidebar
# ============================================
with st.sidebar:
    st.markdown("## ⚙️ HR Screening Settings")
    st.markdown("🧠 **LLM:** Groq LLaMA-3.1-8B-Instant")
    st.markdown("📊 **LangSmith Project:** HR_MultiAgent_Tracing")
    st.markdown("---")
    st.info("🚀 Powered by LangGraph + LangSmith + Groq", icon="⚙️")
    st.caption("“Hiring, but make it *vibes only* 💜”")

# ============================================
# 🌌 Header
# ============================================
st.title("💼 HR Match-O-Matic: Night Mode 🌙")
st.markdown("""
Welcome to your **AI-driven HR Screening Dashboard**.  
Drop in a resume and JD — our multi-agent squad will handle the rest ⚡
""")

# ============================================
# ✍️ Input Fields
# ============================================
col1, col2 = st.columns(2)
with col1:
    resume_input = st.text_area("📄 Candidate Resume", height=220, placeholder="Paste resume content here...")
with col2:
    jd_input = st.text_area("📋 Job Description (JD)", height=220, placeholder="Paste JD content here...")

# ============================================
# 🚀 Run Workflow
# ============================================
if st.button("⚡ Run AI Screening"):
    if not resume_input.strip() or not jd_input.strip():
        st.warning("⚠️ Both Resume and JD are required")
    else:
        st.markdown("### 🧩 Agents Booting Up...")
        status_placeholder = st.empty()

        agent_progress = {
            "Supervisor": "⏳",
            "ResumeParser": "⏳",
            "JDAnalyzer": "⏳",
            "MatchScorer": "⏳"
        }

        def show_agent_status():
            status_html = ""
            for agent, symbol in agent_progress.items():
                color_class = "agent-active" if symbol == "⚙️" else "agent-done" if symbol == "✅" else ""
                status_html += f"<div class='agent-status {color_class}'><b>{symbol} {agent}</b></div>"
            status_placeholder.markdown(status_html, unsafe_allow_html=True)

        show_agent_status()
        time.sleep(1)

        with st.spinner("🧠 Multi-agent workflow running..."):
            state: HRState = {
                "resume": resume_input,
                "jd": jd_input,
                "result": "",
                "next_agent": "supervisor",
                "step_completed": []
            }
            step_logs = []

            while state.get("next_agent") != END:
                current_agent = state["next_agent"].capitalize()
                agent_progress[current_agent] = "⚙️"
                show_agent_status()
                time.sleep(0.8)

                response = graph.invoke(state, config={"thread_id": "hr_workflow_ui"})
                state.update(response)

                agent_progress[current_agent] = "✅"
                show_agent_status()
                time.sleep(0.5)

                for msg in response.get("messages", []):
                    step_logs.append({"agent": current_agent, "message": msg.content})

        if step_logs:
            st.subheader("💬 Agent Conversations")
            for step in step_logs:
                st.markdown(f"<div class='log-box'><b>{step['agent']}</b>: {step['message']}</div>", unsafe_allow_html=True)

        if state.get("result"):
            st.subheader("🎉 Final Verdict")
            st.success(state["result"])

            score_match = re.search(r"Match Score\s*\(0–100\)\s*:\s*(\d+)", state["result"])
            if score_match:
                score = int(score_match.group(1))
                st.metric(label="🔥 Match Score", value=f"{score}/100")
                st.progress(score / 100)

            st.balloons()
