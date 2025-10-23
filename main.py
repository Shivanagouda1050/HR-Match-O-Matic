# ============================================
# 🧠 Multi-Agent HR Screening Workflow
# ✅ LangGraph + LangSmith + Groq Integration
# ============================================

from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langsmith import traceable
import os

# -----------------------------
# 1️⃣ Load Environment Variables (hardcoded)
# -----------------------------
GROQ_API_KEY = "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"
TAVILY_API_KEY = "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"

# ✅ LangSmith setup
LANGSMITH_TRACING = True
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = "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"
LANGSMITH_PROJECT = "HR_MultiAgent_Tracing"

# Enable LangSmith tracing in code
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

print("✅ Environment and LangSmith tracing configured.\n")

# -----------------------------
# 2️⃣ Initialize LLM (Groq)
# -----------------------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)
print("✅ Groq LLM initialized successfully.\n")

# -----------------------------
# 3️⃣ Define Shared State
# -----------------------------
class HRState(TypedDict):
    resume: str
    jd: str
    result: str
    next_agent: str
    step_completed: list

# -----------------------------
# 4️⃣ Define Agent Functions
# -----------------------------
def supervisor_agent(state: HRState):
    """Decides which agent should run next."""
    if "ResumeParser" not in state["step_completed"]:
        next_agent = "ResumeParser"
        msg = "🧑‍💼 Supervisor: Starting with Resume parsing."
    elif "JDAnalyzer" not in state["step_completed"]:
        next_agent = "JDAnalyzer"
        msg = "🧑‍💼 Supervisor: Resume parsed. Now analyzing JD."
    elif "MatchScorer" not in state["step_completed"]:
        next_agent = "MatchScorer"
        msg = "🧑‍💼 Supervisor: JD analyzed. Now scoring match."
    else:
        next_agent = END
        msg = "✅ Supervisor: All steps completed."

    print(f"[SUPERVISOR] {msg}\n")
    return {"messages": [AIMessage(content=msg)], "next_agent": next_agent}

def resume_parser(state: HRState):
    """Extract key info from resume."""
    print("[RESUME PARSER] Processing resume...\n")
    prompt = f"""
    You are an HR assistant. Extract key information from this resume:
    - Skills
    - Experience (in years)
    - Education
    - Certifications

    Resume:
    {state['resume']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content
    print(f"[RESUME PARSER] Extraction Complete ✅\n")
    return {
        **state,
        "result": result,
        "next_agent": "supervisor",
        "step_completed": state["step_completed"] + ["ResumeParser"]
    }

def jd_analyzer(state: HRState):
    """Analyze job description."""
    print("[JD ANALYZER] Analyzing JD...\n")
    prompt = f"""
    You are an HR assistant. Analyze the job description and extract:
    - Required skills
    - Experience level
    - Educational background

    JD:
    {state['jd']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content
    print(f"[JD ANALYZER] Analysis Complete ✅\n")
    return {
        **state,
        "result": result,
        "next_agent": "supervisor",
        "step_completed": state["step_completed"] + ["JDAnalyzer"]
    }

@traceable(name="Match Scorer")
def match_scorer(state: HRState):
    """Compare resume and JD and give compatibility score."""
    print("[MATCH SCORER] Comparing resume and JD...\n")
    prompt = f"""
    You are an HR assistant. Compare candidate resume and JD.
    Provide:
    - Match Score (0–100)
    - 3 Key Strengths
    - 2 Weaknesses

    Resume: {state['resume']}
    JD: {state['jd']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content
    print(f"[MATCH SCORER] Scoring Complete ✅\n")
    return {
        **state,
        "result": result,
        "next_agent": "supervisor",
        "step_completed": state["step_completed"] + ["MatchScorer"]
    }

# -----------------------------
# 5️⃣ Build Graph
# -----------------------------
workflow = StateGraph(HRState)

workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("ResumeParser", resume_parser)
workflow.add_node("JDAnalyzer", jd_analyzer)
workflow.add_node("MatchScorer", match_scorer)

workflow.set_entry_point("supervisor")

for node in ["supervisor", "ResumeParser", "JDAnalyzer", "MatchScorer"]:
    workflow.add_conditional_edges(
        node,
        lambda s: s.get("next_agent", "supervisor"),
        {
            "supervisor": "supervisor",
            "ResumeParser": "ResumeParser",
            "JDAnalyzer": "JDAnalyzer",
            "MatchScorer": "MatchScorer",
            END: END,
        },
    )

graph = workflow.compile(checkpointer=MemorySaver())

# -----------------------------
# 6️⃣ Utility Function for Frontend
# -----------------------------
def run_hr_workflow(resume_text: str, jd_text: str):
    """
    Runs the multi-agent HR screening workflow and returns:
    - final_result (str)
    - step_logs (list of dicts: [{'agent': str, 'message': str}, ...])
    """
    state: HRState = {
        "resume": resume_text,
        "jd": jd_text,
        "result": "",
        "next_agent": "supervisor",
        "step_completed": []
    }

    step_logs = []

    while state.get("next_agent") != END:
        response = graph.invoke(
            state,
            config={"thread_id": "hr_workflow_001"}
        )
        state.update(response)

        for msg in response.get("messages", []):
            step_logs.append({
                "agent": state.get("next_agent", "Agent"),
                "message": msg.content
            })

    return state["result"], step_logs

# -----------------------------
# 7️⃣ Allow Standalone CLI Run
# -----------------------------
if __name__ == "__main__":
    print("📄 Running HR Workflow in CLI mode.\n")
    resume = input("Paste Resume content:\n> ")
    jd = input("\nPaste Job Description:\n> ")

    final_result, logs = run_hr_workflow(resume, jd)
    print("\n==============================")
    print("✅ FINAL HR MATCH RESULT")
    print("==============================")
    print(final_result)
    print("\n✨ Tracing active on LangSmith project:", LANGSMITH_PROJECT)
