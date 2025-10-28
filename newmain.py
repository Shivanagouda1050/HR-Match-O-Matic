# ============================================
# 🧠 Multi-Agent HR Screening Workflow
# ✅ PDF Support + LangGraph + Groq + LangSmith
# ============================================

from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import pdfplumber
from langsmith import traceable

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

if not GROQ_API_KEY or not LANGSMITH_API_KEY:
    raise ValueError("🚨 Missing GROQ_API_KEY or LANGSMITH_API_KEY in .env file.")

# Enable LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "HR_MultiAgent_Tracing"

# Initialize Groq LLM
llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=GROQ_API_KEY)


# ✅ Shared State Schema
class HRState(TypedDict):
    resume: str
    jd: str
    result: str
    next_agent: str
    step_completed: list


# ✅ PDF → TEXT Extraction
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()


# ✅ Agents
def supervisor_agent(state: HRState):
    if "ResumeParser" not in state["step_completed"]:
        next_agent = "ResumeParser"
    elif "JDAnalyzer" not in state["step_completed"]:
        next_agent = "JDAnalyzer"
    elif "MatchScorer" not in state["step_completed"]:
        next_agent = "MatchScorer"
    else:
        next_agent = END

    return {
        "messages": [AIMessage(content=f"Next agent: {next_agent}")],
        "next_agent": next_agent
    }


def resume_parser(state: HRState):
    prompt = f"""
    Extract structured info:
    - Skills
    - Experience years
    - Education
    Resume:
    {state['resume']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        **state,
        "result": response.content,
        "next_agent": "supervisor",
        "step_completed": state["step_completed"] + ["ResumeParser"]
    }


def jd_analyzer(state: HRState):
    prompt = f"""
    Extract structured info:
    - Required skills
    - Experience level
    Job Description:
    {state['jd']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        **state,
        "result": response.content,
        "next_agent": "supervisor",
        "step_completed": state["step_completed"] + ["JDAnalyzer"]
    }


@traceable(name="Match Scorer")
def match_scorer(state: HRState):
    prompt = f"""
    Compare:
    - Skills overlap
    - Experience match
    Return:
    - Match Score (0–100)
    - 3 Strengths
    - 2 Weaknesses

    Resume: {state['resume']}
    JD: {state['jd']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        **state,
        "result": response.content,
        "next_agent": "supervisor",
        "step_completed": state["step_completed"] + ["MatchScorer"]
    }


# ✅ LangGraph Workflow Build
workflow = StateGraph(HRState)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("JDAnalyzer", jd_analyzer)
workflow.add_node("ResumeParser", resume_parser)
workflow.add_node("MatchScorer", match_scorer)
workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda s: s["next_agent"],
    {
        "ResumeParser": "ResumeParser",
        "JDAnalyzer": "JDAnalyzer",
        "MatchScorer": "MatchScorer",
        END: END,
    },
)

workflow.add_conditional_edges(
    "ResumeParser",
    lambda s: s["next_agent"],
    {"supervisor": "supervisor"},
)

workflow.add_conditional_edges(
    "JDAnalyzer",
    lambda s: s["next_agent"],
    {"supervisor": "supervisor"},
)

workflow.add_conditional_edges(
    "MatchScorer",
    lambda s: s["next_agent"],
    {"supervisor": "supervisor"},
)

# ✅ Run Workflow Function Called by Fronte
graph = workflow.compile()

def run_hr_workflow(resume_pdf_path: str, jd_pdf_path: str):
    resume_text = extract_text_from_pdf(resume_pdf_path)
    jd_text = extract_text_from_pdf(jd_pdf_path)

    state: HRState = {
        "resume": resume_text,
        "jd": jd_text,
        "result": "",
        "next_agent": "supervisor",
        "step_completed": []
    }

    final_state = graph.invoke(state)
    return final_state["result"]
