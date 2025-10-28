# ============================================
# ✅ Updated Backend — Proper Match Output Parsing
# ============================================

from typing import TypedDict, Optional
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import pdfplumber
import pandas as pd
import re
from datetime import datetime

# Load ENV Keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ LLM Setup
llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

class HRState(TypedDict):
    resume: str
    jd: str
    result: str
    name: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    education: Optional[str]
    exp_gap: Optional[str]
    match_score: Optional[str]
    strengths: Optional[str]
    weaknesses: Optional[str]
    next_agent: str
    step_completed: list


# ✅ PDF TEXT Extractor
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()


# ✅ Resume Parser
def resume_parser(state: HRState):
    text = state["resume"]

    name = re.findall(r"[A-Z][a-z]+\s[A-Z][a-z]+", text)
    email = re.search(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+", text)
    phone = re.search(r"\b[0-9]{10}\b", text)

    return {
        **state,
        "name": name[0] if name else "Unknown",
        "email": email.group(0) if email else "Unknown",
        "phone": phone.group(0) if phone else "Unknown",
        "education": "Extracted Education",
        "exp_gap": "Unknown",
        "next_agent": "JDAnalyzer",
        "step_completed": state["step_completed"] + ["ResumeParser"]
    }


def jd_analyzer(state: HRState):
    return {
        **state,
        "next_agent": "MatchScorer",
        "step_completed": state["step_completed"] + ["JDAnalyzer"]
    }


# ✅ Match Score Agent — Improved
def match_scorer(state: HRState):
    prompt = f"""
    Compare the Resume & JD, give output EXACTLY in this structure:

    Match Score: <ONLY NUMBER 0-100>

    Strengths:
    - <strength>
    - <strength>

    Weaknesses:
    - <weakness>
    - <weakness>

    Resume: {state['resume']}
    JD: {state['jd']}
    """

    response = llm.invoke([HumanMessage(content=prompt)]).content
    response_text = response.strip()

    # ✅ Parse Output
    score = re.search(r"Match Score:\s*(\d+)", response_text)
    strengths = re.findall(r"- (.*)", response_text.split("Strengths:")[1].split("Weaknesses:")[0]) if "Strengths:" in response_text else []
    weaknesses = re.findall(r"- (.*)", response_text.split("Weaknesses:")[1]) if "Weaknesses:" in response_text else []

    return {
        **state,
        "result": response_text,
        "match_score": score.group(1) if score else "0",
        "strengths": " | ".join(strengths) if strengths else "None",
        "weaknesses": " | ".join(weaknesses) if weaknesses else "None",
        "next_agent": END,
        "step_completed": state["step_completed"] + ["MatchScorer"]
    }


# ✅ LangGraph Execution
workflow = StateGraph(HRState)
workflow.add_node("ResumeParser", resume_parser)
workflow.add_node("JDAnalyzer", jd_analyzer)
workflow.add_node("MatchScorer", match_scorer)
workflow.set_entry_point("ResumeParser")

workflow.add_edge("ResumeParser", "JDAnalyzer")
workflow.add_edge("JDAnalyzer", "MatchScorer")
workflow.add_edge("MatchScorer", END)

graph = workflow.compile()


# ✅ FINAL EXPORT FUNCTION
def run_hr_workflow(resume_pdf_path: str, jd_pdf_path: str):
    state: HRState = {
        "resume": extract_text_from_pdf(resume_pdf_path),
        "jd": extract_text_from_pdf(jd_pdf_path),
        "result": "",
        "name": "",
        "phone": "",
        "email": "",
        "education": "",
        "exp_gap": "",
        "match_score": "",
        "strengths": "",
        "weaknesses": "",
        "next_agent": "ResumeParser",
        "step_completed": []
    }

    final = graph.invoke(state)

    df = pd.DataFrame([{
        "Name": final["name"],
        "Mobile": final["phone"],
        "Email": final["email"],
        "Education": final["education"],
        "Experience Gap": final["exp_gap"],
        "Match Score": final["match_score"],
        "Strengths": final["strengths"],
        "Weaknesses": final["weaknesses"]
    }])

    csv_file = "candidate_match_report.csv"
    df.to_csv(csv_file, index=False)

    return df, csv_file, final["result"]
