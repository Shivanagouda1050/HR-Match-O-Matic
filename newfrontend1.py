import streamlit as st
from newmain1 import run_hr_workflow
import pandas as pd

st.set_page_config(page_title="MindMesh - CV Matcher", page_icon="🧠", layout="centered")

# UI Title
st.title("🧠 MindMesh — AI CV MatchMaker")
st.write("Upload your Resume & JD PDFs. AI Agents will match & analyse your profile ✅")

# Upload Inputs
resume_pdf = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
jd_pdf = st.file_uploader("📝 Upload Job Description (PDF)", type=["pdf"])

# Run Button
if st.button("🚀 Run Job Matching"):
    if not resume_pdf or not jd_pdf:
        st.warning("⚠️ Please upload both Resume & JD to begin!")
    else:
        with open("resume.pdf", "wb") as f:
            f.write(resume_pdf.read())
        with open("jd.pdf", "wb") as f:
            f.write(jd_pdf.read())

        with st.spinner("🤖 Agents are analyzing documents... Please wait..."):
            try:
                df, csv_out, match_report = run_hr_workflow("resume.pdf", "jd.pdf")
            except Exception as e:
                st.error(f"❌ Error Occurred: {e}")
                st.stop()

        st.success("✅ Match Report Ready!")

        # ✅ Show MatchScorer Text Output Only
        st.subheader("📌 AI Match Result")
        st.text_area("Detailed Result:", match_report, height=300)



        # ✅ CSV Download
        with open(csv_out, "rb") as f:
            st.download_button(
                label="📥 Download CSV Report",
                data=f,
                file_name="candidate_match_report.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.caption("⚡ Built with 💙 by Shiva — Powered by Multi-Agent Intelligence")
