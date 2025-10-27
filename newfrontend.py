import streamlit as st
from newmain import run_hr_workflow

st.set_page_config(page_title="MindMesh - CV Matcher", page_icon="🧠", layout="centered")

st.title("🧠 MindMesh — AI Job Match Maker")
st.write("Upload your Resume & JD PDFs — Agents will match skills and score compatibility ✅")

resume_pdf = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
jd_pdf = st.file_uploader("📝 Upload Job Description (PDF)", type=["pdf"])

if st.button("🚀 Run Matching Workflow"):
    if not resume_pdf or not jd_pdf:
        st.warning("⚠️ Please upload both Resume and JD PDFs.")
    else:
        resume_path = "uploaded_resume.pdf"
        jd_path = "uploaded_jd.pdf"

        with open(resume_path, "wb") as f:
            f.write(resume_pdf.read())
        with open(jd_path, "wb") as f:
            f.write(jd_pdf.read())

        st.info("🤖 Agents analyzing documents... this may take a few seconds")

        progress_placeholder = st.progress(0)
        progress_placeholder.progress(30)

        result = run_hr_workflow(resume_path, jd_path)

        progress_placeholder.progress(100)

        st.success("✅ Match Report Ready!")

        st.subheader("📊 Compatibility Report")
        st.text_area("Results:", result, height=300)

st.markdown("---")
st.markdown("<center>Built with 💙 by Shiva ⚡</center>", unsafe_allow_html=True)
