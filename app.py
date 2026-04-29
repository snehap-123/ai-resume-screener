import streamlit as st
import pdfplumber
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Page config
st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.title("🚀 AI Resume Screener")
st.write("Upload your resume and compare it with a job description")

# ---------------------------
# SESSION STATE
# ---------------------------
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

# ---------------------------
# SAMPLE JOB DESCRIPTIONS
# ---------------------------
sample_jds = {
    "Select Sample JD": "",
    "Python Developer": "Looking for a Python developer with experience in Django, REST APIs, SQL, and backend development.",
    "Data Scientist": "Looking for a data scientist skilled in Python, Machine Learning, pandas, scikit-learn, and data visualization.",
    "Frontend Developer": "Looking for a frontend developer with React, JavaScript, HTML, CSS, and UI/UX experience.",
    "AI/ML Engineer": "Looking for an AI engineer with deep learning, TensorFlow, NLP, and model deployment experience."
}

selected_jd = st.selectbox("Choose Sample Job Description", list(sample_jds.keys()))

if selected_jd != "Select Sample JD":
    st.session_state.job_desc = sample_jds[selected_jd]

# Job Description Input
job_desc = st.text_area("Paste Job Description", value=st.session_state.job_desc)

# Upload resume
uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])


# ---------------------------
# FUNCTIONS
# ---------------------------

def extract_text(pdf):
    text = ""
    with pdfplumber.open(pdf) as pdf_file:
        for page in pdf_file.pages:
            text += page.extract_text() or ""
    return text


def extract_keywords(text):
    doc = nlp(text.lower())
    return [
        token.text for token in doc
        if token.is_alpha and not token.is_stop and token.pos_ in ["NOUN", "PROPN"]
    ]


# Skill categories
skill_categories = {
    "Programming": ["python", "java", "c++", "javascript"],
    "Machine Learning": ["ml", "machine", "learning", "pandas", "sklearn", "tensorflow"],
    "Database": ["sql", "mysql", "mongodb"]
}

# ---------------------------
# MAIN LOGIC
# ---------------------------

if uploaded_file:
    st.success("✅ Resume uploaded successfully")

    resume_text = extract_text(uploaded_file)

    if job_desc:
        with st.spinner("Analyzing Resume..."):

            resume_keywords = extract_keywords(resume_text)
            jd_keywords = extract_keywords(job_desc)

            # Remove noise
            resume_keywords = [w for w in resume_keywords if len(w) > 2]
            jd_keywords = [w for w in jd_keywords if len(w) > 2]

            # Convert to string
            resume_str = " ".join(resume_keywords)
            jd_str = " ".join(jd_keywords)

            # TF-IDF similarity
            vector = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([resume_str, jd_str])
            similarity = cosine_similarity(vector)[0][1]
            score = round(similarity * 100, 2)

            # Skill comparison
            matched = sorted(set(resume_keywords) & set(jd_keywords))
            missing = sorted(set(jd_keywords) - set(resume_keywords))

            # ---------------------------
            # UI OUTPUT
            # ---------------------------
            st.markdown("## 📊 Analysis Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Match Score", f"{score}%")
                st.progress(int(score))

            with col2:
                st.metric("Missing Skills", len(missing))

            # Matched Skills
            st.subheader("✅ Matched Skills")
            if matched:
                st.success(", ".join(matched[:20]))
            else:
                st.write("No strong matches found")

            # Missing Skills
            st.subheader("❌ Missing Skills")
            if missing:
                st.error(", ".join(missing[:20]))
                st.info("💡 Tip: Add these skills to improve your match score")
            else:
                st.write("No major missing skills 🎉")

            # ---------------------------
            # SKILL CATEGORY ANALYSIS
            # ---------------------------
            st.markdown("## 📂 Skill Category Analysis")

            for category, skills in skill_categories.items():
                matched_skills = [s for s in matched if s in skills]
                score_cat = (len(matched_skills) / len(skills)) * 100
                st.write(f"{category}: {round(score_cat, 2)}%")

            # ---------------------------
            # DOWNLOAD REPORT (FIXED)
            # ---------------------------
            report = f"""
AI Resume Screener Report

Match Score: {score}%

Matched Skills:
{', '.join(matched)}

Missing Skills:
{', '.join(missing)}
"""

            st.download_button(
                label="📥 Download TXT Report",
                data=report.encode("utf-8"),
                file_name="resume_report.txt",
                mime="text/plain"
            )

            # CSV Report
            max_len = max(len(matched), len(missing))
            matched_extended = matched + [""] * (max_len - len(matched))
            missing_extended = missing + [""] * (max_len - len(missing))

            df = pd.DataFrame({
                "Matched Skills": matched_extended,
                "Missing Skills": missing_extended
            })

            st.download_button(
                label="📊 Download CSV Report",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="resume_report.csv",
                mime="text/csv"
            )

    else:
        st.warning("⚠️ Please paste a job description")

else:
    st.info("📄 Please upload a resume to begin")