# import subprocess
# import sys
# This "calls" the terminal from inside Python
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
import streamlit as st
import pandas as pd
import json
import re
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

debug = st.sidebar.checkbox("Debug Mode")
# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="LinkedIn Profile Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme style overrides
st.markdown(
    """
    <style>
        .main {background-color: #0e1117; color: white;}
        .stTextInput > div > div > input {color: white;}
        .stSelectbox > div > div > div {color: black;}
        table {color: white;}
    </style>
""",
    unsafe_allow_html=True,
)


# ------------------- LOAD MODELS -------------------
@st.cache_resource
def load_models():
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    nli_model = pipeline("text-classification", model="roberta-large-mnli")
    nlp = spacy.load("en_core_web_sm")
    return sbert_model, nli_model, nlp


sbert_model, nli_model, nlp = load_models()

# ------------------- HELPER FUNCTIONS -------------------


def cleaning_data(df_profiles):
    # Data Cleaning
    clean_data = pd.DataFrame()
    clean_data["id"] = df_profiles["id"]
    clean_data["First Name"] = df_profiles["firstName.localized.en_US"]
    clean_data["Last Name"] = df_profiles["lastName.localized.en_US"]
    clean_data["FullName"] = (
        df_profiles["firstName.localized.en_US"]
        + " "
        + df_profiles["lastName.localized.en_US"]
    )
    clean_data["Summary"] = df_profiles["summary.localized.en_US"]
    clean_data["Industry"] = df_profiles["industry"]
    clean_data["Posts"] = df_profiles["posts"]
    clean_data["Qualifications"] = df_profiles["education.elements"]
    clean_data["Certifications"] = df_profiles["certifications"]
    clean_data["Experience"] = df_profiles["positions.elements"]
    clean_data["Descrip_score"] = None
    # print(clean_data)


def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\\w\\s]", "", text)
    return text


def extract_skills(text, keywords):
    doc = nlp(text.lower())
    found = set()
    for token in doc:
        if token.text in keywords:
            found.add(token.text)
    return found


def get_match_score(candidate_text, job_text, job_skills):
    # 1. Skill overlap
    cand_skills = ""
    job_skills_low = [s.lower() for s in job_skills]
    cand_skills = extract_skills(candidate_text, job_skills_low)
    overlap = cand_skills.intersection(job_skills)
    if debug:
        st.write("Extracted Skills: ", cand_skills)
        st.write("Overlap: ", overlap)
    skill_score = len(overlap) / len(job_skills) if job_skills else 0

    # 2. Semantic similarity
    embeddings = sbert_model.encode([candidate_text, job_text], convert_to_tensor=True)
    sim_score = float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))
    if sim_score < 0:
        sim_score = 0

    # 3. NLI relationship
    relation = nli_model(f"{candidate_text} </s></s> {job_text}")[0]
    if debug:
        st.write("relation: ", relation)
    nli_bonus = (
        1.0
        if relation["label"] == "ENTAILMENT"
        else 0.5 if relation["label"] == "NEUTRAL" else 0
    )

    # Final weighted score (out of 100)
    final_score = (skill_score * 40) + (sim_score * 40) + (nli_bonus * 20)
    if final_score < 60:
        label = "Not Recommended"
    elif final_score < 90:
        label = "Considerable"
    else:
        label = "Highly Recommended"
    return round(final_score, 2), {
        "skills_matched": list(overlap),
        "skill_score": skill_score,
        "semantic_score": round(sim_score, 2),
        "nli_relation": label,
    }


def post_caption_score(summary, job_desc, keywords):
    sim_score = get_bert_similarity(summary, job_desc) * 2
    boost = keyword_boost(summary, job_desc, keywords)
    return round(0.8 * sim_score + 0.2 * boost, 3)


def engagement_analysis(no_comment, no_likes, no_shares):
    comment_score = 10 * no_comment
    engagement_score = no_likes * 0.2 + comment_score * 0.8
    return engagement_score


def final_analysis(df_profiles):
    results = []
    for _, row in df_profiles.iterrows():
        posts = row.posts
        post_summary_score_list = []
        engagement_score_list = []
        candidate_text = (
            str(row.get("summary.localized.en_US", ""))
            + " "
            + str(row.get("industry", ""))
        )
        Emp_descrip_score, details = get_match_score(
            candidate_text, job_text, set([s.strip().lower() for s in job_skills])
        )
        for post in posts:
            content = post["content"]
            comments = post["comments"]
            likes = post["likes"]
            shares = post["shares"]

            # Caption Analysis
            post_summary_score = post_caption_score(content, job_text, job_skills)
            post_summary_score_list.append(post_summary_score)
            final_caption_score = (
                sum(x or 0 for x in post_summary_score_list)
                / len(post_summary_score_list)
            ) / 10
            # Engagement Analysis
            engagement_score = engagement_analysis(comments, likes, shares)
            engagement_score_list.append(engagement_score)
        final_engagement_score = (
            sum(x or 0 for x in engagement_score_list) / len(engagement_score_list)
        ) / 10
        score = (
            round(
                (Emp_descrip_score * 90)
                + (final_caption_score * 5)
                + (final_engagement_score * 5)
            )
            / 100
        )
        results.append(
            {
                "Candidate": row.get("firstName.localized.en_US", "")
                + " "
                + row.get("lastName.localized.en_US", ""),
                "Overall Score": score,
                "Engagement Score": round(final_engagement_score, 2),
                "Semantic Score": round(details["semantic_score"] * 100, 2),
                "Matched Skills": ", ".join(details["skills_matched"]),
            }
        )
    return results


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_bert_similarity(text1, text2):
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    return float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))


def keyword_boost(text1, text2, keywords):
    count = sum(
        3
        for kw in keywords
        if kw.lower() in text1.lower() and kw.lower() in text2.lower()
    )
    return count / len(keywords)  # normalized boost factor


# ------------------- SIDEBAR -------------------
st.sidebar.title("⚡ Project Navigation")
page = st.sidebar.radio(
    "Go to", ["Overview", "Data Explorer", "Model & Scoring", "Results"]
)

# ------------------- PAGES -------------------
if page == "Overview":
    st.title("🔍 LinkedIn Profile Analyzer")
    st.write(
        """
        This project analyzes LinkedIn profiles and compares them with a job description
        using **skill overlap, semantic similarity, and NLI reasoning**.

        **Features:**
        - Upload or use sample LinkedIn JSON data  
        - Explore candidate data  
        - Score profiles against a job description  
        - Rank candidates by relevance  
    """
    )
    # Sample JSON template
    sample_json = [
        {
            "id": "b874f631-c42d-46d6-8e3b-e7351013af35",
            "firstName": {"localized": {"en_US": "John"}},
            "lastName": {"localized": {"en_US": "Doe"}},
            "profilePicture": {"displayImage": "urn:li:digitalmediaAsset:EOjMcp"},
            "headline": {"localized": {"en_US": "Software Developer"}},
            "location": {
                "country": "TV",
                "postalCode": "83693",
                "geographicArea": "Schmidtstad, Kentucky",
            },
            "industry": "Information Technology & Services",
            "summary": {"localized": {"en_US": "As a dedicated Data Scientist..."}},
            "positions": {
                "elements": [
                    {
                        "title": "Data Scientist",
                        "company": {"name": "Nunez, Bennett and Spencer"},
                        "location": "Vanessaside, DC",
                        "startDate": {"year": "1978", "month": "08"},
                        "endDate": {'year':'','month':''},
                        "description": "Discover rock build open.",
                    }
                ]
            },
            "education": {
                "elements": [
                    {
                        "schoolName": "MIT",
                        "degreeName": "Master of Science",
                        "fieldOfStudy": "Computer Science",
                        "startDate": {"year": 2015},
                        "endDate": {"year": 2017},
                    }
                ]
            },
            "skills": {"elements": [{"name": {"localized": {"en_US": "Python"}}}]},
            "certifications": [],
            "languages": [],
            "followers": 592,
            "total_posts": 5,
            "posts": [],
            "posts_summary": {
                "total_likes": 798,
                "total_comments": 151,
                "total_shares": 70,
            },
            "publicProfileUrl": "https://www.linkedin.com/in/sample-profile",
        }
    ]
    import json

    json_data = json.dumps(sample_json, indent=4)
    st.download_button(
        label="📥 Download Sample JSON Template",
        data=json_data,
        file_name="linkedin_sample_template.json",
        mime="application/json",
    )

elif page == "Data Explorer":
    st.title("📊 Candidate Data")

    uploaded_file = st.file_uploader("Upload LinkedIn JSON", type=["json"])

    if uploaded_file:
        data = json.load(uploaded_file)
    else:
        st.write("⚠ No JSON file uploaded. Using sample data.")
        # with open("linkedIn_data.json", "r", encoding="utf-8") as file:
        #     data = json.load(file)
        st.stop()

    # Job description input
    job_text = st.text_area(
        "Paste Job Description",
        "Looking for a data scientist with Python, ML, and NLP skills.",
    )
    # Skills input
    job_skills = st.text_input(
        "Enter Job Skills (comma separated)", "python, machine, learning, nlp, sql"
    ).split(",")
    if st.button("Explore Candidates"):
        df_profiles = pd.json_normalize(data)
        # cols = [
        #     "firstName.localized.en_US",
        #     "lastName.localized.en_US",
        #     "industry",
        #     "summary.localized.en_US"
        # ]
        # st.write(df_profiles.columns)
        # st.dataframe(df_profiles[cols])
        results = final_analysis(df_profiles)
        results_df = pd.DataFrame(results)
        st.subheader("Candidate Rankings")
        st.line_chart(results_df.set_index("Candidate")["Overall Score"])
        st.subheader("Post Engagement Score")
        st.line_chart(results_df.set_index("Candidate")["Engagement Score"])
        st.subheader("Semantic Score")
        st.line_chart(results_df.set_index("Candidate")["Semantic Score"])
        st.write(results)

elif page == "Model & Scoring":
    st.title("🤖 Profile Scoring")

    # File uploader or fallback
    uploaded_file = st.file_uploader("Upload LinkedIn JSON", type=["json"])
    if uploaded_file:
        data = json.load(uploaded_file)
    else:
        st.write("⚠ No JSON file found. Please upload a LinkedIn JSON file.")
        st.stop()

    df_profiles = pd.json_normalize(data)
    clean_data = cleaning_data(df_profiles)

    # Select candidate
    candidate = st.selectbox(
        "Select Candidate",
        df_profiles["firstName.localized.en_US"]
        + " "
        + df_profiles["lastName.localized.en_US"],
    )
    st.markdown(
        f"<h3 style='color:#FFFFFF;'>Selected Candidate: {candidate}</h3>",
        unsafe_allow_html=True,
    )
    candidate_row = df_profiles[
        df_profiles["firstName.localized.en_US"]
        + " "
        + df_profiles["lastName.localized.en_US"]
        == candidate
    ].iloc[0]
    # Candidate text
    candidate_text = (
        str(candidate_row.get("summary.localized.en_US", ""))
        + " "
        + str(candidate_row.get("industry", ""))
    )
    st.write("Candidate Text:", candidate_text)
    # Job description input
    job_text = st.text_area(
        "Paste Job Description",
        "Looking for a data scientist with Python, ML, and NLP skills.",
    )
    # Skills input
    job_skills = st.text_input(
        "Enter Job Skills (comma separated)", "python, machine, learning, nlp, sql"
    ).split(",")
    # if debug:
    #     st.write("Candidate Row:", candidate_row)
    #     st.write("Candidate Text:", candidate_text)
    if st.button("Calculate Match Score"):
        score, details = get_match_score(
            candidate_text, job_text, set([s.strip().lower() for s in job_skills])
        )

        st.metric("Final Match Score", f"{score}/100")
        st.write("Skills Match: ", details["skills_matched"])
        st.write("Skill Score: ", details["skill_score"])
        st.write("Semantic Score: ", details["semantic_score"])
        st.write("Candidate Review: ", details["nli_relation"])

elif page == "Results":
    st.title("🏆 Ranked Candidates")

    # File uploader or fallback
    uploaded_file = st.file_uploader("Upload LinkedIn JSON", type=["json"])
    if uploaded_file:
        data = json.load(uploaded_file)
    else:
        st.write("⚠ No JSON file found. Please upload a LinkedIn JSON file.")
        st.stop()
        # with open("linkedIn_data.json", "r") as file:
        #     data = json.load(file)

    df_profiles = pd.json_normalize(data)
    clean_data = cleaning_data(df_profiles)

    job_text = st.text_area(
        "Paste Job Description",
        "Looking for a data scientist with Python, ML, and NLP skills.",
    )
    job_skills = st.text_input(
        "Enter Job Skills (comma separated)", "python, machine, learning, nlp, sql"
    ).split(",")
    post_summary_score_list = []
    engagement_score_list = []

    if st.button("Rank Candidates"):
        results = final_analysis(df_profiles)
        results_df = pd.DataFrame(results).sort_values("Overall Score", ascending=False)
        # st.write(results)
        # st.write(results_df)
        st.write(results_df)
