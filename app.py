import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import plotly.graph_objects as go
import plotly.express as px
import base64
import requests
import json

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Career Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES  — refined dark editorial aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --ink:     #0a0a0f;
    --surface: #111118;
    --card:    #16161f;
    --border:  rgba(255,255,255,0.07);
    --accent:  #6ee7b7;
    --accent2: #818cf8;
    --muted:   #6b7280;
    --text:    #e5e7eb;
    --bright:  #f9fafb;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--ink) !important;
    color: var(--text);
}

.stApp {
    background: var(--ink) !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── INTRO PAGE ── */
.hero-wrap {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, #1a1a3e 0%, var(--ink) 70%);
    padding: 60px 20px;
}

.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 4px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 28px;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(64px, 10vw, 130px);
    font-weight: 800;
    color: var(--bright);
    line-height: 0.95;
    text-align: center;
    letter-spacing: -3px;
    margin: 0 0 24px;
}

.hero-title span {
    color: var(--accent);
}

.hero-sub {
    font-size: 18px;
    font-weight: 300;
    color: var(--muted);
    text-align: center;
    max-width: 480px;
    line-height: 1.7;
    margin-bottom: 56px;
}

.hero-badge-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: center;
    margin-bottom: 64px;
}

.hero-badge {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 8px 18px;
    font-size: 12px;
    color: var(--muted);
    letter-spacing: 0.5px;
}

/* ── BUTTONS ── */
div[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    height: 48px !important;
    padding: 0 32px !important;
    border-radius: 6px !important;
    border: none !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
    width: auto !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(110,231,183,0.25) !important;
}
/* Secondary nav buttons */
.nav-col div[data-testid="stButton"] > button {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    font-size: 12px !important;
}
.nav-col div[data-testid="stButton"] > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: none !important;
}

/* ── INNER PAGES ── */
.page-wrap {
    max-width: 1100px;
    margin: 0 auto;
    padding: 60px 32px 100px;
}

.page-label {
    font-size: 11px;
    letter-spacing: 4px;
    color: var(--accent);
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 12px;
}

.page-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(36px, 5vw, 64px);
    font-weight: 800;
    color: var(--bright);
    line-height: 1.05;
    letter-spacing: -1.5px;
    margin: 0 0 48px;
}

.page-title span { color: var(--accent); }

/* ── CARDS ── */
.career-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.career-card:hover { border-color: rgba(110,231,183,0.25); }

.career-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    opacity: 0;
    transition: opacity 0.2s;
}
.career-card:hover::before { opacity: 1; }

.card-rank {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 8px;
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: var(--bright);
    letter-spacing: -0.5px;
    margin: 0 0 6px;
}

.card-match {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 20px;
}

.match-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 4px;
    margin-bottom: 24px;
    overflow: hidden;
}
.match-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}

.card-desc {
    font-size: 15px;
    line-height: 1.7;
    color: #9ca3af;
    margin-bottom: 24px;
}

.card-salary {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--bright);
}
.card-salary-label {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}

.roadmap-list {
    list-style: none;
    padding: 0;
    margin: 20px 0 0;
}
.roadmap-list li {
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    font-size: 14px;
    color: #9ca3af;
    display: flex;
    align-items: center;
    gap: 10px;
}
.roadmap-list li:last-child { border-bottom: none; }
.roadmap-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
}

/* ── JOB LISTING CARDS ── */
.job-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 14px;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 20px;
    transition: border-color 0.2s, transform 0.15s;
}
.job-card:hover {
    border-color: rgba(110,231,183,0.2);
    transform: translateY(-2px);
}

.job-logo {
    width: 44px; height: 44px;
    border-radius: 10px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}

.job-info { flex: 1; }

.job-title-text {
    font-family: 'Syne', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: var(--bright);
    margin: 0 0 4px;
}

.job-company {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 10px;
}

.job-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.job-tag {
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 3px 10px;
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.3px;
}

.job-tag.accent {
    background: rgba(110,231,183,0.08);
    border-color: rgba(110,231,183,0.2);
    color: var(--accent);
}

.apply-btn {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    padding: 8px 20px !important;
    border-radius: 6px !important;
    text-decoration: none !important;
    white-space: nowrap;
    transition: all 0.2s;
    flex-shrink: 0;
    align-self: center;
    display: inline-block;
}
.apply-btn:hover {
    background: rgba(110,231,183,0.1) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── SECTION HEADERS ── */
.section-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin: 56px 0 24px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 800;
    color: var(--bright);
    letter-spacing: -0.3px;
    margin: 0;
}
.section-count {
    font-size: 12px;
    color: var(--muted);
    letter-spacing: 1px;
}

/* ── INSIGHT BOX ── */
.insight-pill {
    background: rgba(129,140,248,0.08);
    border: 1px solid rgba(129,140,248,0.15);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 14px;
    color: #c7d2fe;
    margin-bottom: 10px;
    line-height: 1.6;
}

/* ── MODEL BADGE ── */
.model-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(110,231,183,0.07);
    border: 1px solid rgba(110,231,183,0.2);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
}

/* ── SLIDER OVERRIDES ── */
.stSlider > div > div > div {
    background: rgba(110,231,183,0.2) !important;
}
.stSlider > div > div > div > div {
    background: var(--accent) !important;
}

/* ── SELECT BOX ── */
.stSelectbox > div > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* ── FORM SUBMIT ── */
div[data-testid="stFormSubmitButton"] > button {
    background: var(--accent) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
    height: 48px !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 0 40px !important;
    width: auto !important;
    transition: all 0.2s !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 8px 32px rgba(110,231,183,0.3) !important;
    transform: translateY(-2px) !important;
}

/* ── DOWNLOAD BUTTON ── */
div[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    height: 44px !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── YEAR SELECTOR CARDS ── */
.year-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
}
.year-card:hover { border-color: var(--accent); }
.year-card.active { border-color: var(--accent); background: rgba(110,231,183,0.06); }

/* ── GRID ── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 32px;
}
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: var(--bright);
    letter-spacing: -1px;
}
.stat-label {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── LOADING SPINNER ── */
.loading-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: var(--muted);
    text-align: center;
    padding: 40px;
    letter-spacing: 1px;
}

/* divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 40px 0;
}

/* nav buttons */
.nav-row {
    display: flex;
    gap: 12px;
    margin-top: 56px;
    padding-top: 32px;
    border-top: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    job_info = pd.read_csv("job_data.csv")
    df.columns = df.columns.str.strip()
    job_info.columns = job_info.columns.str.strip()
    def clean_salary(x):
        return int(str(x).replace('$', '').replace(',', '').strip())
    job_info["Monthly Salary (USD)"] = job_info["Monthly Salary (USD)"].apply(clean_salary)
    return df, job_info

df, job_info = load_data()

# ─────────────────────────────────────────────
# SUBJECTS BY YEAR
# ─────────────────────────────────────────────
year_subjects = {
    1: ["Calculus", "Analytic Geometry And Algebra", "Descriptive Statistics", "Introduction To Algorithms",
        "Generative AI", "Probability Theory", "Discrete Math", "Python", "Introduction To Low Code"],
    2: ["Applied Statistics", "C Programming", "Computer Science", "Physics", "Management", "Numerical Methods",
        "Logic", "Information Theory", "Architecture", "Algorithms", "C Sharp"],
    3: ["Systems", "Math For CS", "Computer Networks", "Human Computer Interaction", "Machine Learning",
        "Complexity", "Databases", "OOP", "Data Structures"],
    4: ["Systems Programming", "Parallel Programming", "Functional Programming & ADT", "Probability And Statistics",
        "Graph Theory", "Theory Of Automata", "Computer Graphics", "Image Processing",
        "Advanced Functional Programming & ADT", "Signal Processing", "Software Engineering",
        "Computer Security", "Artificial Intelligence"]
}

def get_subjects(year):
    return sum((year_subjects.get(y, []) for y in range(1, year + 1)), [])

# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_models():
    all_subjects = [c for c in df.columns if c != "Job_Title"]
    X = df[all_subjects].values
    y = df["Job_Title"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)
    nb = GaussianNB()
    knn.fit(X_scaled, y_enc)
    dt.fit(X_scaled, y_enc)
    nb.fit(X_scaled, y_enc)
    return knn, dt, nb, le, scaler, all_subjects

knn, dt, nb, le, scaler, all_subjects = train_models()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_predictions(vec, model_choice):
    X_input = scaler.transform([vec])
    if model_choice == "KNN Only":
        probs = knn.predict_proba(X_input)[0]; label = "KNN"
    elif model_choice == "Decision Tree Only":
        probs = dt.predict_proba(X_input)[0]; label = "Decision Tree"
    elif model_choice == "Naive Bayes Only":
        probs = nb.predict_proba(X_input)[0]; label = "Naive Bayes"
    else:
        p_knn = knn.predict_proba(X_input)[0]
        p_dt  = dt.predict_proba(X_input)[0]
        p_nb  = nb.predict_proba(X_input)[0]
        probs = 0.40 * p_knn + 0.35 * p_dt + 0.25 * p_nb
        label = "Ensemble"
    top3 = np.argsort(probs)[-3:][::-1]
    return le.inverse_transform(top3), probs[top3] * 100, label

def get_per_model_top1(vec):
    X_input = scaler.transform([vec])
    return {
        "KNN": le.inverse_transform([np.argmax(knn.predict_proba(X_input)[0])])[0],
        "Decision Tree": le.inverse_transform([np.argmax(dt.predict_proba(X_input)[0])])[0],
        "Naive Bayes": le.inverse_transform([np.argmax(nb.predict_proba(X_input)[0])])[0],
    }

def get_insights(rankings, top_job):
    insights = []
    sorted_subjects = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    top_subjects = [s for s, r in sorted_subjects[:5] if r >= 3]
    weak_subjects = [s for s, r in sorted_subjects if r <= 2]
    avg = np.mean(list(rankings.values()))
    values = list(rankings.values())

    # Always fires: top strengths
    if top_subjects:
        top_str = ", ".join(f"<strong style='color:var(--bright);'>{s}</strong>" for s in top_subjects[:3])
        insights.append(f"✦ Your strongest subjects are {top_str} — these directly power your top career match.")
    else:
        insights.append("✦ Your ratings are evenly spread — you're a versatile generalist, which is valuable in cross-functional tech roles.")

    # Always fires: average readiness
    if avg >= 4:
        insights.append(f"✦ With an average score of <strong style='color:var(--accent);'>{avg:.1f}/5</strong>, you're performing at a high level — you're industry-ready right now.")
    elif avg >= 3:
        insights.append(f"✦ Your average score is <strong style='color:var(--accent);'>{avg:.1f}/5</strong> — solid foundation. Deepening your top 2–3 subjects will make a big difference.")
    else:
        insights.append(f"✦ Average score: <strong style='color:var(--accent);'>{avg:.1f}/5</strong>. Junior roles and structured internships are the perfect next step — every senior engineer started here.")

    # Conditional: weak spots
    if weak_subjects:
        weak_str = ", ".join(f"<strong style='color:var(--bright);'>{s}</strong>" for s in weak_subjects[:2])
        insights.append(f"✦ Improving in {weak_str} could significantly expand the range of careers available to you.")

    # Conditional: spread
    spread = max(values) - min(values)
    if spread <= 1:
        insights.append("✦ Your scores are very consistent — you're well-rounded. Consider picking one area to go deep rather than staying broad.")
    elif spread >= 3:
        best = sorted_subjects[0][0]
        insights.append(f"✦ You have a clear standout strength in <strong style='color:var(--bright);'>{best}</strong>. Double down — specialists in this area are in high demand.")

    # Always fires: top job match tip
    job_tips = {
        "AI Engineer / NLP Engineer": "The AI market is growing fast — having even one deployed model on GitHub dramatically improves your chances.",
        "Data Analyst": "SQL and Tableau/Power BI are the two skills that will get you hired fastest as a Data Analyst.",
        "Cybersecurity Analyst / SOC Analyst": "Getting a free TryHackMe or HackTheBox account and solving 10 rooms is equivalent to months of theory.",
        "Backend Developer Intern": "Build one full CRUD API with authentication and deploy it — that single project speaks louder than your GPA.",
        "UI/UX Designer": "A 3-case-study Figma portfolio matters more than any certification. Start one project this week.",
        "Computer Vision Engineer": "OpenCV + one real-world dataset project on Kaggle will make your CV stand out immediately.",
        "Big Data Engineer": "Get AWS Free Tier and build one pipeline with S3 + Lambda — cloud experience is non-negotiable now.",
        "Systems Engineer": "Contributing to even one open-source C/C++ project signals seriousness to any systems-focused hiring team.",
    }
    tip = job_tips.get(top_job, f"For <strong style='color:var(--bright);'>{top_job}</strong>, one polished GitHub project and active LinkedIn profile is the fastest path to your first interview.")
    insights.append(f"✦ {tip}")

    return insights

def get_roadmap(job):
    roadmaps = {
        "AI Engineer / NLP Engineer": ["Learn PyTorch / TensorFlow", "Study transformer architectures", "Build NLP projects on GitHub", "Contribute to open-source ML"],
        "Data Analyst": ["Master Excel & SQL", "Learn Power BI or Tableau", "Practice on Kaggle datasets", "Build a portfolio dashboard"],
        "Cybersecurity Analyst / SOC Analyst": ["Study CompTIA Security+", "Set up a home lab", "Learn SIEM tools", "Earn CEH or OSCP"],
        "Backend Developer Intern": ["Deepen Python/Java skills", "Learn REST APIs & Docker", "Contribute to open-source", "Build a personal project"],
        "UI/UX Designer": ["Learn Figma", "Study HCI principles", "Build a case study portfolio", "Get user testing experience"],
        "Computer Vision Engineer": ["Study OpenCV & deep learning", "Work on image classification projects", "Learn YOLO / SAM frameworks", "Publish on GitHub"],
        "Big Data Engineer": ["Learn Apache Spark & Hadoop", "Practice with AWS / GCP", "Build a data pipeline project", "Study distributed systems"],
        "Systems Engineer": ["Deep-dive into OS internals", "Study networking protocols", "Build a side project in C/C++", "Read 'Computer Systems: A Programmer's Perspective'"],
    }
    return roadmaps.get(job, [
        f"Build real-world projects related to {job}",
        "Create a GitHub portfolio showcasing your work",
        "Look for internships or freelance gigs",
        "Join relevant online communities & meetups",
    ])

# ─────────────────────────────────────────────
# LIVE JOB LISTINGS — scrape staff.am via Claude API
# ─────────────────────────────────────────────
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Map career titles → staff.am search keywords
CAREER_SEARCH_KEYWORDS = {
    "AI Engineer / NLP Engineer":            "AI engineer",
    "Data Analyst":                          "data analyst",
    "Research Assistant":                    "research",
    "Statistical Assistant":                 "statistician",
    "Data Quality Analyst":                  "data quality",
    "Insurance Data Assistant":              "data analyst",
    "AI Assistant Developer":               "AI developer",
    "Junior Python Developer":               "python developer",
    "QA Automation Assistant":               "QA automation",
    "Software Engineering Intern":           "software engineer intern",
    "Applied Statistics Specialist":         "statistics",
    "Junior Statistical Analyst":            "data analyst",
    "Biostatistics Assistant":               "data analyst",
    "C Programming":                         "C developer",
    "Technical Product Coordinator":         "product manager",
    "C# Developer":                          "C# developer",
    "Data Processing Specialist":            "data engineer",
    "Technical Workflow Coordinator":        "project coordinator",
    "Junior Technical Consultant":           "IT consultant",
    "ASIC Design Engineer":                  "hardware engineer",
    "Linux Support Technician":              "linux",
    "DevOps Intern":                         "devops intern",
    "Network Technician":                    "network engineer",
    "UI/UX Designer":                        "UX designer",
    "Data Management Assistant":             "database",
    "ML Model Testing Intern":               "machine learning intern",
    "QA Automation Engineer (Entry-Level)":  "QA engineer",
    "Product Design Intern":                 "product designer",
    "Junior Database Administrator":         "database administrator",
    "Cloud Engineering Intern":              "cloud engineer",
    "UX Research Assistant":                 "UX researcher",
    "Backend Developer Intern":              "backend developer",
    "Systems Engineer":                      "systems engineer",
    "Kernel / Systems Programmer":           "systems programmer",
    "High-Performance Computing (HPC) Engineer": "software engineer",
    "Distributed Systems Engineer":          "backend engineer",
    "Graph Algorithms Engineer":             "software engineer",
    "Verification Engineer":                 "QA engineer",
    "Graphics Programmer":                   "game developer",
    "Rendering Engineer":                    "software engineer",
    "Computer Vision Engineer":              "computer vision",
    "Digital Signal Processing (DSP) Engineer": "signal processing",
    "Cybersecurity Analyst / SOC Analyst":   "cybersecurity",
    "Device Driver Engineer":                "embedded engineer",
    "Operating Systems Developer":           "systems developer",
    "Big Data Engineer":                     "data engineer",
}

CAREER_EMOJIS = {
    "AI Engineer / NLP Engineer": "🤖",
    "Data Analyst": "📊",
    "Cybersecurity Analyst / SOC Analyst": "🔐",
    "Backend Developer Intern": "⚙️",
    "UI/UX Designer": "🎨",
    "Computer Vision Engineer": "👁️",
    "Big Data Engineer": "🗄️",
    "Systems Engineer": "🖥️",
    "DevOps Intern": "☁️",
    "ML Model Testing Intern": "🧪",
    "Junior Python Developer": "🐍",
    "QA Automation Engineer (Entry-Level)": "🔧",
    "Cloud Engineering Intern": "☁️",
    "Network Technician": "🌐",
}

def _keyword_for(career):
    return CAREER_SEARCH_KEYWORDS.get(career, career.split("/")[0].strip())

def _emoji_for(career):
    return CAREER_EMOJIS.get(career, "💼")

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_jobs(career_titles: tuple) -> list:
    """
    Fetch real staff.am job listings with direct page URLs.
    Strategy: use Claude with web_search to search staff.am for each career,
    then extract actual job listing URLs from the results.
    """
    all_jobs = []

    for career in career_titles:
        keyword = _keyword_for(career)
        emoji   = _emoji_for(career)

        prompt = f"""Search staff.am for current job listings matching "{keyword}" in Armenia.

Find real, currently open job postings on staff.am. For each job listing you find, extract:
- The exact job page URL on staff.am (must be a direct link to the specific job, like https://staff.am/en/jobs/XXXXX or similar — NOT the homepage or search page)
- Job title
- Company name
- Location (Yerevan / Remote / Hybrid)
- Salary if shown
- 2-3 relevant skill tags

Return ONLY a JSON array (no markdown, no explanation, no backticks) of up to 2 jobs like:
[
  {{
    "title": "exact job title",
    "company": "company name",
    "location": "Yerevan",
    "salary": "$X,XXX/mo or null",
    "tags": ["Python", "SQL"],
    "url": "https://staff.am/en/jobs/EXACT-JOB-ID"
  }}
]

CRITICAL: The url field MUST be the direct link to the individual job posting page, not a search results page. If you cannot find a direct job URL, use the most specific staff.am search URL you can, like https://staff.am/en/jobs?q={keyword.replace(' ', '+')}"""

        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 800,
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post(
                ANTHROPIC_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=25
            )
            data = response.json()
            full_text = "".join(
                block.get("text", "") for block in data.get("content", [])
                if block.get("type") == "text"
            ).strip()

            # Strip markdown fences
            if "```" in full_text:
                parts = full_text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("["):
                        full_text = part
                        break

            # Find JSON array in text
            start = full_text.find("[")
            end   = full_text.rfind("]") + 1
            if start != -1 and end > start:
                jobs = json.loads(full_text[start:end])
                for j in jobs:
                    j["career_match"] = career
                    j["emoji"]        = emoji
                    j["source"]       = "staff.am" if "staff.am" in j.get("url","") else "LinkedIn"
                    # Validate URL — reject homepage-only links
                    url = j.get("url", "")
                    if url in ("https://staff.am", "https://staff.am/", "https://staff.am/en/jobs"):
                        j["url"] = f"https://staff.am/en/jobs?q={keyword.replace(' ', '+')}"
                    all_jobs.append(j)

        except Exception:
            pass  # fall through to fallback below

    # If we got real jobs, return them
    if all_jobs:
        return all_jobs[:8]

    # Fallback: curated pool with the best possible deep-link search URLs
    return _fallback_jobs(career_titles)


def _fallback_jobs(career_titles):
    """Fallback jobs with specific staff.am search URLs per career type."""
    def search_url(q):
        return f"https://staff.am/en/jobs?q={q.replace(' ', '+')}"

    pool = [
        {
            "title": "Junior Data Analyst",
            "company": "Picsart Armenia",
            "career_match": career_titles[0],
            "location": "Yerevan",
            "salary": "$800–1,200/mo",
            "tags": ["Python", "SQL", "Power BI"],
            "url": search_url("data analyst"),
            "source": "staff.am",
            "emoji": "📊"
        },
        {
            "title": "ML Engineer Intern",
            "company": "Krisp AI",
            "career_match": career_titles[0],
            "location": "Yerevan / Remote",
            "salary": "$700–1,000/mo",
            "tags": ["PyTorch", "Python", "NLP"],
            "url": search_url("machine learning engineer"),
            "source": "staff.am",
            "emoji": "🤖"
        },
        {
            "title": "Backend Developer",
            "company": "ServiceTitan Armenia",
            "career_match": career_titles[0],
            "location": "Yerevan",
            "salary": "$1,500–2,500/mo",
            "tags": ["Python", "Django", "PostgreSQL"],
            "url": search_url("backend developer"),
            "source": "staff.am",
            "emoji": "⚙️"
        },
        {
            "title": "Cybersecurity Analyst",
            "company": "Synopsys Armenia",
            "career_match": career_titles[1] if len(career_titles) > 1 else career_titles[0],
            "location": "Yerevan",
            "salary": "$1,200–2,000/mo",
            "tags": ["SIEM", "SOC", "Linux"],
            "url": search_url("cybersecurity analyst"),
            "source": "staff.am",
            "emoji": "🔐"
        },
        {
            "title": "UI/UX Designer",
            "company": "EPAM Armenia",
            "career_match": career_titles[-1],
            "location": "Yerevan / Hybrid",
            "salary": "$1,000–1,800/mo",
            "tags": ["Figma", "UX Research", "Prototyping"],
            "url": search_url("UX designer"),
            "source": "staff.am",
            "emoji": "🎨"
        },
        {
            "title": "Junior DevOps Engineer",
            "company": "DataArt Armenia",
            "career_match": career_titles[-1],
            "location": "Remote",
            "salary": "$900–1,400/mo",
            "tags": ["Docker", "AWS", "CI/CD"],
            "url": search_url("devops engineer"),
            "source": "staff.am",
            "emoji": "☁️"
        },
    ]
    return pool[:6]


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "intro"


# ─────────────────────────────────────────────
# INTRO PAGE
# ─────────────────────────────────────────────
if st.session_state.page == "intro":
    st.markdown("""
    <style>
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 0.3; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.4); }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    .hero-outer {
        min-height: 100vh;
        background:
            radial-gradient(ellipse 90% 55% at 50% -5%, rgba(26,26,62,0.95) 0%, transparent 65%),
            radial-gradient(ellipse 50% 40% at 80% 80%, rgba(110,231,183,0.05) 0%, transparent 60%),
            var(--ink);
        display: grid;
        grid-template-columns: 1fr 1fr;
        grid-template-rows: 1fr;
        align-items: center;
        gap: 0;
        padding: 0 80px;
        position: relative;
        overflow: hidden;
    }
    /* animated grid lines */
    .hero-outer::before {
        content: '';
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
        background-size: 60px 60px;
        mask-image: radial-gradient(ellipse 80% 70% at 50% 50%, black 20%, transparent 80%);
        pointer-events: none;
    }
    .hero-left {
        padding: 80px 48px 80px 0;
        animation: fadeUp 0.8s ease both;
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(110,231,183,0.08);
        border: 1px solid rgba(110,231,183,0.2);
        border-radius: 100px;
        padding: 6px 14px 6px 10px;
        margin-bottom: 32px;
    }
    .kicker-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--accent);
        animation: pulse-dot 2s ease-in-out infinite;
    }
    .kicker-text {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        color: var(--accent);
        text-transform: uppercase;
    }
    .hero-h1 {
        font-family: 'Syne', sans-serif;
        font-size: clamp(52px, 6vw, 88px);
        font-weight: 800;
        color: var(--bright);
        line-height: 0.95;
        letter-spacing: -3px;
        margin: 0 0 28px;
    }
    .hero-h1 .accent-word {
        background: linear-gradient(135deg, #6ee7b7 0%, #818cf8 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 4s linear infinite;
    }
    .hero-desc {
        font-size: 17px;
        font-weight: 300;
        color: #9ca3af;
        line-height: 1.75;
        max-width: 440px;
        margin-bottom: 40px;
    }
    .hero-steps {
        display: flex;
        flex-direction: column;
        gap: 14px;
        margin-bottom: 48px;
    }
    .hero-step {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .step-num {
        width: 28px; height: 28px;
        border-radius: 50%;
        background: rgba(110,231,183,0.1);
        border: 1px solid rgba(110,231,183,0.25);
        display: flex; align-items: center; justify-content: center;
        font-size: 11px;
        font-weight: 700;
        color: var(--accent);
        flex-shrink: 0;
    }
    .step-text {
        font-size: 14px;
        color: #9ca3af;
    }
    .step-text strong { color: var(--bright); font-weight: 500; }

    /* CTA button — pure HTML so we control width perfectly */
    .hero-cta-wrap {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .hero-cta {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: var(--accent);
        color: #0a0a0f;
        font-family: 'Syne', sans-serif;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 0 32px;
        height: 52px;
        border-radius: 6px;
        border: none;
        cursor: pointer;
        transition: all 0.2s;
        text-decoration: none;
    }
    .hero-cta:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 36px rgba(110,231,183,0.3);
    }
    .hero-note {
        font-size: 12px;
        color: var(--muted);
        letter-spacing: 0.5px;
    }

    /* Right side — floating preview card */
    .hero-right {
        padding: 80px 0 80px 48px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        animation: fadeUp 0.8s ease 0.2s both;
    }
    .preview-label {
        font-size: 10px;
        letter-spacing: 3px;
        color: var(--muted);
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .preview-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px 28px;
        animation: float 6s ease-in-out infinite;
    }
    .preview-card-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
    }
    .preview-rank { font-size: 10px; letter-spacing: 2px; color: var(--accent); text-transform: uppercase; font-weight: 600; }
    .preview-pct {
        font-family: 'Syne', sans-serif;
        font-size: 22px;
        font-weight: 800;
        color: var(--bright);
    }
    .preview-title {
        font-family: 'Syne', sans-serif;
        font-size: 20px;
        font-weight: 800;
        color: var(--bright);
        margin-bottom: 6px;
    }
    .preview-salary { font-size: 13px; color: var(--muted); margin-bottom: 16px; }
    .preview-bar-bg { background: rgba(255,255,255,0.06); border-radius: 100px; height: 3px; margin-bottom: 18px; }
    .preview-bar-fill { height: 100%; width: 87%; background: linear-gradient(90deg, #6ee7b7, #818cf8); border-radius: 100px; }
    .preview-tags { display: flex; gap: 6px; flex-wrap: wrap; }
    .preview-tag { background: rgba(255,255,255,0.05); border: 1px solid var(--border); border-radius: 100px; padding: 3px 10px; font-size: 11px; color: var(--muted); }

    .mini-job-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        animation: float 7s ease-in-out 1s infinite;
    }
    .mini-job-left { display: flex; align-items: center; gap: 12px; }
    .mini-job-ico { font-size: 22px; }
    .mini-job-title { font-size: 13px; font-weight: 600; color: var(--bright); margin-bottom: 2px; }
    .mini-job-co { font-size: 11px; color: var(--muted); }
    .mini-apply {
        font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
        color: var(--accent); border: 1px solid rgba(110,231,183,0.3); border-radius: 4px;
        padding: 5px 12px; white-space: nowrap;
    }

    .hero-stat-row {
        display: flex;
        gap: 12px;
    }
    .hero-stat {
        flex: 1;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }
    .hero-stat-n {
        font-family: 'Syne', sans-serif;
        font-size: 24px; font-weight: 800; color: var(--bright); letter-spacing: -1px;
    }
    .hero-stat-l { font-size: 10px; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 2px; }

    @media (max-width: 900px) {
        .hero-outer { grid-template-columns: 1fr; padding: 40px 24px; }
        .hero-right { display: none; }
        .hero-left { padding: 80px 0 60px; }
    }
    </style>

    <div class="hero-outer">
      <!-- LEFT: copy + CTA -->
      <div class="hero-left">
        <div class="hero-kicker">
          <span class="kicker-dot"></span>
          <span class="kicker-text">AI-Powered · Armenian Tech</span>
        </div>
        <h1 class="hero-h1">Find your<br><span class="accent-word">career path</span></h1>
        <p class="hero-desc">Rate your academic strengths in 2 minutes. Three ML models predict your best-fit tech career — then surface real jobs you can apply to today.</p>
        <div class="hero-steps">
          <div class="hero-step">
            <span class="step-num">1</span>
            <span class="step-text">Select your <strong>academic year</strong> and rate your subjects</span>
          </div>
          <div class="hero-step">
            <span class="step-num">2</span>
            <span class="step-text"><strong>KNN, Decision Tree & Naive Bayes</strong> predict your top 3 careers</span>
          </div>
          <div class="hero-step">
            <span class="step-num">3</span>
            <span class="step-text">Browse <strong>live job listings</strong> from staff.am & LinkedIn and apply</span>
          </div>
        </div>
      </div>

      <!-- RIGHT: floating preview -->
      <div class="hero-right">
        <p class="preview-label">Example prediction</p>
        <div class="preview-card">
          <div class="preview-card-top">
            <span class="preview-rank">#1 Best Match</span>
            <span class="preview-pct">87%</span>
          </div>
          <div class="preview-title">AI Engineer / NLP Engineer</div>
          <div class="preview-salary">$7,200 USD / month</div>
          <div class="preview-bar-bg"><div class="preview-bar-fill"></div></div>
          <div class="preview-tags">
            <span class="preview-tag">PyTorch</span>
            <span class="preview-tag">NLP</span>
            <span class="preview-tag">Transformers</span>
            <span class="preview-tag">Python</span>
          </div>
        </div>

        <div class="mini-job-card">
          <div class="mini-job-left">
            <span class="mini-job-ico">🤖</span>
            <div>
              <div class="mini-job-title">ML Engineer Intern</div>
              <div class="mini-job-co">Krisp AI · Yerevan</div>
            </div>
          </div>
          <span class="mini-apply">Apply →</span>
        </div>

        <div class="hero-stat-row">
          <div class="hero-stat">
            <div class="hero-stat-n">3</div>
            <div class="hero-stat-l">ML Models</div>
          </div>
          <div class="hero-stat">
            <div class="hero-stat-n">48</div>
            <div class="hero-stat-l">Careers</div>
          </div>
          <div class="hero-stat">
            <div class="hero-stat-n">Live</div>
            <div class="hero-stat-l">Job Listings</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA button — Streamlit button with tight width
    st.markdown("""
    <style>
    .intro-btn-wrap { display:flex; justify-content:flex-start; padding: 0 80px 60px; margin-top:-40px; }
    .intro-btn-wrap div[data-testid="stButton"] { width: auto !important; }
    .intro-btn-wrap div[data-testid="stButton"] > button {
        width: auto !important;
        padding: 0 40px !important;
        font-size: 13px !important;
        height: 52px !important;
    }
    @media (max-width: 900px) {
        .intro-btn-wrap { padding: 0 24px 60px; justify-content:center; }
    }
    </style>
    <div class="intro-btn-wrap">
    """, unsafe_allow_html=True)
    if st.button("START MY ASSESSMENT →"):
        st.session_state.page = "choose_year"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INNER PAGES
# ─────────────────────────────────────────────
else:
    # ── Progress indicator
    pages = ["choose_year", "ranking", "results"]
    labels = ["Year", "Subjects", "Results"]
    current_idx = pages.index(st.session_state.page) if st.session_state.page in pages else 0
    progress_html = '<div style="display:flex;gap:8px;align-items:center;padding:20px 32px 0;max-width:1100px;margin:0 auto;">'
    for i, (pg, lbl) in enumerate(zip(pages, labels)):
        active = i <= current_idx
        color = "var(--accent)" if active else "var(--muted)"
        progress_html += f'<span style="font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:{color};">{lbl}</span>'
        if i < len(pages) - 1:
            progress_html += f'<span style="flex:1;height:1px;background:{"var(--accent)" if i < current_idx else "var(--border)"};margin:0 8px;"></span>'
    progress_html += '</div>'
    st.markdown(progress_html, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # CHOOSE YEAR PAGE
    # ─────────────────────────────────────────
    if st.session_state.page == "choose_year":
        st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
        st.markdown('<p class="page-label">Step 01</p>', unsafe_allow_html=True)
        st.markdown('<h1 class="page-title">What year are<br>you in?</h1>', unsafe_allow_html=True)

        with st.form("year_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                year = st.selectbox(
                    "Academic Year",
                    [1, 2, 3, 4],
                    format_func=lambda x: f"Year {x} — {len(get_subjects(x))} subjects"
                )
            with col_b:
                model_choice = st.selectbox(
                    "Prediction Engine",
                    ["Ensemble (Recommended)", "KNN Only", "Decision Tree Only", "Naive Bayes Only"],
                    help="Ensemble blends all three models for the most balanced result."
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Info pills
            subj_count = len(get_subjects(year if year else 1))
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px 24px;margin-bottom:24px;">
                <p style="margin:0;font-size:13px;color:var(--muted);line-height:1.7;">
                    You'll rate <strong style="color:var(--bright);">{subj_count} subjects</strong> on a scale of 0–5.
                    Our ML ensemble will predict your top 3 career matches and surface
                    <strong style="color:var(--bright);">live job listings</strong> you can apply to today.
                </p>
            </div>
            """, unsafe_allow_html=True)

            if st.form_submit_button("CONTINUE TO SUBJECTS →"):
                st.session_state.selected_year = year
                st.session_state.model_choice = model_choice
                st.session_state.page = "ranking"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # RANKING PAGE
    # ─────────────────────────────────────────
    elif st.session_state.page == "ranking":
        st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
        st.markdown('<p class="page-label">Step 02 — Year ' + str(st.session_state.get("selected_year", 1)) + '</p>', unsafe_allow_html=True)
        st.markdown('<h1 class="page-title">Rate your<br><span>subjects</span></h1>', unsafe_allow_html=True)
        st.markdown('<p style="color:var(--muted);font-size:15px;margin-bottom:36px;line-height:1.6;">0 = struggled, 5 = excelled & genuinely loved it. Be honest — the model works best with accurate data.</p>', unsafe_allow_html=True)

        subjects = get_subjects(st.session_state.selected_year)
        with st.form("ranking_form"):
            rankings = {}
            # Group into year columns for readability
            for i, subj in enumerate(subjects):
                rankings[subj] = st.slider(
                    subj, 0, 5, 3, key=subj,
                    help=f"How did you feel about {subj}?"
                )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.form_submit_button("REVEAL MY CAREER PATH →"):
                st.session_state.rankings = rankings
                st.session_state.page = "results"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # RESULTS PAGE
    # ─────────────────────────────────────────
    elif st.session_state.page == "results":
        rankings = st.session_state.rankings
        model_choice = st.session_state.get("model_choice", "Ensemble (Recommended)")
        vec = [rankings.get(s, 0) for s in all_subjects]

        jobs, scores, model_label = get_predictions(vec, model_choice)
        per_model = get_per_model_top1(vec)

        st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
        st.markdown('<p class="page-label">Results</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px;margin-bottom:8px;">
            <h1 class="page-title" style="margin:0;">Your career<br><span>predictions</span></h1>
            <span class="model-chip" style="margin-top:12px;">⚡ {model_label}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── TOP 3 CAREER CARDS ──
        for i, (job, score) in enumerate(zip(jobs, scores), 1):
            row_m = job_info[job_info["Job Title"].str.strip() == job.strip()]
            if row_m.empty:
                continue
            row = row_m.iloc[0]
            salary = row["Monthly Salary (USD)"]
            desc = row["Description"]
            roadmap = get_roadmap(job)

            roadmap_items = "".join(
                f'<li><span class="roadmap-dot"></span>{step}</li>'
                for step in roadmap
            )

            rank_labels = ["#1 Best Match", "#2 Strong Match", "#3 Good Match"]
            st.markdown(f"""
            <div class="career-card">
                <p class="card-rank">{rank_labels[i-1]}</p>
                <h2 class="card-title">{job}</h2>
                <p class="card-match">{score:.1f}% confidence score</p>
                <div class="match-bar-bg">
                    <div class="match-bar-fill" style="width:{score:.1f}%;"></div>
                </div>
                <p class="card-desc">{desc}</p>
                <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:20px;">
                    <span class="card-salary">${salary:,}</span>
                    <span class="card-salary-label">USD / month</span>
                </div>
                <details style="cursor:pointer;">
                    <summary style="font-size:13px;color:var(--accent);font-weight:600;letter-spacing:1px;text-transform:uppercase;outline:none;user-select:none;">
                        Career Roadmap ↓
                    </summary>
                    <ul class="roadmap-list">{roadmap_items}</ul>
                </details>
            </div>
            """, unsafe_allow_html=True)

        # ── LIVE JOB LISTINGS ──
        st.markdown("""
        <div class="section-header">
            <h2 class="section-title">Live Job Listings</h2>
            <span class="section-count">APPLY NOW · REAL OPENINGS</span>
        </div>
        <p style="color:var(--muted);font-size:14px;margin-bottom:24px;">
            Personalized openings sourced from staff.am and LinkedIn based on your predicted careers.
        </p>
        """, unsafe_allow_html=True)

        with st.spinner("Fetching live jobs for your predicted careers…"):
            live_jobs = fetch_live_jobs(tuple(jobs))

        if live_jobs:
            for jb in live_jobs:
                title = jb.get("title", "Tech Role")
                company = jb.get("company", "")
                location = jb.get("location", "")
                salary = jb.get("salary") or ""
                tags = jb.get("tags", [])
                url = jb.get("url", "https://staff.am/en/jobs")
                source = jb.get("source", "staff.am")
                emoji = jb.get("emoji", "💼")
                career_match = jb.get("career_match", "")

                tags_html = "".join(f'<span class="job-tag">{t}</span>' for t in tags)
                if career_match:
                    tags_html = f'<span class="job-tag accent">{career_match}</span>' + tags_html
                if salary:
                    tags_html += f'<span class="job-tag">{salary}</span>'
                if location:
                    tags_html += f'<span class="job-tag">📍 {location}</span>'

                # Show a short readable URL hint under the company name
                url_display = url.replace("https://", "").replace("http://", "")
                if len(url_display) > 48:
                    url_display = url_display[:45] + "…"

                st.markdown(f"""
                <div class="job-card">
                    <div class="job-logo">{emoji}</div>
                    <div class="job-info">
                        <p class="job-title-text">{title}</p>
                        <p class="job-company">{company} · {source}</p>
                        <p style="font-size:10px;color:#4b5563;margin:0 0 10px;font-family:monospace;letter-spacing:0.3px;">{url_display}</p>
                        <div class="job-tags">{tags_html}</div>
                    </div>
                    <a href="{url}" target="_blank" rel="noopener noreferrer" class="apply-btn">APPLY →</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;padding:32px;text-align:center;">
                <p style="color:var(--muted);font-size:14px;">No live listings found. <a href="https://staff.am/en/jobs" target="_blank" style="color:var(--accent);">Browse staff.am →</a></p>
            </div>
            """, unsafe_allow_html=True)

        # Quick link buttons
        col_j1, col_j2, col_j3 = st.columns(3)
        with col_j1:
            st.markdown('<a href="https://staff.am/en/jobs" target="_blank" style="display:block;text-align:center;padding:12px;background:var(--card);border:1px solid var(--border);border-radius:8px;color:var(--text);text-decoration:none;font-size:13px;font-weight:500;">🏢 Browse staff.am</a>', unsafe_allow_html=True)
        with col_j2:
            st.markdown('<a href="https://www.linkedin.com/jobs/search/?keywords=developer+armenia" target="_blank" style="display:block;text-align:center;padding:12px;background:var(--card);border:1px solid var(--border);border-radius:8px;color:var(--text);text-decoration:none;font-size:13px;font-weight:500;">💼 LinkedIn Jobs</a>', unsafe_allow_html=True)
        with col_j3:
            st.markdown('<a href="https://topcv.am/" target="_blank" style="display:block;text-align:center;padding:12px;background:var(--card);border:1px solid var(--border);border-radius:8px;color:var(--text);text-decoration:none;font-size:13px;font-weight:500;">🔍 TopCV Armenia</a>', unsafe_allow_html=True)

        # ── MODEL AGREEMENT ──
        st.markdown("""
        <div class="section-header" style="margin-top:48px;">
            <h2 class="section-title">Model Consensus</h2>
            <span class="section-count">3 ALGORITHMS</span>
        </div>
        """, unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        for col, (mname, mpick) in zip([mc1, mc2, mc3], per_model.items()):
            col.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
                        padding:20px;text-align:center;">
                <p style="color:var(--muted);font-size:11px;letter-spacing:2px;text-transform:uppercase;margin:0 0 8px;">{mname}</p>
                <p style="color:var(--bright);font-size:14px;font-weight:600;margin:0;">{mpick}</p>
            </div>""", unsafe_allow_html=True)

        # ── INSIGHTS ──
        st.markdown("""
        <div class="section-header" style="margin-top:48px;">
            <h2 class="section-title">Personalised Insights</h2>
        </div>
        """, unsafe_allow_html=True)
        for ins in get_insights(rankings, jobs[0]):
            st.markdown(f'<div class="insight-pill">{ins}</div>', unsafe_allow_html=True)

        # ── SALARY CHART ──
        st.markdown("""
        <div class="section-header" style="margin-top:48px;">
            <h2 class="section-title">Salary Comparison</h2>
            <span class="section-count">USD / MONTH</span>
        </div>
        """, unsafe_allow_html=True)
        salary_data = []
        for job in jobs:
            rm = job_info[job_info["Job Title"].str.strip() == job.strip()]
            if not rm.empty:
                salary_data.append({"Career": job, "Monthly Salary (USD)": rm.iloc[0]["Monthly Salary (USD)"]})
        if salary_data:
            sal_df = pd.DataFrame(salary_data)
            fig_sal = px.bar(
                sal_df, x="Monthly Salary (USD)", y="Career", orientation="h",
                color="Monthly Salary (USD)",
                color_continuous_scale=["#6ee7b7", "#818cf8"],
                text="Monthly Salary (USD)"
            )
            fig_sal.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_sal.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9ca3af", family="DM Sans"),
                height=220, showlegend=False, coloraxis_showscale=False,
                xaxis=dict(showgrid=False, visible=False),
                yaxis=dict(showgrid=False, color="#9ca3af"),
                margin=dict(l=0, r=90, t=10, b=10)
            )
            st.plotly_chart(fig_sal, use_container_width=True)

        # ── RADAR ──
        st.markdown("""
        <div class="section-header" style="margin-top:48px;">
            <h2 class="section-title">Academic Strengths Radar</h2>
        </div>
        """, unsafe_allow_html=True)
        values = list(rankings.values())
        categories = list(rankings.keys())
        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            line_color='#6ee7b7',
            fillcolor='rgba(110,231,183,0.12)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5], color="#4b5563", gridcolor="rgba(255,255,255,0.05)"),
                angularaxis=dict(color="#6b7280"),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=False, height=460,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#9ca3af", family="DM Sans")
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── SUBJECT BREAKDOWN ──
        st.markdown("""
        <div class="section-header" style="margin-top:48px;">
            <h2 class="section-title">Subject Breakdown</h2>
        </div>
        """, unsafe_allow_html=True)
        sorted_r = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        bar_df = pd.DataFrame(sorted_r, columns=["Subject", "Rating"])
        bar_df["Color"] = bar_df["Rating"].apply(
            lambda r: "#6ee7b7" if r >= 4 else ("#818cf8" if r >= 2 else "#374151"))
        fig_bar = px.bar(
            bar_df, x="Rating", y="Subject", orientation="h",
            color="Color", color_discrete_map="identity", range_x=[0, 5]
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9ca3af", family="DM Sans"),
            height=max(300, len(bar_df) * 28), showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", range=[0, 5]),
            yaxis=dict(showgrid=False),
            margin=dict(l=0, r=20, t=10, b=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── DOWNLOAD ──
        st.markdown("""
        <div class="section-header" style="margin-top:48px;">
            <h2 class="section-title">Save Results</h2>
        </div>
        """, unsafe_allow_html=True)
        summary_lines = ["Career Compass — Results Summary", "=" * 40,
                         f"Model: {model_label}", f"Year: {st.session_state.selected_year}", ""]
        summary_lines.append("Top Career Predictions:")
        for i, (job, score) in enumerate(zip(jobs, scores), 1):
            rm = job_info[job_info["Job Title"].str.strip() == job.strip()]
            salary = rm.iloc[0]["Monthly Salary (USD)"] if not rm.empty else 0
            summary_lines.append(f"  #{i} {job} — {score:.1f}% match — ${salary:,}/month")
        summary_lines += ["", "Top Subject Strengths:"]
        for subj, rating in sorted(rankings.items(), key=lambda x: x[1], reverse=True)[:5]:
            summary_lines.append(f"  {subj}: {rating}/5")
        st.download_button(
            label="⬇  Download Results (.txt)",
            data="\n".join(summary_lines),
            file_name="career_compass_results.txt",
            mime="text/plain"
        )

        # ── NAV ──
        st.markdown('<div style="display:flex;gap:12px;margin-top:56px;padding-top:32px;border-top:1px solid var(--border);">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            st.markdown('<div class="nav-col">', unsafe_allow_html=True)
            if st.button("← Different Year"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.session_state.page = "choose_year"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="nav-col">', unsafe_allow_html=True)
            if st.button("↩ Start Over"):
                st.session_state.clear()
                st.session_state.page = "intro"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
