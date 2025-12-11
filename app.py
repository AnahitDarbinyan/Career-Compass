import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go
import base64

# BACKGROUND
def set_background(opacity=1.0):
    with open("assets/main_background.jpg", "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()
    overlay = f"rgba(0,0,0,{1 - opacity + 0.3})"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient({overlay}, {overlay}),
                        url("data:image/jpeg;base64,{img_base64}") 
                        center/cover fixed no-repeat;
        }}
        </style>
        """, unsafe_allow_html=True
    )

st.set_page_config(page_title="Career Compass", layout="centered")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
    
    .title {
        font-family: 'Orbitron', sans-serif;
        font-size: 110px;
        font-weight: 900;
        color: white;
        text-align: center;
        letter-spacing: 12px;
        margin: 100px 0 20px 0;
        text-shadow: 0 0 50px #a78bfa, 0 0 100px #7c3aed;
    }
    .subtitle {
        font-size: 28px;
        color: #e0e7ff;
        text-align: center;
        margin-bottom: 100px;
        letter-spacing: 3px;
    }
    .big-button {
        display: flex;
        justify-content: center;
        margin: 60px 0;
    }
    div[data-testid="stButton"] > button {
        background: white !important;
        color: #581c87 !important;
        font-size: 30px !important;
        font-weight: 900 !important;
        height: 90px !important;
        width: 480px !important;
        border-radius: 80px !important;
        border: none !important;
        box-shadow: 0 15px 40px rgba(0,0,0,0.6) !important;
        letter-spacing: 4px;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-8px) scale(1.05) !important;
        box-shadow: 0 30px 70px rgba(124,62,173,0.8) !important;
    }
    .big-title {
        font-size: 58px;
        text-align: center;
        color: white;
        margin: 80px 0 40px 0;
    }
    .radar-title {
        text-align: center !important;
        font-size: 28px !important;
        color: white !important;
        margin: 40px 0 20px 0;
    }
</style>
""", unsafe_allow_html=True)

df = pd.read_csv("dataset.csv")
job_info = pd.read_csv("job_data.csv")

df.columns = df.columns.str.strip()
job_info.columns = job_info.columns.str.strip()

def clean_salary(x):
    return int(str(x).replace('$', '').replace(',', '').strip())

job_info["Monthly Salary (USD)"] = job_info["Monthly Salary (USD)"].apply(clean_salary)

# YEAR SUBJECTS
year_subjects = {
    1: ["Calculus", "Analytic Geometry And Algebra", "Descriptive Statistics", "Introduction To Algorithms", 
        "Generative AI", "Probability Theory", "Discrete Math", "Python", "Introduction To Low Code"],
    2: ["Applied Statistics", "C Programming", "Computer Science", "Physics", "Management", "Numerical Methods", 
        "Logic", "Information Theory", "Architecture", "Algorithms", "C Sharp"],
    3: ["Systems", "Math For CS", "Computer Networks", "Human Computer Interaction", "Machine Learning", 
        "Complexity", "Databases", "OOP", "Data Structures"],
    4: ["Systems Programming", "Parallel Programming", "Functional Programming & ADT", "Probability And Statistics", 
        "Graph Theory", "Theory Of Automata", "Computer Graphics", "Image Processing", 
        "Advanced Functional Programming & ADT", "Signal Processing", "Software Engineering", "Computer Security", "Artificial Intelligence"]
}

def get_subjects(year):
    return sum((year_subjects.get(y, []) for y in range(1, year + 1)), [])

all_subjects = [c for c in df.columns if c != "Job_Title"]
X = df[all_subjects].values
y = df["Job_Title"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
knn.fit(X_scaled, y_enc)

if "page" not in st.session_state:
    st.session_state.page = "intro"

# MAIN PAGE
if st.session_state.page == "intro":
    set_background(opacity=1.0)

    st.markdown('<h1 class="title">CAREER COMPASS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Find the tech career you were born to lead</p>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    div[data-testid="stButton"] > button {
        width: 560px !important;
        height: 100px !important;
        font-size: 34px !important;
        font-weight: 900 !important;
        color: #581c87 !important;
        background-color: white !important;
        border-radius: 80px !important;
        box-shadow: 0 15px 50px rgba(0,0,0,0.7) !important;
        margin: 100px auto 0 auto;
        display: block;
        letter-spacing: 5px;
        font-family: 'Orbitron', sans-serif;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 35px 80px rgba(124,62,173,0.9);
    }
    </style>
    """, unsafe_allow_html=True)


    if st.button("BEGIN YOUR JOURNEY"):
        st.session_state.page = "choose_year"
        st.rerun()

# OTHER PAGES
else:
    set_background(opacity=0.7)

    if st.session_state.page == "choose_year":
        st.markdown("<h1 class='big-title'>Select Your Year</h1>", unsafe_allow_html=True)
        with st.form("year_form"):
            _, col, _ = st.columns([1,1.5,1])
            with col:
                year = st.selectbox("Current Academic Year", [1,2,3,4], format_func=lambda x: f"Year {x}")
            if st.form_submit_button("CONTINUE", use_container_width=True):
                st.session_state.selected_year = year
                st.session_state.page = "ranking"
                st.rerun()

    elif st.session_state.page == "ranking":
        st.markdown("<h1 class='big-title' style='color:#e0e7ff;'>Rate Your Subjects</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#ccc; font-size:21px;'>0 = Struggled • 5 = Excelled & Loved</p>", unsafe_allow_html=True)

        subjects = get_subjects(st.session_state.selected_year)
        with st.form("ranking_form"):
            rankings = {}
            for subj in subjects:
                rankings[subj] = st.slider(subj, 0, 5, 3, key=subj)

            if st.form_submit_button("REVEAL MY CAREER PATH", use_container_width=True):
                st.session_state.rankings = rankings
                st.session_state.page = "results"
                st.rerun()

    elif st.session_state.page == "results":
        st.markdown("<h1 class='big-title' style='color:#d8b4fe;'>Your Top 3 Career Predictions</h1>", unsafe_allow_html=True)

        vec = [st.session_state.rankings.get(s, 0) for s in all_subjects]
        probs = knn.predict_proba(scaler.transform([vec]))[0]
        top3 = np.argsort(probs)[-3:][::-1]
        jobs = le.inverse_transform(top3)
        scores = (probs[top3] * 100)

        for i, (job, score) in enumerate(zip(jobs, scores), 1):
            row = job_info[job_info["Job Title"].str.strip() == job.strip()].iloc[0]
            desc = row["Description"]
            salary = row["Monthly Salary (USD)"]

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #4b0082, #6a0dad); 
                border-radius: 20px; 
                padding: 22px; 
                margin: 25px auto;
                max-width: 780px; 
                color: white; 
                box-shadow: 0 12px 35px rgba(0,0,0,0.7);">
                <h2 style="text-align:center; margin:0; font-size:34px;">#{i} → {job}</h2>
                <p style="text-align:center; font-size:25px; margin:12px 0 18px; font-weight:bold;">
                    Match: {score:.1f}%
                </p>
                <div style="
                        background: rgba(0,0,0,0.15); 
                        border-radius: 12px; 
                        padding: 16px;
                ">
                    <p style="margin:0; font-size:17px; line-height:1.6; color: #e0d4f7;">{desc}</p>
                </div>
                <p style="text-align:center; font-size:28px; margin:20px 0 0;">
                    <strong>${salary:,} USD/month</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)


        
        st.markdown("<h3 class='radar-title'>Your Academic Superpower Radar</h3>", unsafe_allow_html=True)

        values = list(st.session_state.rankings.values())
        categories = list(st.session_state.rankings.keys())
        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            line_color='#e0aaff',
            fillcolor='rgba(167,139,250,0.5)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,5])),
            showlegend=False,
            height=520,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <style>
        div[data-testid="stButton"] > button.start-over-btn {
            width: 560px !important;
            height: 100px !important;
            font-size: 34px !important;
            font-weight: 900 !important;
            color: #581c87 !important;
            background-color: white !important;
            border-radius: 80px !important;
            box-shadow: 0 15px 50px rgba(0,0,0,0.7) !important;
            margin: 60px auto 100px auto;
            display: block;
            letter-spacing: 5px;
            font-family: 'Orbitron', sans-serif;

        }
        div[data-testid="stButton"] > button.start-over-btn:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 35px 80px rgba(124,62,173,0.9);
        }
        </style>
        """, unsafe_allow_html=True)

        if st.button("Start Over", key="start_over", help="Restart the app"):
            st.session_state.clear()
            st.session_state.page = "intro"
            st.rerun()
