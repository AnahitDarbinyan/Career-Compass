# 🎯 Career Compass – AI-Driven Career Guidance Platform

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.30-orange.svg)](https://streamlit.io/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![App Preview](assets/App%20Preview.png)

_Discover your ideal career paths, visualize your strengths, and explore real-time job listings with Career Compass._


---

## Overview

**Career Compass** is an AI-driven, interactive platform that helps students and young professionals discover the most suitable career paths based on their academic strengths, skills, and personal preferences.  

With a **redesigned dark editorial aesthetic**, personalized job listings, and dynamic visualizations, users can:

- Receive **Top 3 career predictions**  
- Explore **detailed job descriptions, salaries, and match scores**  
- Access **live job listings** tailored to predicted careers  

The platform combines **Machine Learning models**, **Streamlit UI**, and the **Anthropic API** to make career guidance engaging, interactive, and realistic.  

---

## Key Features

- 🎓 **Top 3 career predictions** based on academic and skill ratings  
- 📊 Machine Learning models: **K-Nearest Neighbors (KNN)**, **Decision Tree**, **Naive Bayes**, and **Ensemble**  
- 📖 Detailed career information: job title, description, salary, and match percentage  
- 📈 **Strength radar chart** visualizing subjects and skills  
- 💻 **Redesigned UI**: clean dark editorial aesthetic, Syne headings, DM Sans body, emerald and indigo accents, hover animations, and top-border gradient on cards  
- 🚀 **Progress bar navigation**: Year → Subjects → Results  
- 📊 **Charts re-themed** for dark palette  
- 🌐 **Live Job Listings**: fetches real-time jobs via Anthropic API and Claude from Armenian tech sites (staff.am, LinkedIn, TopCV)  
  - Each card shows: company, role, skill tags, location, salary, and direct APPLY → links  
  - Falls back to curated local jobs if API fails  
  - Quick shortcut buttons to browse staff.am, LinkedIn, and TopCV directly  

---

## Tech Stack

- **Python 3.8+**  
- **Streamlit** for UI and interactive pages  
- **Pandas & NumPy** for data processing  
- **Scikit-learn** for ML models  
- **Plotly** for visualizations  
- **Joblib** for saving/loading models  
- **Anthropic API (Claude)** for live job listings  

---

## Installation

1. Clone the repository:  
```bash
git clone https://github.com/AnahitDarbinyan/Career-Compass.git
cd Career-Compass
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Live Job Listings Feature

After career predictions, the app calls the **Anthropic API** with web search enabled and generates current job listings that match predicted careers:

- **Listings sourced from:** staff.am, LinkedIn, and TopCV.am  
- **Each job card displays:**  
  - Company name  
  - Role/title  
  - Skill tags  
  - Location  
  - Salary (if available)  
  - Direct **APPLY →** link  
- **Fallback:** curated Armenian tech jobs if API fails  
- **Quick access buttons:** browse staff.am, LinkedIn, and TopCV directly  

---

## Future Enhancements

- Expand dataset with more subjects, skills, and career options  
- Integrate real student performance data for improved predictions  
- Add personalized career guidance tips alongside predictions  
- Enhance UI/UX with additional interactive visualizations and dashboards  
- Admin interface to dynamically update career options and curated jobs  

---

## Contribution

Contributions are welcome!  

- Fork the repository  
- Create a new branch for your feature/bugfix  
- Submit a pull request  
- For major changes, open an issue first to discuss  

---

## Acknowledgements

- Inspired by AI-driven career guidance research and recommendation systems  
- Streamlit documentation and tutorials for interactive applications  
- Open-source libraries: **Scikit-learn**, **Plotly**, **Pandas**, **NumPy**  
- Anthropic Claude API for realistic job listing generation  
