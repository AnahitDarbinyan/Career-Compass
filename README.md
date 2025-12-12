# 🎯 Career Compass – AI-Driven Career Guidance System

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-orange.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**Career Compass** is an interactive AI-driven application designed to help students discover suitable career paths based on their academic strengths. By combining **Machine Learning** models with an intuitive **Streamlit** interface, users receive personalized career recommendations along with detailed job descriptions, monthly salaries, and a visualization of their strongest subjects.


## Features

- 🎓 Predicts top 3 career options based on subject ratings  
- 📊 Uses Machine Learning models: **K-Nearest Neighbors**, **Decision Tree**, and **Naive Bayes**  
- 📖 Displays detailed career information: job title, description, salary, and match percentage  
- 📈 Strength radar chart visualizing academic strengths  
- 💻 Interactive **Streamlit** interface with dynamic pages and state management  

## Tech Stack

- **Python 3.8+**  
- **Streamlit** for UI and interactive pages  
- **Pandas, NumPy** for data processing  
- **Scikit-learn** for Machine Learning models  
- **Plotly** for visualizations  

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

## Future Enhancements
- Expand the dataset to include more subjects and career options
- Integrate real student data for better prediction accuracy
- Provide career guidance tips and resources alongside predictions
- Improve UI/UX with additional interactive visualizations

## Contribution
Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss the modifications.

## Acknowledgements
- Inspired by career guidance research and AI-driven recommendation systems
- Streamlit documentation and tutorials for interactive app development
