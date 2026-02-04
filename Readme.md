🍽️ Smart Swiggy Recommender System
Overview

The Swiggy Restaurant Recommender is an advanced AI-powered web app built with Streamlit that helps users discover restaurants tailored to their preferences. This project integrates hybrid recommendation algorithms, personalized suggestions, ML predictions, Google Maps integration, and an interactive favorites system to mimic a real-world food delivery platform.

This app is ideal for portfolio projects, demonstrating full-stack AI application design, data analysis, and interactive UI development.

Features
✅ Step 1 — Basic + Filters + Banner

Fully responsive Streamlit interface

Banner image and header

Filter restaurants by:

City

Cuisine

Rating

Cost

Top restaurants displayed with:

Name, city, cuisine

Rating and cost

✅ Step 2 — Similarity + Favorites + Personalized

Find similar restaurants using cosine similarity on encoded features

Favorites system to save and view favorite restaurants

Personalized recommendations based on user activity history

Feedback system (like/dislike) to improve suggestions

✅ Step 3 — ML + Google Maps + Images + Insights

ML-based recommendations using Random Forest / XGBoost on restaurant features (optional)

Google Maps integration to view restaurant location and directions

Restaurant cards include food images

Dataset insights using Seaborn visualizations

Top 10 restaurants summary with rating, city, cuisine, and cost

Installation

Clone the repository:

git clone https://github.com/yourusername/swiggy-recommender.git
cd swiggy-recommender


Create and activate a Python virtual environment:

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

How to Run
streamlit run app.py


Navigate through the sidebar to access:

Browse Restaurants

Find Similar Restaurants

Favorites

AI Recommendations

Insights

File Structure
swiggy-recommender/
│
├─ app.py                  # Main Streamlit application
├─ swiggy_cleaned_data.csv # Cleaned restaurant dataset
├─ swiggy_encoded_data.csv # Encoded features for similarity
├─ models/                 # ML models and preprocessing pickle files
├─ images/                 # Sample restaurant images
├─ requirements.txt        # Python dependencies
└─ README.md               # Project documentation

Technologies Used

Python 3.10+

Streamlit — Web app framework

Pandas / Numpy — Data manipulation

Scikit-learn — Similarity computations, ML models

Plotly & Seaborn — Data visualization

Google Maps API — Restaurant location links

Future Enhancements

User login system and profiles

Recommendation history for multiple sessions

Sentiment analysis on restaurant reviews

Dockerization and cloud deployment

Enhanced ML-based ranking using more features

Screenshots

Home / Banner

Filter Search

Similar Restaurants

Favorites

Personalized Recommendations

Insights & Top 10 Restaurants

(You can add actual screenshots here)

Author

Sholingan S
Email: sholingan@gmail.com

LinkedIn: linkedin.com/in/sholingans