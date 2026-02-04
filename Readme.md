🍏 Food Classification Using Nutritional Data
Overview

This project focuses on classifying food items based on their nutritional content using machine learning. The model can predict the category of a food item (e.g., Fast Food, Dessert, Healthy, etc.) given its nutritional information such as calories, protein, fat, carbohydrates, sugar, etc.

It is a great demonstration of data preprocessing, feature engineering, and classification algorithms applied to real-world nutritional datasets.

Features

Food Classification: Predicts food category using nutritional values.

Data Preprocessing: Handles missing values, normalization, and encoding.

Exploratory Data Analysis (EDA): Insights into nutritional distributions and food types.

Machine Learning Models: Supports multiple classifiers such as:

Random Forest

XGBoost

Logistic Regression

K-Nearest Neighbors (KNN)

Evaluation Metrics: Accuracy, precision, recall, F1-score, and confusion matrix.

Visualization: Nutritional distribution and category analysis using Seaborn and Matplotlib.

Dataset

Dataset contains nutritional information of food items:

Calories

Protein

Fat

Carbohydrates

Sugar

Fiber

Food Category (Target)

Source: Public datasets from Kaggle / UCI / self-curated nutritional dataset.

CSV file: food_nutrition.csv

Installation

Clone the repository:

git clone https://github.com/yourusername/Food-Classification-Using-Nutritional-Data.git
cd Food-Classification-Using-Nutritional-Data


Create and activate a Python virtual environment:

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate


Install required packages:

pip install -r requirements.txt

How to Run
Jupyter Notebook
jupyter notebook


Open Food_Classification.ipynb and run the notebook to explore data, train models, and evaluate performance.

Python Script
python train_model.py


Trains the selected model and outputs classification metrics.

Saves trained models in models/ folder for future prediction.

File Structure
Food-Classification-Using-Nutritional-Data/
│
├─ data/
│   └─ food_nutrition.csv      # Nutritional dataset
├─ models/
│   └─ trained_model.pkl       # Saved trained ML model
├─ notebooks/
│   └─ Food_Classification.ipynb
├─ train_model.py              # Script to train ML model
├─ requirements.txt            # Python dependencies
└─ README.md                   # Project documentation

Technologies Used

Python 3.10+

Pandas / NumPy — Data manipulation

Scikit-learn — ML models and evaluation

XGBoost — Boosted tree model for classification

Seaborn / Matplotlib — Data visualization

Joblib / Pickle — Model serialization

Future Enhancements

Deploy a web app with Streamlit for live food classification.

Add deep learning models (e.g., feed-forward neural networks) for higher accuracy.

Integrate image-based food classification using computer vision.

Add interactive nutritional recommendations based on user preferences.

Author

Sholingan S
Email: sholingan@gmail.com

LinkedIn: linkedin.com/in/sholingans
License

This project is MIT Licensed — free for personal, educational, and portfolio use.