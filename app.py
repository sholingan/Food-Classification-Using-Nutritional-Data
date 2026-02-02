import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer

# ---------------- PAGE CONFIG ----------------
# ---------------- TITLE RIBBON ----------------
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #FBBF24, #F97316);
        color: white;
        font-size: 46px;
        font-weight: 700;
        padding: 15px 20px;
        border-radius: 12px;
        display: inline-block;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 12px;
    ">
        🥗 Smart Meal-based Food Recommendation & Food Classification
    </div>
    """,
    unsafe_allow_html=True
)

st.set_page_config(
    page_title="🥗 Smart Meal-based Food Recommendation & Food Classification",
    layout="wide"
)



# Ribbon-style caption
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #6EE7B7, #3B82F6);
        color: white;
        font-size: 25px;
        font-weight: 600;
        padding: 12px 18px;
        border-radius: 12px;
        display: inline-block;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    ">
        Healthy-aware • Dataset-driven • Meal • KNN • Classification • Charts • 
        Bulk Nutritional Analysis • ML-powered • Nutrition Insights
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    if os.path.exists("food_dataset.csv"):
        df = pd.read_csv("food_dataset.csv")
    else:
        st.warning("⚠️ food_dataset.csv not found. Please upload it.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
        else:
            st.stop()
    df = df.drop_duplicates().dropna(how='all')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ---------------- NUTRITION COLUMNS ----------------
POSSIBLE_NUTRITION_COLS = [
    "Calories", "Protein", "Fat", "Carbs",
    "Sugar", "Fiber", "Sodium",
    "Cholesterol", "Glycemic Index",
    "Water Content", "Serving Size"
]
nutrition_cols = [c for c in POSSIBLE_NUTRITION_COLS if c in df.columns]

if len(nutrition_cols) < 3:
    st.error("❌ Not enough nutrition columns in dataset")
    st.stop()

# ---------------- AUTO COLUMN DETECTION ----------------
def find_col(possible):
    for c in df.columns:
        if c.lower() in possible:
            return c
    return None

COLS = {
    "Calories": find_col(["calories"]),
    "Protein": find_col(["protein"]),
    "Fat": find_col(["fat"]),
    "Carbs": find_col(["carbs", "carbohydrates"]),
    "Sugar": find_col(["sugar"]),
    "Fiber": find_col(["fiber"]),
    "Sodium": find_col(["sodium"]),
    "Cholesterol": find_col(["cholesterol"]),
}

MEAL_COL = find_col(["meal_type", "meal"])
COOK_COL = find_col(["cooking_method", "method"])
FOOD_COL = find_col(["food_name", "name"])

NUM_COLS = [v for v in COLS.values() if v is not None]

# ---------------- HANDLE NaN ----------------
df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
imputer = SimpleImputer(strategy="median")
df[NUM_COLS] = imputer.fit_transform(df[NUM_COLS])

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🍽️ Smart Filters")

# Meal Type
if MEAL_COL:
    meals = sorted(df[MEAL_COL].dropna().unique())
    selected_meals = st.sidebar.multiselect("Meal Type", meals, default=meals)
    if selected_meals:
        df = df[df[MEAL_COL].isin(selected_meals)]

# Cooking Method
COOK_OPTIONS = ["raw", "baked", "fried", "grilled"]
if COOK_COL:
    cooks = sorted(df[COOK_COL].dropna().unique())
    all_cook_options = sorted(list(set(cooks + COOK_OPTIONS)))
    selected_cook = st.sidebar.multiselect("Cooking Method", all_cook_options)
    if selected_cook:
        df = df[df[COOK_COL].isin(selected_cook)]

# ----------------- Dietary flags -----------------
st.sidebar.subheader("🥦 Dietary Preferences")

# Map display label -> actual dataset column
DIET_FLAGS = {
    "🥖 Gluten Free": ["gluten_free", "Gluten_Free", "contains_gluten"],
    "🌱 Vegan": ["vegan", "Vegan"],
    "🥬 Vegetarian": ["vegetarian", "Vegetarian"]
}

for label, possible_cols in DIET_FLAGS.items():
    # Find the first matching column in df
    col = next((c for c in possible_cols if c in df.columns), None)
    if col:  # Only show checkbox if column exists
        if st.sidebar.checkbox(label, value=False):
            df = df[df[col] == True]

# Health-based filters
st.sidebar.markdown("---")
st.sidebar.subheader("🩺 Health-Based Filters")
if "Calories" in df.columns:
    if st.sidebar.checkbox("🔥 Low Calorie (< 300 kcal)"):
        df = df[df["Calories"] < 300]
if "Protein" in df.columns:
    if st.sidebar.checkbox("💪 High Protein (> 15g)"):
        df = df[df["Protein"] > 15]
if "Sugar" in df.columns:
    if st.sidebar.checkbox("🩸 Low Sugar (< 10g)"):
        df = df[df["Sugar"] < 10]
if "Fat" in df.columns:
    if st.sidebar.checkbox("❤️ Low Fat (< 10g)"):
        df = df[df["Fat"] < 10]
if "Fiber" in df.columns:
    if st.sidebar.checkbox("🌾 High Fiber (> 5g)"):
        df = df[df["Fiber"] > 5]
if "Sodium" in df.columns:
    if st.sidebar.checkbox("🧂 Low Sodium (< 300mg)"):
        df = df[df["Sodium"] < 300]

# ---------------- DATASET PREVIEW ----------------
st.subheader("📊 Dataset Preview (15 Sample)")
st.dataframe(df.head(15), use_container_width=True)

# Download full dataset
csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download Full Dataset",
    data=csv,
    file_name="full_food_dataset.csv",
    mime="text/csv"
)


# ---------------- USER INPUT ----------------
st.subheader("🔢 Enter Your Preferences")
user_input = {}
cols = st.columns(4)
for i, (label, col_name) in enumerate(COLS.items()):
    if col_name:
        with cols[i % 4]:
            user_input[col_name] = st.number_input(
                label,
                min_value=0.0,
                value=float(df[col_name].median())
            )
input_df = pd.DataFrame([user_input])
input_df[NUM_COLS] = imputer.transform(input_df[NUM_COLS])

# Save preference
if "saved_pref" not in st.session_state:
    st.session_state.saved_pref = input_df
if st.button("💾 Save My Preference"):
    st.session_state.saved_pref = input_df
    st.success("Preferences saved!")

# ---------------- SCALING & KNN ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[NUM_COLS])
input_scaled = scaler.transform(input_df[NUM_COLS])

model = NearestNeighbors(n_neighbors=5, metric="euclidean")
model.fit(X_scaled)
distances, indices = model.kneighbors(input_scaled)
recommendations = df.iloc[indices[0]].copy()

# Safe Match %
max_dist = distances[0].max()
if max_dist == 0:
    recommendations["Match %"] = 100.0
else:
    recommendations["Match %"] = (1 - distances[0] / max_dist) * 100
recommendations["Match %"] = recommendations["Match %"].round(1)

# ---------------- HEALTH SCORE ----------------
def health_score(row):
    score = 100
    if COLS["Calories"] and row[COLS["Calories"]] > 400: score -= 20
    if COLS["Sugar"] and row[COLS["Sugar"]] > 15: score -= 20
    if COLS["Fat"] and row[COLS["Fat"]] > 20: score -= 15
    if COLS["Fiber"] and row[COLS["Fiber"]] < 3: score -= 10
    return max(score, 0)
recommendations["Health Score"] = recommendations.apply(health_score, axis=1)

# ---------------- MEAL-WISE RESULTS ----------------
st.subheader("🍱 Meal-wise Food Recommendations")

food_col = "Food_Name" if "Food_Name" in recommendations.columns else None
prep_col = "Preparation_Method" if "Preparation_Method" in recommendations.columns else None

if MEAL_COL:
    # Get all meal types from the original filtered recommendations
    all_meals = df[MEAL_COL].unique()

    for meal in sorted(all_meals):
        st.markdown(f"### 🍽 {meal.title()}")

        meal_df = recommendations[recommendations[MEAL_COL] == meal]

        if meal_df.empty:
            st.info("No foods match your filters for this meal")
            continue

        meal_df = meal_df.sort_values("Match %", ascending=False)

        for _, row in meal_df.iterrows():
            food_name = row.get(food_col, "Unknown")
            prep_method = row.get(prep_col, "Unknown")
            health_score = row.get("Health Score", 0)
            match_pct = row.get("Match %", 0.0)

            display_text = f"✅ {food_name} ({prep_method}) | Match {match_pct:.1f}% | Health {health_score}"

            if health_score < 50:
                st.error(display_text)
            else:
                st.success(display_text)



# ---------------- FOOD CLASSIFICATION ----------------
st.subheader("📌 Food Classification")
def classify_food(row):
    if COLS["Calories"] and row[COLS["Calories"]] < 250: return "Low Calorie"
    if COLS["Protein"] and row[COLS["Protein"]] > 15: return "High Protein"
    if COLS["Sugar"] and row[COLS["Sugar"]] > 20: return "High Sugar"
    return "Balanced"
df["Food Class"] = df.apply(classify_food, axis=1)
if FOOD_COL:
    st.dataframe(df[[FOOD_COL,"Food Class"]].head(10))
else:
    st.dataframe(df[["Food Class"]].head(10))

# ---------------- NUTRITION DISTRIBUTION (GRID VIEW) ----------------
st.subheader("📈 Nutritional Distribution (Bulk View)")

selected_nutrients = st.multiselect(
    "Select one or more nutrients",
    nutrition_cols,
    default=nutrition_cols[:8]
)

if selected_nutrients:
    # Define number of columns per row
    num_cols_per_row = 4  # You can increase to 3 or 4 for wider screens
    for i in range(0, len(selected_nutrients), num_cols_per_row):
        cols = st.columns(num_cols_per_row)
        for j, nutrient in enumerate(selected_nutrients[i:i+num_cols_per_row]):
            with cols[j]:
                st.markdown(f"**{nutrient} Distribution**")
                fig, ax = plt.subplots()
                ax.hist(df[nutrient], bins=30, color="#4CAF50", edgecolor="black")
                ax.set_xlabel(nutrient)
                ax.set_ylabel("Count")
                st.pyplot(fig)
else:
    st.info("Please select at least one nutrient to view charts")

# ---------------- MEAL TYPE DISTRIBUTION ----------------
if MEAL_COL:
    st.subheader("🍽 Meal Type Distribution")
    meal_count = df[MEAL_COL].value_counts().reset_index()
    meal_count.columns = ["Meal", "Count"]
    fig = px.bar(meal_count, x="Meal", y="Count", title="Foods per Meal Type")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- AVERAGE CALORIES BY MEAL ----------------
if "Calories" in df.columns and MEAL_COL:
    st.subheader("⚡ Average Calories by Meal Type")
    avg_cal = df.groupby(MEAL_COL, as_index=False)["Calories"].mean()
    fig = px.bar(avg_cal, x=MEAL_COL, y="Calories", color=MEAL_COL, title="Average Calories by Meal")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- CALORIES vs PROTEIN ----------------
if {"Calories", "Protein"}.issubset(df.columns):
    st.subheader("🥩 Calories vs Protein by Meal Type")

    if MEAL_COL:
        fig = px.scatter(
            df,
            x="Protein",
            y="Calories",
            color=MEAL_COL,
            hover_data=["Food_Name"] if "Food_Name" in df.columns else None,
            size_max=15,
            opacity=0.7,
            title="Calories vs Protein by Meal Type",
            labels={"Protein": "Protein (g)", "Calories": "Calories (kcal)"}
        )
    else:
        fig = px.scatter(
            df,
            x="Protein",
            y="Calories",
            hover_data=["Food_Name"] if "Food_Name" in df.columns else None,
            opacity=0.7,
            title="Calories vs Protein"
        )

    st.plotly_chart(fig, use_container_width=True)


# ---------------- TOP 10 FOODS ----------------
if FOOD_COL:
    st.subheader("🍔 Top 10 Most Frequent Foods")
    top_foods = df[FOOD_COL].value_counts().head(10).reset_index()
    top_foods.columns = [FOOD_COL,"Count"]
    fig = px.bar(top_foods, x=FOOD_COL, y="Count", title="Top 10 Foods in Dataset")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- FOOTER -----------------------------
st.markdown(
    """
    <div style="
        border: 2px solid #1f2937; 
        border-radius: 10px; 
        padding: 15px; 
        text-align: center; 
        background-color: #f9fafb;
    ">
        ✅ <strong>Meal + KNN + Health + Classification + Charts + Filters + Bulk Nutritional Analysis</strong><br><br>
        Created by <strong>Sholingan S</strong><br>
        <a href='https://www.linkedin.com/in/sholingans/' target='_blank' style="text-decoration:none; color:#0A66C2; font-weight:bold;">
            <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' 
                 width='24' height='24' style='vertical-align:middle; margin-right:5px;'/>
            Follow me on LinkedIn
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

