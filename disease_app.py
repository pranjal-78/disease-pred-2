# disease_app.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="AI Disease Chatbot",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI-driven Disease Prediction Chatbot")
st.markdown("""
Welcome! Enter your symptoms or select them from the checklist.
The system predicts possible diseases including **COVID-19**, **Malaria**, **Dengue**, **Flu**, and **Common Cold**.
⚠️ This is for awareness only — not a medical diagnosis.
""")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("balanced_disease_dataset.csv")
    return df

df = load_data()
X = df.drop("Disease", axis=1)
y = df["Disease"]
symptoms_list = list(X.columns)

# ----------------------------
# Train Model
# ----------------------------
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

model = train_model(X, y)

# ----------------------------
# Symptom Extraction Function
# ----------------------------
def extract_symptoms(text: str):
    """Extracts known symptoms from user text."""
    text_l = text.lower()
    found = {s: 0 for s in symptoms_list}

    # direct match
    for s in symptoms_list:
        if s.replace("_", " ") in text_l:
            found[s] = 1

    # synonym dictionary
    synonyms = {
        "fever": ["temperature", "hot body"],
        "dry_cough": ["coughing", "persistent cough"],
        "fatigue": ["tired", "exhausted"],
        "loss_of_taste": ["taste loss", "can't taste", "no taste"],
        "loss_of_smell": ["smell loss", "can't smell", "no smell"],
        "shortness_of_breath": ["breathless", "difficulty breathing"],
        "sore_throat": ["throat pain", "throat hurts"],
        "headache": ["head pain", "migraine"],
        "muscle_pain": ["body ache", "muscle ache"],
        "joint_pain": ["arthralgia", "joint ache"],
        "chills": ["shivering"],
        "sweating": ["perspiration"],
        "rash": ["skin spots", "skin rash"],
        "diarrhea": ["loose stool", "nausea"],
        "runny_nose": ["nasal congestion", "stuffy nose"]
    }

    # synonym detection
    for symptom, words in synonyms.items():
        if any(w in text_l for w in words):
            found[symptom] = 1

    return found

# ----------------------------
# Prediction Function
# ----------------------------
def predict_disease(input_features):
    """Predicts top 3 possible diseases with probabilities."""
    features = np.array(list(input_features.values())).reshape(1, -1)
    probs = model.predict_proba(features)[0]
    classes = model.classes_

    top_indices = np.argsort(probs)[::-1][:3]
    return [(classes[i], probs[i]) for i in top_indices]

# ----------------------------
# Sidebar: Symptom Checklist
# ----------------------------
st.sidebar.header("🩹 Select Symptoms (Optional)")
selected_symptoms_sidebar = st.sidebar.multiselect(
    "Pick symptoms from the list:",
    options=[s.replace("_", " ").title() for s in symptoms_list]
)

# Map sidebar selections to binary input
sidebar_input = {s: 0 for s in symptoms_list}
for s in selected_symptoms_sidebar:
    key = s.lower().replace(" ", "_")
    sidebar_input[key] = 1

# ----------------------------
# Main Input Section
# ----------------------------
st.subheader("✍️ Enter Symptoms")
user_text_input = st.text_input(
    "Describe your symptoms (e.g., 'I have fever, dry cough, and lost my sense of smell'):"
)

# Combine sidebar + text input
combined_input = sidebar_input.copy()
if user_text_input:
    text_features = extract_symptoms(user_text_input)
    for k in combined_input:
        combined_input[k] = max(combined_input[k], text_features[k])

# ----------------------------
# Prediction Section
# ----------------------------
if st.button("🔬 Predict Disease"):
    if sum(combined_input.values()) == 0:
        st.warning("⚠️ Please provide at least one symptom via checklist or text input.")
    else:
        top_predictions = predict_disease(combined_input)

        st.subheader("🎯 Top Predictions")
        for i, (disease, prob) in enumerate(top_predictions, start=1):
            confidence = (
                "🔴 High" if prob > 0.7 else
                "🟡 Medium" if prob > 0.4 else
                "🟢 Low"
            )

            st.markdown(f"**{i}. {disease}** — Probability: {prob:.2%} — Confidence: {confidence}")
            st.progress(float(prob))

# ----------------------------
# Detected Symptoms
# ----------------------------
st.subheader("🧾 Detected Symptoms")
detected = [s.replace("_", " ").title() for s, v in combined_input.items() if v == 1]
st.write(detected if detected else "No symptoms detected yet.")

# ----------------------------
# Dataset Insights
# ----------------------------
with st.expander("🔍 Dataset Insights"):
    st.write(f"- Total Records: {len(df)}")
    st.write(f"- Number of Diseases: {df['Disease'].nunique()}")
    st.write(f"- Number of Symptoms: {len(symptoms_list)}")
    st.bar_chart(df['Disease'].value_counts())
