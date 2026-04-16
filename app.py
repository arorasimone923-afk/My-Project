from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

app = Flask(__name__)

# 🧠 Training Data
texts = [
    "frequent urination increased thirst increased hunger unexplained weight loss fatigue blurred vision slow healing wounds dry skin tingling in hands and feet",
    "persistent cough coughing blood chest pain fatigue weight loss night sweats fever loss of appetite shortness of breath",
    "breast lump change in breast size nipple discharge skin dimpling pain in breast",
    "severe headache throbbing pain sensitivity to light sensitivity to sound nausea vomiting dizziness aura"
]

labels = [
    "Diabetes 🩸",
    "Tuberculosis 🫁",
    "Breast Cancer 👩",
    "Migraine 🤯"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

# Symptoms List
all_symptoms = [
    "frequent urination", "increased thirst", "increased hunger",
    "unexplained weight loss", "fatigue", "blurred vision",
    "slow healing wounds", "dry skin", "tingling in hands and feet",
    "persistent cough", "coughing blood", "chest pain", "night sweats",
    "fever", "loss of appetite", "shortness of breath",
    "breast lump", "change in breast size", "nipple discharge",
    "skin dimpling", "pain in breast", "severe headache",
    "throbbing pain", "sensitivity to light", "sensitivity to sound",
    "nausea", "vomiting", "dizziness", "aura"
]

# Health Tips
health_tips = {
    "Diabetes 🩸": "Maintain a healthy diet, exercise regularly, and monitor blood sugar levels.",
    "Tuberculosis 🫁": "Seek medical attention and complete the full course of prescribed medication.",
    "Breast Cancer 👩": "Regular screenings and early detection significantly improve outcomes.",
    "Migraine 🤯": "Manage stress, maintain sleep hygiene, and avoid known triggers."
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/details', methods=['GET', 'POST'])
def details():
    if request.method == 'POST':
        name = request.form['name']
        return render_template('next.html', name=name)
    return render_template('details.html')

@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms():
    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        user_input = " ".join(selected_symptoms).lower()
        X_test = vectorizer.transform([user_input])
        probs = model.predict_proba(X_test)[0]
        classes = model.classes_
        top_idx = np.argsort(probs)[::-1][:3]
        top_predictions = [(classes[i], probs[i] * 100) for i in top_idx]
        top_disease = top_predictions[0][0]
        tip = health_tips.get(top_disease, "")
        return render_template(
            'result.html',
            predictions=top_predictions,
            tip=tip
        )
    return render_template('symptoms.html', all_symptoms=all_symptoms)

@app.route('/about')
def about():
    return render_template('about.html')

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
