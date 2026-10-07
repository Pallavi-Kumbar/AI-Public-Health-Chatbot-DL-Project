# 🩺 AI-Driven Public Health Chatbot for Disease Awareness

An AI-powered public health chatbot that uses **Deep Learning and Natural Language Processing (NLP)** to analyze user-provided symptom descriptions and provide general health-awareness information.

The system uses a **Bidirectional Long Short-Term Memory (Bi-LSTM)** neural network to classify symptom descriptions into one of **24 predefined health conditions**. A **TF-IDF-based semantic similarity mechanism** is additionally used to evaluate how closely the user's input matches symptom descriptions from the training data.

The trained model is integrated into an interactive **Streamlit web application**.

> ⚠️ **Disclaimer:** This application is developed for educational and general health-awareness purposes only. It does not provide medical diagnosis, treatment, or professional medical advice. Users should consult a qualified healthcare professional for medical concerns.

---

## 📌 Project Overview

People often describe the same symptoms using different words and sentence structures. A system that relies only on exact symptom matching may fail when users phrase their symptoms differently.

This project addresses this problem by developing a **Deep Learning-based health-awareness chatbot** capable of processing natural-language symptom descriptions and classifying them into relevant health conditions.

The system combines:

- Natural Language Processing
- Text preprocessing
- Data augmentation
- Text vectorization
- Word embeddings
- Bidirectional LSTM
- Softmax classification
- TF-IDF semantic similarity
- Cosine similarity
- Confidence-based filtering
- Streamlit deployment

---

## 🎯 Objectives

The main objectives of this project are:

1. Develop a Deep Learning model for symptom-based health classification.
2. Process natural-language descriptions of symptoms.
3. Improve model robustness using paraphrased training examples.
4. Classify user inputs into predefined health conditions.
5. Use prediction confidence and semantic similarity to filter uncertain predictions.
6. Provide general health-awareness information about the predicted condition.
7. Develop an interactive and user-friendly web application using Streamlit.
8. Demonstrate the application of Deep Learning and NLP in the public-health awareness domain.

---

# 🧠 System Architecture

The overall architecture of the project is:

```text
                    ┌──────────────────────┐
                    │   Symptom Dataset    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Preprocessing   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Augmentation    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Train / Validation   │
                    │ / Test Split         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Text Vectorization   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Embedding Layer      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Bidirectional LSTM   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Dense Layers         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Softmax Prediction   │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │                      │
                    ▼                      ▼
          ┌─────────────────┐    ┌────────────────────┐
          │ Prediction      │    │ TF-IDF Semantic    │
          │ Confidence      │    │ Similarity         │
          └────────┬────────┘    └─────────┬──────────┘
                   │                       │
                   └───────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Confidence +         │
                    │ Similarity Filtering │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Health Information   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Web App    │
                    └──────────────────────┘

---

## 📊 Dataset

The project uses the **Symptom2Disease** dataset.

### Dataset Characteristics

- **Original samples:** 1,200
- **Number of health-condition classes:** 24
- **Input:** Natural-language symptom descriptions
- **Target:** Health-condition label

The dataset contains symptom descriptions associated with different health conditions.

### 🏥 Health Conditions

The model supports the following 24 classes:

1. Acne
2. Arthritis
3. Bronchial Asthma
4. Cervical spondylosis
5. Chicken pox
6. Common Cold
7. Dengue
8. Dimorphic Hemorrhoids
9. Fungal infection
10. Hypertension
11. Impetigo
12. Jaundice
13. Malaria
14. Migraine
15. Pneumonia
16. Psoriasis
17. Typhoid
18. Varicose Veins
19. Allergy
20. Diabetes
21. Drug Reaction
22. Gastroesophageal Reflux Disease
23. Peptic Ulcer Disease
24. Urinary Tract Infection

---

## 🔄 Data Preprocessing

The input symptom descriptions undergo preprocessing before being provided to the model.

The preprocessing pipeline includes:

- Converting text to lowercase
- Removing HTML tags
- Removing unnecessary special characters
- Removing extra whitespace
- Preparing clean textual input for model processing

### Example

**Original:**

```text
My skin is Yellowish!!!

## 🔁 Data Augmentation

To improve the ability of the model to understand different ways of describing symptoms, additional paraphrased symptom sentences were added to the original dataset.

For example, for **Jaundice**:

```text
My skin has become yellowish and I feel very tired.

My eyes look yellow and I feel weak.

I noticed yellowing of my skin and eyes.

My skin is turning yellow and I have been feeling fatigued.

# 🤖 Deep Learning Model

The primary classification model used in the project is a **Bidirectional Long Short-Term Memory (Bi-LSTM)** neural network.

Bi-LSTM is suitable for text classification because it processes the input sequence in both forward and backward directions. This allows the model to capture contextual information from different parts of a symptom description.

## 🏗️ Model Architecture

```text
Input Text
     ↓
TextVectorization
     ↓
Embedding Layer
     ↓
Bidirectional LSTM
     ↓
Dropout
     ↓
Dense Layer
     ↓
Dropout
     ↓
Softmax Output

```

### 🖥️ Application Workflow
The Streamlit application follows this workflow:
```text
User enters symptoms
        ↓
Text preprocessing
        ↓
Bi-LSTM prediction
        ↓
Prediction confidence calculated
        ↓
TF-IDF semantic similarity calculated
        ↓
Confidence + similarity checked
        ↓
      ┌───────────────┐
      │               │
      ▼               ▼
   Accepted        Rejected
      │               │
      ▼               ▼
Condition         Uncertain/
Information       unsupported input
      │
      ▼
Health-awareness
response

```

## 📁 Project Structure
```text
AI-Public-Health-Chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── health_model.keras
│   ├── label_encoder.pkl
│   ├── semantic_vectorizer.pkl
│   └── X_train_semantic.pkl
│
├── data/
│   ├── __init__.py
│   └── health_information.py
│
└── training/
    └── AI_Public_Health_Chatbot.ipynb

```

# ⚙️ Installation and Setup

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Public-Health-Chatbot.git
```

Navigate to the project directory:

```bash
cd AI-Public-Health-Chatbot
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
.venv\Scripts\activate
```

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the browser.

---

# 🧑‍💻 Development Workflow

The project was developed in two major stages.

## Stage 1 — Model Development

The Deep Learning model was developed and trained using **Google Colab**.

The training workflow was:

```text
Dataset
   ↓
Data Exploration
   ↓
Text Preprocessing
   ↓
Data Augmentation
   ↓
Train / Validation / Test Split
   ↓
Text Vectorization
   ↓
Bi-LSTM Training
   ↓
Model Evaluation
   ↓
Model and Supporting Artifacts Saved
```

## Stage 2 — Application Development

The trained model was integrated into a **Streamlit application** using **Visual Studio Code**.

The application workflow was:

```text
Saved Bi-LSTM Model
        ↓
Model Loading
        ↓
User Symptom Input
        ↓
Prediction
        ↓
Confidence Calculation
        ↓
Semantic Similarity
        ↓
Prediction Filtering
        ↓
Health Information
        ↓
Streamlit Interface
```

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Deep Learning

- TensorFlow
- Keras
- Bidirectional LSTM
- Word Embeddings

## Machine Learning / NLP

- Scikit-learn
- TF-IDF
- Cosine Similarity
- Logistic Regression

## Data Processing

- Pandas
- NumPy

## Application Development

- Streamlit

## Development Tools

- Google Colab
- Visual Studio Code
- Git
- GitHub