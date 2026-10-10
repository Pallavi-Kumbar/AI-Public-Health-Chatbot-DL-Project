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

## 🧠 System Architecture

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

```

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
```

## 🔁 Data Augmentation

To improve the ability of the model to understand different ways of describing symptoms, additional paraphrased symptom sentences were added to the original dataset.

For example, for **Jaundice**:

```text
My skin has become yellowish and I feel very tired.

My eyes look yellow and I feel weak.

I noticed yellowing of my skin and eyes.

My skin is turning yellow and I have been feeling fatigued.
```

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

## ⚙️ Important Model Parameters

| Parameter | Value |
|---|---:|
| Vocabulary Size | 10,000 |
| Sequence Length | 60 |
| Embedding Dimension | 128 |
| LSTM Units | 96 |
| Dense Units | 64 |
| Dropout | 0.25 / 0.20 |
| Optimizer | Adam |
| Learning Rate | 0.005 |
| Batch Size | 32 |
| Maximum Epochs | 40 |
| Early Stopping Patience | 5 |
| Output Activation | Softmax |


## Training Configuration
The model uses:
- Adam optimizer for optimization
- Sparse categorical cross-entropy as the loss function
- Softmax activation for multi-class classification
- Early stopping to prevent unnecessary training
- Model checkpointing to preserve the best-performing model
Early stopping monitors validation performance and stops training when the validation loss does not improve for a specified number of epochs.

## 🔍 TF-IDF Semantic Similarity
In addition to the Bi-LSTM prediction, the application uses TF-IDF-based semantic similarity.
TF-IDF converts text into numerical vectors based on the importance of words in the dataset.
The application then uses cosine similarity to compare the user's symptom description with symptom descriptions from the training data.
The purpose of this component is to determine whether the user's input is sufficiently similar to the type of symptom descriptions the model has seen during training.
## 🎯 Prediction Filtering
The system combines two signals:
```text
Bi-LSTM Prediction Confidence
              +
TF-IDF Semantic Similarity
              ↓
       Final Decision
```
The current thresholds are:
```test
Confidence Threshold = 0.70

Similarity Threshold = 0.20
```
A prediction is accepted only when both conditions are satisfied:
```text
Confidence ≥ 0.70
        AND
Similarity ≥ 0.20
```
If the input does not satisfy these conditions, the application avoids presenting the prediction as a reliable health-condition result.
This provides an additional layer of uncertainty handling for unfamiliar or poorly represented inputs.

## 📈 Model Evaluation
The project evaluates the classification model using standard classification metrics:
- Accuracy
- Precision
- Recall
- F1-score 
- Confusion Matrix

### 📊 Baseline Model
A TF-IDF + Logistic Regression model was initially developed as a baseline for comparison.
The baseline achieved approximately:
```text
Accuracy: 88.1%
```
This baseline was useful for comparing traditional machine-learning classification with the Deep Learning approach.

### 🧠 Bi-LSTM Model
The original Bi-LSTM model achieved approximately:
```text
Test Accuracy: 80.2%
```


## 💻 Streamlit Application
The trained model is integrated into an interactive Streamlit web application.
The application allows users to:
- Enter a natural-language description of their symptoms.
- Submit the symptoms for analysis.
- Receive a predicted health condition when the model is sufficiently confident.
- View a general condition overview.
- View common symptoms and health-awareness information.
- View information about when professional medical attention may be appropriate.
- View technical prediction information such as confidence and similarity scores.
The Bi-LSTM was retained as the primary deployed model because the project is focused on Deep Learning-based natural-language health classification.
The model was subsequently improved using additional paraphrased training examples and fine-tuning.
Note: The final performance metrics should be updated here after completing the final augmented-model training and evaluation.


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
