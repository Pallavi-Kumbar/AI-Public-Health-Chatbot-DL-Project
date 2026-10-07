import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import re

from sklearn.metrics.pairwise import cosine_similarity

from data.health_information import health_information


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Public Health Chatbot",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = "model/health_model.keras"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"
SEMANTIC_VECTORIZER_PATH = "model/semantic_vectorizer.pkl"
X_TRAIN_SEMANTIC_PATH = "model/X_train_semantic.pkl"


# ============================================================
# THRESHOLDS
# ============================================================

CONFIDENCE_THRESHOLD = 0.70
SIMILARITY_THRESHOLD = 0.20


# ============================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================

@st.cache_resource
def load_resources():

    model = tf.keras.models.load_model(MODEL_PATH)

    label_encoder = joblib.load(
        LABEL_ENCODER_PATH
    )

    semantic_vectorizer = joblib.load(
        SEMANTIC_VECTORIZER_PATH
    )

    X_train_semantic = joblib.load(
        X_TRAIN_SEMANTIC_PATH
    )

    return (
        model,
        label_encoder,
        semantic_vectorizer,
        X_train_semantic
    )


try:

    (
        model,
        label_encoder,
        semantic_vectorizer,
        X_train_semantic
    ) = load_resources()

except Exception as e:

    st.error(
        "Unable to load the trained model or supporting files."
    )

    st.exception(e)

    st.stop()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Keep only letters, numbers and spaces
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity_score(text):

    query_vector = semantic_vectorizer.transform(
        [clean_text(text)]
    )

    similarities = cosine_similarity(
        query_vector,
        X_train_semantic
    )

    return float(
        similarities.max()
    )


# ============================================================
# HEALTH TOPIC PREDICTION
# ============================================================

def predict_health_topic(user_text):

    cleaned = clean_text(user_text)

    # TensorFlow string tensor
    input_tensor = tf.constant(
        [cleaned],
        dtype=tf.string
    )

    # Model prediction
    prediction_tensor = model(
        input_tensor,
        training=False
    )

    probabilities = prediction_tensor.numpy()[0]

    # Highest probability class
    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_label = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(
        probabilities[predicted_index]
    )

    # Semantic similarity
    similarity = semantic_similarity_score(
        cleaned
    )

    # Acceptance logic
    accepted = (
        (confidence >= CONFIDENCE_THRESHOLD and similarity >= SIMILARITY_THRESHOLD) 
    )

    return {
        "label": predicted_label,
        "confidence": confidence,
        "similarity": similarity,
        "accepted": accepted
    }


# ============================================================
# GENERATE HEALTH AWARENESS RESPONSE
# ============================================================

def generate_awareness_response(result):

    # --------------------------------------------------------
    # LOW CONFIDENCE / LOW SIMILARITY
    # --------------------------------------------------------

    if not result["accepted"]:

        return (
            "I don't have enough confidence to provide "
            "specific health-awareness information from "
            "the symptoms provided.\n\n"

            "Please describe:\n"
            "- your main symptoms\n"
            "- how long you have had them\n"
            "- whether they are getting better or worse\n"
            "- any other symptoms occurring at the same time\n\n"

            "**Important:** This chatbot provides general "
            "health-awareness information and is not a "
            "medical diagnostic tool."
        )

    # --------------------------------------------------------
    # GET PREDICTED LABEL
    # --------------------------------------------------------

    label = result["label"]

    information = health_information.get(label)

    if information is None:

        return (
            "I could not find health-awareness information "
            "for the predicted topic."
        )

    # --------------------------------------------------------
    # BUILD RESPONSE
    # --------------------------------------------------------

    response = []

    response.append(
        f"### Possible Condition :  {label}"
    )

    response.append("")

    response.append(
        "### Condition Overview"
    )

    response.append(
        information["overview"]
    )

    response.append("")

    response.append(
        "### Commonly associated features"
    )

    for symptom in information["symptoms"]:

        response.append(
            f"- {symptom}"
        )

    response.append("")

    response.append(
        "### Health awareness"
    )

    for point in information["awareness"]:

        response.append(
            f"- {point}"
        )

    response.append("")

    response.append(
        "### When to seek medical help"
    )

    response.append(
        information["when_to_seek_help"]
    )

    response.append("")

    response.append(
        "**Important:** This information is for general "
        "health awareness only and does not establish a "
        "diagnosis. Symptoms can have multiple causes. "
        "For persistent, severe, worsening, or concerning "
        "symptoms, consult a qualified healthcare professional."
    )

    return "\n".join(response)


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("🩺AI-Driven Public Health Chatbot")

st.write(
    "   Describe your symptoms to receive general "
    "health-awareness information."
)

st.info(
    "This chatbot is designed for health awareness and "
    "education. It does not provide a medical diagnosis."
)


# ============================================================
# USER INPUT
# ============================================================

user_input = st.text_area(
    "Describe your symptoms:",
    placeholder=(
        "Example: I have a headache, nausea and sensitivity "
        "to light for the last two days."
    ),
    height=150
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Symptoms",
    use_container_width=True
):

    if not user_input.strip():

        st.warning(
            "Please enter some symptoms first."
        )

    else:

        with st.spinner(
            "Analyzing your symptoms..."
        ):

            result = predict_health_topic(
                user_input
            )

        # ----------------------------------------------------
        # ACCEPTED RESULT
        # ----------------------------------------------------

        if result["accepted"]:

            response = generate_awareness_response(
                result
            )

            st.markdown(response)

            # Internal model information
            with st.expander(
                "Model information"
            ):

                st.write(
                    f"Predicted topic: {result['label']}"
                )

                st.write(
                    f"Model confidence: "
                    f"{result['confidence']:.2%}"
                )

                st.write(
                    f"Semantic similarity: "
                    f"{result['similarity']:.2%}"
                )

        # ----------------------------------------------------
        # REJECTED RESULT
        # ----------------------------------------------------

        else:

            response = generate_awareness_response(
                result
            )

            st.warning(
                "The chatbot could not confidently "
                "identify a health-awareness topic."
            )

            st.markdown(response)

            with st.expander(
                "Why was this response returned?"
            ):
                st.write(
                    f"Predicted topic: {result['label']}"
                )

                st.write(
                    f"Model confidence: "
                    f"{result['confidence']:.2%}"
                )

                st.write(
                    f"Semantic similarity: "
                    f"{result['similarity']:.2%}"
                )

                st.write(
                    "The chatbot requires both confidence "
                    "and semantic similarity to meet the "
                    "configured thresholds."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Driven Public Health Chatbot | "
    "For educational and health-awareness purposes only."
)