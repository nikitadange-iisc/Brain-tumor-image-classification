import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
EXPLAINABILITY_ROOT = PROJECT_ROOT / "artifacts" / "explainability"
CLASSES = ["glioma", "healthy", "meningioma", "pituitary"]
IMAGE_SIZE = (224, 224)


def load_metrics(model_name: str) -> dict | None:
    metrics_path = RESULTS_ROOT / model_name / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text())


def metrics_to_frame(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(metrics["per_class"]).T[["precision", "recall", "f1_score", "specificity", "roc_auc_ovr"]]


@st.cache_resource
def load_saved_model(model_name: str):
    model_path = PROJECT_ROOT / "artifacts" / "models" / f"{model_name}.keras"
    if not model_path.exists():
        return None
    return tf.keras.models.load_model(model_path)


def prepare_uploaded_image(uploaded_file) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(uploaded_file).convert("RGB")
    display_image = image.copy()
    resized_image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)
    return display_image, image_batch


st.set_page_config(page_title="Brain Tumor MRI Dashboard", layout="wide")
st.title("Brain Tumor MRI Classification Dashboard")

st.write(
    "This dashboard shows saved model outputs from the notebooks: overall metrics, per-class metrics, "
    "confusion matrices, ROC curves, and Grad-CAM examples."
)

model_name = st.selectbox("Select model", ["custom_cnn", "efficientnet_b0"])
metrics = load_metrics(model_name)

if metrics is None:
    st.warning(f"No saved metrics found for {model_name}. Run the modeling notebook first.")
else:
    st.subheader("Overall Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Macro F1", f"{metrics['macro_avg']['f1_score']:.4f}")
    col3.metric("Macro ROC-AUC", f"{metrics['macro_avg']['roc_auc_ovr']:.4f}")

    st.subheader("Per-Class Metrics")
    st.dataframe(metrics_to_frame(metrics))

    conf_matrix_path = RESULTS_ROOT / model_name / "confusion_matrix.png"
    roc_curve_path = RESULTS_ROOT / model_name / "roc_curves.png"

    image_col1, image_col2 = st.columns(2)
    if conf_matrix_path.exists():
        image_col1.image(Image.open(conf_matrix_path), caption="Confusion Matrix", use_container_width=True)
    if roc_curve_path.exists():
        image_col2.image(Image.open(roc_curve_path), caption="ROC Curves", use_container_width=True)

gradcam_path = EXPLAINABILITY_ROOT / "gradcam_examples.png"
st.subheader("Explainability")
if gradcam_path.exists():
    st.image(Image.open(gradcam_path), caption="Grad-CAM Examples", use_container_width=True)
else:
    st.info("No Grad-CAM image found yet. Run the explainability notebook first.")

st.subheader("Single-Image Prediction")
st.write(
    "Upload a brain MRI image and the selected saved model will predict the most likely class. "
    "Use MRI images similar to the training data for meaningful results."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    prediction_model = load_saved_model(model_name)
    if prediction_model is None:
        st.warning(f"Saved model file not found for {model_name}. Run the modeling notebook first.")
    else:
        display_image, image_batch = prepare_uploaded_image(uploaded_file)
        probabilities = prediction_model.predict(image_batch, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASSES[predicted_index]
        confidence = float(probabilities[predicted_index])

        preview_col, result_col = st.columns([1, 1])
        preview_col.image(display_image, caption="Uploaded MRI", use_container_width=True)
        result_col.metric("Predicted Class", predicted_class.capitalize())
        result_col.metric("Confidence", f"{confidence:.2%}")

        probability_frame = pd.DataFrame(
            {
                "class": [class_name.capitalize() for class_name in CLASSES],
                "probability": probabilities,
            }
        ).sort_values("probability", ascending=False)

        st.dataframe(probability_frame.style.format({"probability": "{:.2%}"}), use_container_width=True)
        st.bar_chart(probability_frame.set_index("class"))
