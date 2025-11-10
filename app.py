import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from visualize_results import plot_radar, visualize_token_importance
from model_utils import get_probability_dist

MODEL_PATH_DIR = "my_awesome_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Moral Foundations Classification Visualization")


@st.cache_resource
def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    id2label = model.config.id2label
    return tokenizer, model, id2label


tokenizer, model, id2label = load_model(MODEL_PATH_DIR)

user_input = st.text_area("Enter text to classify:", "Type your text here...")

if st.button("Classify and Visualize"):
    with st.spinner("Classifying and visualizing..."):
        probs = get_probability_dist(user_input, model, tokenizer, device)
        labels = [id2label[i] for i in range(len(probs))]

        st.subheader("Probability Distribution")
        ax = plot_radar(probs, labels)
        st.pyplot(ax.figure)

        st.subheader("Token Importance Visualization")
        html_viz = visualize_token_importance(
            text=user_input,
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
            max_prob_thresh=0.1,
            max_classes=5,
        )
        st.components.v1.html(html_viz.data, height=800)
