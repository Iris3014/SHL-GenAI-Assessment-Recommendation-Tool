import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

# Streamlit config
st.set_page_config(page_title="SHL GenAI Assessment Recommender", layout="wide")

# Model and API setup
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# Cache model load
@st.cache_resource
def load_local_model():
    return SentenceTransformer(HF_MODEL_NAME)

# Cache CSV load
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "dataset", "shl_catalog.csv")
    return pd.read_csv(csv_path)

# Embedding functions
def get_hf_embeddings(texts, model):
    return model.encode(texts)

def get_openrouter_embeddings(texts, api_key):
    embedder = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_API_BASE,
    )
    return embedder.embed_documents(texts)

# Main App
def main():
    st.title("SHL GenAI Assessment Recommendation Tool")
    st.markdown("This tool recommends relevant SHL assessments based on your job description using Retrieval-Augmented Generation (RAG).")

    # Sidebar Settings
    st.sidebar.title("Settings")
    use_openai = st.sidebar.checkbox("Use OpenRouter Embeddings (Recommended)")
    openai_api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

    job_description = st.text_area("📄 Paste the Job Description here:")

    df = load_data()
    st.subheader("Available SHL Assessments")
    st.dataframe(df.drop(columns=["url"]))

    if st.button("Recommend Assessments"):
        if not job_description.strip():
            st.warning("Please enter a job description first.")
            return

        with st.spinner("Analyzing and generating recommendations..."):
            try:
                corpus = df["description"].tolist()

                if use_openai and openai_api_key:
                    query_embedding = get_openrouter_embeddings([job_description], openai_api_key)[0]
                    corpus_embeddings = get_openrouter_embeddings(corpus, openai_api_key)
                else:
                    model = load_local_model()
                    query_embedding = get_hf_embeddings([job_description], model)[0]
                    corpus_embeddings = get_hf_embeddings(corpus, model)

                similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
                df["similarity"] = similarities
                top_matches = df.sort_values("similarity", ascending=False).head(3)

                st.subheader("✅ Top Recommended Assessments")
                for _, row in top_matches.iterrows():
                    st.markdown(f"### [{row['name']}]({row['url']})")
                    st.write(f"**Description:** {row['description']}")
                    st.write(f"**Remote Testing:** {row['remote_testing']}")
                    st.write(f"**Adaptive Support:** {row['adaptive_support']}")
                    st.write(f"**Duration:** {row['duration']}")
                    st.progress(float(row["similarity"]))
            except Exception as e:
                st.error(f"Embedding failed: {e}")

if __name__ == "__main__":
    main()
