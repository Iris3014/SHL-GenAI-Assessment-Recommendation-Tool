import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Prevent TensorFlow-related import issues

import streamlit as st
import pandas as pd
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="SHL GenAI Assessment Recommender", layout="wide")

@st.cache_data
def load_data():
    csv_path = "SHL/das/shl.csv"  # ✅ This works in deployment
    return pd.read_csv(csv_path)


# Load embedding model
@st.cache_resource
def load_local_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_local_embedding(texts, model):
    return model.encode(texts)

def get_openai_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Streamlit UI logic
def main():
    st.title("SHL GenAI Assessment Recommendation Tool")
    st.markdown("""
    This tool recommends relevant SHL assessments based on your job description using Retrieval-Augmented Generation (RAG).
    """)

    # Sidebar options
    st.sidebar.title("Settings")
    use_openai = st.sidebar.checkbox("Use OpenAI Embeddings (Needs API Key)")
    if use_openai:
        openai.api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    threshold = st.sidebar.slider("Minimum Similarity Threshold", 0.0, 1.0, 0.3, 0.01)

    # Input
    job_description = st.text_area("📝 Paste the Job Description here:")

    # Load data
    df = load_data()
    df["description"] = df["description"].fillna("")

    st.subheader("Available SHL Assessments")
    st.dataframe(df.drop(columns=["url"]))

    # Matching logic would go here (not shown in your image)

if __name__ == "__main__":
    main()
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Prevent TensorFlow-related import issues

import streamlit as st
import pandas as pd
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="SHL GenAI Assessment Recommender", layout="wide")

# Load CSV
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    csv_path = os.path.join(base_path, "datasets", "shl_catalog.csv")
    return pd.read_csv(csv_path)

# Load embedding model
@st.cache_resource
def load_local_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_local_embedding(texts, model):
    return model.encode(texts)

def get_openai_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Streamlit UI logic
def main():
    st.title("SHL GenAI Assessment Recommendation Tool")
    st.markdown("""
    This tool recommends relevant SHL assessments based on your job description using Retrieval-Augmented Generation (RAG).
    """)

    # Sidebar options
    st.sidebar.title("Settings")
    use_openai = st.sidebar.checkbox("Use OpenAI Embeddings (Needs API Key)")
    if use_openai:
        openai.api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    threshold = st.sidebar.slider("Minimum Similarity Threshold", 0.0, 1.0, 0.3, 0.01)

    # Input
    job_description = st.text_area("📝 Paste the Job Description here:")

    # Load data
    df = load_data()
    df["description"] = df["description"].fillna("")

    st.subheader("Available SHL Assessments")
    st.dataframe(df.drop(columns=["url"]))

    # Matching logic would go here (not shown in your image)

if __name__ == "__main__":
    main()
