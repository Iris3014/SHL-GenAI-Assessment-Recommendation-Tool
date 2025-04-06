import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import pandas as pd
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure Streamlit page
st.set_page_config(page_title="SHL GenAI Assessment Recommender", layout="wide")

# Load local SentenceTransformer model
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer("./models/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Failed to load model from ./models: {e}")
        st.stop()

# Load SHL catalog dataset
@st.cache_data
def load_data():
    try:
        csv_path = os.path.join("dataset", "shl_catalog.csv")
        return pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

# Get local embeddings
def get_local_embedding(texts, model):
    return model.encode(texts)

# Get OpenAI embeddings
def get_openai_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Streamlit App Main
def main():
    st.title("SHL GenAI Assessment Recommendation Tool")
    st.markdown("Use this tool to get personalized SHL assessment recommendations based on your job description input.")

    # Sidebar for settings
    st.sidebar.title("Settings")
    use_openai = st.sidebar.checkbox("🔑 Use OpenAI Embeddings")
    openai.api_key = st.sidebar.text_input("OpenAI API Key", type="password") if use_openai else None

    # Job description input
    job_description = st.text_area("📄 Paste the Job Description here:")

    # Load catalog data
    df = load_data()
    st.subheader("📘 Available SHL Assessments")
    st.dataframe(df.drop(columns=["url"]), use_container_width=True)

    if st.button("🚀 Recommend Assessments"):
        if not job_description.strip():
            st.warning("⚠️ Please enter a job description.")
            return

        with st.spinner("🔍 Processing your input and finding best matches..."):
            corpus = df["description"].tolist()

            try:
                if use_openai and openai.api_key:
                    query_embedding = get_openai_embedding(job_description)
                    corpus_embeddings = [get_openai_embedding(desc) for desc in corpus]
                else:
                    model = load_model()
                    query_embedding = get_local_embedding([job_description], model)[0]
                    corpus_embeddings = get_local_embedding(corpus, model)
            except Exception as e:
                st.error(f"❌ Embedding failed: {e}")
                return

            # Compute cosine similarity
            similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
            df["similarity"] = similarities
            top_matches = df.sort_values("similarity", ascending=False).head(3)

            st.subheader("✅ Top Recommended Assessments")
            for _, row in top_matches.iterrows():
                st.markdown(f"### 🔗 [{row['name']}]({row['url']})")
                st.write(f"📝 **Description:** {row['description']}")
                st.write(f"📡 **Remote Testing:** {row['remote_testing']}")
                st.write(f"⚙️ **Adaptive Support:** {row['adaptive_support']}")
                st.write(f"⏱️ **Duration:** {row['duration']}")
                st.progress(float(row["similarity"]))

if __name__ == "__main__":
    main()
