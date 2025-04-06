import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Avoid TensorFlow issues

import streamlit as st
import pandas as pd
import openai
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit config
st.set_page_config(page_title="SHL GenAI Assessment Recommender", layout="wide")

# Cache CSV load
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "dataset", "shl_catalog.csv")
    return pd.read_csv(csv_path)

# OpenAI Embedding
def get_openai_embedding(text, api_key):
    openai.api_key = api_key
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Main App
def main():
    st.title("SHL GenAI Assessment Recommendation Tool")
    st.markdown("This tool recommends relevant SHL assessments based on your job description using Retrieval-Augmented Generation (RAG).")

    # Sidebar Settings
    st.sidebar.title("Settings")
    use_openai = st.sidebar.checkbox("Use OpenAI Embeddings (Requires API Key)", value=True)
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    job_description = st.text_area("📄 Paste the Job Description here:")

    df = load_data()
    st.subheader("Available SHL Assessments")
    st.dataframe(df.drop(columns=["url"]))

    if st.button("Recommend Assessments"):
        if not job_description.strip():
            st.warning("Please enter a job description first.")
            return

        if use_openai and not api_key:
            st.error("Please provide your OpenAI API key in the sidebar.")
            return

        with st.spinner("Analyzing and generating recommendations..."):
            corpus = df["description"].tolist()

            try:
                query_embedding = get_openai_embedding(job_description, api_key)
                corpus_embeddings = [get_openai_embedding(desc, api_key) for desc in corpus]

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
                return

if __name__ == "__main__":
    main()
