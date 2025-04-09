import os
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai

app = FastAPI(title="SHL GenAI Recommender API")

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "dataset", "shl_catalog.csv")
df = pd.read_csv(csv_path)

# Load local model once
model = SentenceTransformer("all-MiniLM-L6-v2")

class Assessment(BaseModel):
    name: str
    description: str
    duration: str
    remote_testing: str
    adaptive_support: str

class RecommendationResponse(BaseModel):
    recommendations: List[Assessment]

def get_local_embedding(texts):
    return model.encode(texts)

def get_openai_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

@app.get("/recommend", response_model=RecommendationResponse)
def recommend(
    job_description: str = Query(...),
    use_openai: bool = Query(False),
    openai_key: str = Query(None)
):
    if use_openai:
        if not openai_key:
            return {"error": "Missing OpenAI key."}
        openai.api_key = openai_key
        query_embedding = get_openai_embedding(job_description)
        corpus_embeddings = [get_openai_embedding(desc) for desc in df["description"]]
    else:
        query_embedding = get_local_embedding([job_description])[0]
        corpus_embeddings = get_local_embedding(df["description"].tolist())

    similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
    df["similarity"] = similarities
    top3 = df.sort_values("similarity", ascending=False).head(3)

    recommendations = [
        {
            "name": row["name"],
            "description": row["description"],
            "duration": row["duration"],
            "remote_testing": row["remote_testing"],
            "adaptive_support": row["adaptive_support"],
        }
        for _, row in top3.iterrows()
    ]

    return {"recommendations": recommendations}
