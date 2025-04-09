import os
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
from fastapi.responses import HTMLResponse

# Initialize FastAPI app
app = FastAPI(title="SHL GenAI Recommender API")

# Root route
@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h2>✅ Welcome to the SHL GenAI Assessment Recommendation Tool API</h2>"

# Load CSV (make sure it's in the same directory as api.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "shl_catalog.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError("❌ shl_catalog.csv not found in the same directory as api.py!")

df = pd.read_csv(csv_path).fillna("")

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Pydantic response models
class Assessment(BaseModel):
    name: str
    description: str
    duration: str
    remote_testing: str
    adaptive_support: str

class RecommendationResponse(BaseModel):
    recommendations: List[Assessment]

# Embedding functions
def get_local_embedding(texts):
    return model.encode(texts)

def get_openai_embedding(text, api_key):
    openai.api_key = api_key
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

# Main endpoint
@app.get("/recommend", response_model=RecommendationResponse)
def recommend(
    job_description: str = Query(..., description="Job description text"),
    use_openai: bool = Query(False, description="Use OpenAI embeddings"),
    openai_key: str = Query(None, description="OpenAI API key if use_openai is true")
):
    if use_openai:
        if not openai_key:
            return {"recommendations": []}
        query_embedding = get_openai_embedding(job_description, openai_key)
        corpus_embeddings = [get_openai_embedding(desc, openai_key) for desc in df["description"]]
    else:
        query_embedding = get_local_embedding([job_description])[0]
        corpus_embeddings = get_local_embedding(df["description"].tolist())

    similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
    df["similarity"] = similarities
    top3 = df.sort_values("similarity", ascending=False).head(3)

    recommendations = [
        Assessment(
            name=row["name"],
            description=row["description"],
            duration=row["duration"],
            remote_testing=row["remote_testing"],
            adaptive_support=row["adaptive_support"]
        )
        for _, row in top3.iterrows()
    ]

    return RecommendationResponse(recommendations=recommendations)
