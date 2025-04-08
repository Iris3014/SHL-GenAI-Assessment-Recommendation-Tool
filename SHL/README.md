 ---
title: SHL GenAI Assessment Recommender
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# SHL GenAI RAG Tool

A Streamlit-based web application for recommending SHL assessments based on job descriptions using Retrieval-Augmented Generation (RAG) principles.

## 🔍 Features

- Upload and search assessment data from a CSV file
- Input a job description or custom query
- Semantic search using OpenAI embeddings (optional)
- Ranks SHL assessments based on relevance
- Clean UI with expandable assessment details

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- SentenceTransformers
- OpenAI (optional)
- Scikit-learn (cosine similarity)

## 📁 Folder Structure
SHL/ 
├── app.py
├── datasets/ 
│ └── shl_catalog.csv
├── requirements.txt 
└── README.md

Note: Due to free-tier resource limits on Hugging Face Spaces, the deployed app link may temporarily be inactive.
However, the project works as expected — screenshots from the localhost run are provided in the screenshots/ folder to demonstrate the functionality.
Please refer to the codebase and the screenshots for a complete understanding.
