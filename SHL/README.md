---
title: SHL GenAI Assessment Recommender
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
---

This Streamlit app recommends SHL assessments based on a job description using local or OpenAI embeddings.

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

