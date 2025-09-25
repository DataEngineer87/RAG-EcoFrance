# app.py (Chatbot RAG avec Hugging Face API)
# Prérequis :
#  - docs.index et docs.json créés par build_index.py
#  - HF API key stockée dans clehug.env ou dans Secrets de Streamlit Cloud

import streamlit as st
import faiss
import numpy as np
import json
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # dossier du script
index_file = os.path.join(BASE_DIR, "docs.index")
docs_file = os.path.join(BASE_DIR, "docs.json")

# Modèles Hugging Face
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
llm_model = "mistralai/Mistral-7B-Instruct-v0.2"

k = 4

# === Charger le fichier clehug.env si présent (local) ===
dotenv_path = os.path.join(BASE_DIR, "clehug.env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

# === Récupérer la clé HF depuis l'environnement (Secrets ou .env) ===
hf_token = os.environ.get("HUGGINGFACE_API_KEY")
if not hf_token:
    st.error("❌ Clé Hugging Face API manquante. Ajoutez-la dans clehug.env ou dans les Secrets de Streamlit Cloud.")
    st.stop()

client = InferenceClient(token=hf_token)

# === Charger index et documents ===
@st.cache_resource
def load_index_and_docs():
    if not os.path.exists(index_file):
        st.error(f"❌ Fichier introuvable : {index_file}")
        st.stop()
    if not os.path.exists(docs_file):
        st.error(f"❌ Fichier introuvable : {docs_file}")
        st.stop()

    index = faiss.read_index(index_file)
    with open(docs_file, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs

index, docs = load_index_and_docs()

# === UI PRINCIPALE ===
st.set_page_config(page_title="Chatbot RAG", page_icon="🤖", layout="centered")

st.markdown(
    "<h3 style='text-align: center;'>🤖 CHATBOT RAG - INFORMATION SUR L'ÉCONOMIE FRANÇAISE</h3>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center;'>Que souhaitez-vous savoir sur l'économie française ?</h4>",
    unsafe_allow_html=True
)
st.divider()

question = st.text_input("💬 Entrez votre question :",_
