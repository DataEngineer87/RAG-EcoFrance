import streamlit as st
import faiss
import numpy as np
import json
from huggingface_hub import InferenceClient

# --- Configuration ---
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
client = InferenceClient(token="TON_HUGGINGFACE_TOKEN")  # Remplace par ton token HF

# --- Charger l’index FAISS et les documents ---
index = faiss.read_index("Scripts/docs.index")  # Attention au chemin correct

# Charger JSON correctement
with open("Scripts/docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

# --- Fonctions ---
def embed_query(query: str):
    """
    Crée un embedding via Hugging Face Inference API (feature_extraction)
    Compatible avec Faiss.
    """
    resp = client.feature_extraction(model=embedding_model, inputs=query)
    emb_array = np.array(resp, dtype="float32").reshape(1, -1)
    return emb_array

def retrieve_context(query, k=4):
    """
    Récupère les k documents les plus proches via Faiss.
    """
    qv = embed_query(query)
    D, I = index.search(qv, k=k)
    results = [docs[idx] for idx in I[0]]
    return results

# --- Streamlit App ---
st.title("RAG EcoFrance")
question = st.text_input("Posez votre question :")
submit = st.button("Envoyer")
k = 4  # Nombre de documents à récupérer

if submit and question.strip():
    with st.spinner("🔎 Recherche dans l'index et génération de la réponse..."):
        retrieved = retrieve_context(question, k=k)
        st.subheader("📚 Contexte utilisé")
        for doc in retrieved:
            st.write(doc)
